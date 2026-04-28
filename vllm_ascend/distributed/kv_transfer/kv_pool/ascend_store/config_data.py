from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import NewRequestData


BlockIdGroups = list[list[int]]


def normalize_block_id_groups(
    block_ids: tuple[list[int], ...] | list[list[int]] | list[int] | None,
) -> BlockIdGroups:
    if block_ids is None or len(block_ids) == 0:
        return []
    first_group = block_ids[0]
    if isinstance(first_group, (list, tuple)):
        return [list(group) for group in block_ids]  # type: ignore[arg-type]
    return [list(block_ids)]  # type: ignore[list-item]


def has_any_block_id(block_ids: tuple[list[int], ...] | list[list[int]] | list[int] | None) -> bool:
    return any(group for group in normalize_block_id_groups(block_ids))


@dataclass(frozen=True)
class CacheRegion:
    group_id: int
    base_addr: int
    block_len: int


@dataclass(frozen=True)
class TransferItem:
    start: int
    end: int
    key: PoolKey
    addrs: list[int]
    sizes: list[int]
    block_id: int
    block_hash_index: int
    group_id: int


# Parameters related to the key
@dataclass
class KeyMetadata:
    """name of the LLM model"""

    model_name: str
    """ worker id when running under a distributed setting """
    head_or_tp_rank: int
    """ Initialize the current prefill context model parallel rank """
    pcp_rank: int
    """ Initialize the current decode context model parallel rank """
    dcp_rank: int
    """ Initialize the current pipeline parallel rank """
    pp_rank: int


@dataclass(order=True)
class PoolKey:
    key_metadata: KeyMetadata
    chunk_hash: str
    group_id: int | None = field(default=None, kw_only=True)
    group_type: str | None = field(default=None, kw_only=True)

    def __hash__(self):
        if self.group_id is None:
            return hash(
                (
                    self.key_metadata.model_name,
                    self.key_metadata.head_or_tp_rank,
                    self.key_metadata.pcp_rank,
                    self.key_metadata.dcp_rank,
                    self.key_metadata.pp_rank,
                    self.chunk_hash,
                )
            )
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.group_id,
                self.group_type,
                self.chunk_hash,
            )
        )

    def to_string(self):
        if self.group_id is None:
            return (
                f"{self.key_metadata.model_name}"
                f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
                f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
                f"@pp_rank:{self.key_metadata.pp_rank}@{self.chunk_hash}"
            )
        group_suffix = "" if self.group_id is None else f"@group:{self.group_id}:{self.group_type}"
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}{group_suffix}@{self.chunk_hash}"
        )

    def split_layers(self, num_layers: int) -> list["LayerPoolKey"]:
        """Split the key into multiple keys for each layer"""
        keys = []
        for layer_id in range(num_layers):
            keys.append(
                LayerPoolKey(
                    self.key_metadata,
                    self.chunk_hash,
                    layer_id,
                    group_id=self.group_id,
                    group_type=self.group_type,
                )
            )
        return keys


@dataclass(order=True)
class LayerPoolKey(PoolKey):
    """A key for the layer cache engine"""

    layer_id: int

    def __hash__(self):
        if self.group_id is None:
            return hash(
                (
                    self.key_metadata.model_name,
                    self.key_metadata.head_or_tp_rank,
                    self.key_metadata.pcp_rank,
                    self.key_metadata.dcp_rank,
                    self.chunk_hash,
                    self.layer_id,
                )
            )
        return hash(
            (
                self.key_metadata.model_name,
                self.key_metadata.head_or_tp_rank,
                self.key_metadata.pcp_rank,
                self.key_metadata.dcp_rank,
                self.key_metadata.pp_rank,
                self.group_id,
                self.group_type,
                self.chunk_hash,
                self.layer_id,
            )
        )

    def to_string(self):
        if self.group_id is None:
            return (
                f"{self.key_metadata.model_name}"
                f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
                f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}@{self.chunk_hash}@{self.layer_id}"
            )
        group_suffix = "" if self.group_id is None else f"@group:{self.group_id}:{self.group_type}"
        return (
            f"{self.key_metadata.model_name}"
            f"@pcp{self.key_metadata.pcp_rank}@dcp{self.key_metadata.dcp_rank}"
            f"@head_or_tp_rank:{self.key_metadata.head_or_tp_rank}"
            f"@pp_rank:{self.key_metadata.pp_rank}{group_suffix}@{self.chunk_hash}@{self.layer_id}"
        )


class ChunkedTokenDatabase:
    def __init__(self, metadata: KeyMetadata, block_size: int, partitions: list[int] | None):
        self.metadata = metadata
        self.block_size = block_size
        self.kv_caches_base_addr: list[int] = []
        self.block_len: list[int] = []
        self.partitions = partitions
        self.group_types: list[str] = ["attention"]
        self.group_block_sizes: list[int] = [block_size]
        self.group_regions: dict[int, list[CacheRegion]] = {}

    def _make_key_by_hash(self, chunk_hash: str, layer_id: int | None = None, group_id: int | None = None):
        assert self.metadata is not None
        group_type = self.group_types[group_id] if group_id is not None and group_id < len(self.group_types) else None
        return PoolKey(
            self.metadata,
            chunk_hash,
            group_id=group_id if self._uses_group_namespace() else None,
            group_type=group_type if self._uses_group_namespace() else None,
        )

    def _uses_group_namespace(self) -> bool:
        return len(self.group_types) > 1

    def set_kv_caches_base_addr(self, kv_caches_base_addr: list[int]):
        self.kv_caches_base_addr = kv_caches_base_addr
        self._refresh_default_regions()

    def set_block_len(self, block_len: list[int]):
        self.block_len = block_len
        self._refresh_default_regions()

    def set_kv_cache_groups(self, group_types: list[str], group_block_sizes: list[int]):
        self.group_types = group_types or ["attention"]
        self.group_block_sizes = group_block_sizes or [self.block_size]
        self._refresh_default_regions()

    def set_kv_cache_regions(self, group_regions: dict[int, list[CacheRegion]]):
        self.group_regions = {group_id: list(regions) for group_id, regions in group_regions.items()}

    def _refresh_default_regions(self):
        if not self.kv_caches_base_addr or not self.block_len or self._uses_group_namespace():
            return
        length = len(self.block_len)
        self.group_regions = {
            0: [
                CacheRegion(0, base_addr, self.block_len[index % length])
                for index, base_addr in enumerate(self.kv_caches_base_addr)
            ]
        }

    def prepare_value(self, start: int, end: int, block_ids: tuple[list[int], ...] | list[list[int]] | list[int]):
        block_id_groups = normalize_block_id_groups(block_ids)
        if self._uses_group_namespace():
            block_hashes = [""] * (start // self.block_size + 1)
            items = self.iter_transfer_items(end, block_hashes, block_id_groups, start_token=start)
            addrs: list[int] = []
            sizes: list[int] = []
            block_id = -1
            for item in items:
                addrs.extend(item.addrs)
                sizes.extend(item.sizes)
                block_id = item.block_id
            return addrs, sizes, block_id
        addr_list = []
        size_list = []
        block_ids_0 = block_id_groups[0] if block_id_groups else []
        block_id = block_ids_0[start // self.block_size]
        length = len(self.block_len)
        for index, base_addr in enumerate(self.kv_caches_base_addr):
            addr = base_addr + block_id * self.block_len[index % length]
            size = int(self.block_len[index % length] / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list, block_id

    def prepare_value_layer(
        self,
        start: int,
        end: int,
        block_ids: tuple[list[int], ...] | list[list[int]] | list[int],
        layer_id: int,
    ):
        block_ids_0 = normalize_block_id_groups(block_ids)[0]
        block_id = block_ids_0[start // self.block_size]
        addr_list = []
        size_list = []
        length = len(self.block_len)
        for i in range(length):
            addr = self.kv_caches_base_addr[layer_id * length] + block_id * self.block_len[i]
            size = int(self.block_len[i] / self.block_size * (end - start))
            addr_list.append(addr)
            size_list.append(size)
        return addr_list, size_list

    def iter_lookup_keys(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        mask_num: int = 0,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        if not self._uses_group_namespace():
            yield from self.process_tokens(token_len, block_hashes, mask_num)
            return
        for start, end, key in self.process_tokens(token_len, block_hashes, mask_num):
            for group_id in range(len(self.group_types)):
                yield start, end, self._make_key_by_hash(key.chunk_hash, group_id=group_id)

    def iter_transfer_items(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        block_ids: tuple[list[int], ...] | list[list[int]] | list[int],
        mask_num: int = 0,
        start_token: int | None = None,
        latest_state_only: bool = False,
    ) -> Iterable[TransferItem]:
        block_id_groups = normalize_block_id_groups(block_ids)
        if not block_id_groups:
            return
        token_iter = self.process_tokens(token_len, block_hashes, mask_num)
        for hash_index, (start, end, base_key) in enumerate(token_iter):
            if start_token is not None and start != start_token:
                continue
            for group_id, group_block_ids in enumerate(block_id_groups):
                if not group_block_ids:
                    continue
                group_type = self.group_types[group_id] if group_id < len(self.group_types) else "attention"
                if latest_state_only and group_type == "gdn_attention" and end < token_len:
                    continue
                group_block_size = (
                    self.group_block_sizes[group_id] if group_id < len(self.group_block_sizes) else self.block_size
                )
                block_index = start // group_block_size
                if block_index >= len(group_block_ids):
                    if group_type == "gdn_attention" and len(group_block_ids) == 1:
                        block_index = 0
                    else:
                        continue
                block_id = group_block_ids[block_index]
                addrs = []
                sizes = []
                regions = self.group_regions.get(group_id, [])
                for region in regions:
                    addr = region.base_addr + block_id * region.block_len
                    if group_type == "gdn_attention":
                        size = region.block_len
                    else:
                        size = int(region.block_len / group_block_size * (end - start))
                    addrs.append(addr)
                    sizes.append(size)
                yield TransferItem(
                    start=start,
                    end=end,
                    key=self._make_key_by_hash(base_key.chunk_hash, group_id=group_id),
                    addrs=addrs,
                    sizes=sizes,
                    block_id=block_id,
                    block_hash_index=hash_index,
                    group_id=group_id,
                )

    def process_tokens(
        self,
        token_len: int,
        block_hashes: list[BlockHash] | list[str],
        mask_num: int = 0,
    ) -> Iterable[tuple[int, int, PoolKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param Union[torch.Tensor, List[int]] tokens: The tokens to process.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        if not block_hashes:
            return
        if not isinstance(block_hashes[0], str):
            block_hashes = [
                h.hex()  # type: ignore[union-attr]
                for h in block_hashes
            ]
        start_idx = 0
        for chunk_id, hash_val in enumerate(block_hashes):
            start_idx = chunk_id * self.block_size
            if start_idx >= token_len:
                break
            end_idx = min(start_idx + self.block_size, token_len)
            if start_idx < mask_num:
                continue
            else:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)

    def decode_adaptor_prefill_pp(self, key, addr, size):
        if self.partitions is None or len(self.partitions) == 1:
            return key, addr, size

        new_key = []
        new_addr = []
        new_size = []

        for i, (addr_list, size_list) in enumerate(zip(addr, size)):
            start = 0
            for j, part in enumerate(self.partitions):
                # part * 2 because addr and size contain both k and v
                end = len(addr_list) if j == len(self.partitions) - 1 else start + part * 2
                new_str = key[i].replace(  # type: ignore[attr-defined]
                    "@pp_rank:0", f"@pp_rank:{j}", 1
                )
                new_key.append(new_str)
                new_addr.append(addr_list[start:end])
                new_size.append(size_list[start:end])
                start = end
        return new_key, new_addr, new_size


# Parameters related to the connector metadata
@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in kvpool
    kvpool_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool

    token_len: int = 0


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    token_len: int

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: BlockIdGroups

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    # The token ids that has been scheduled so far
    # NOTE: This field will only be used when you enable kv-event
    token_ids: list[int] | None = None

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        unfolded_block_ids = normalize_block_id_groups(new_request.block_ids)

        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            token_len=num_tokens_to_compute,
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=0,
        )

    def update(
        self,
        new_block_ids: tuple[list[int], ...] | list[list[int]] | list[int],
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again
        """
        new_block_id_groups = normalize_block_id_groups(new_block_ids)
        while len(self.allocated_block_ids) < len(new_block_id_groups):
            self.allocated_block_ids.append([])
        for group_id, group_block_ids in enumerate(new_block_id_groups):
            self.allocated_block_ids[group_id].extend(group_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Number of tokens in this chunk
    token_len_chunk: int

    block_ids: BlockIdGroups

    block_hashes: list[BlockHash]

    can_save: bool | None = None
    # load_spec
    load_spec: LoadSpec | None = None

    is_last_chunk: bool | None = None

    current_event: torch.npu.Event | None = None

    # The following parameters are only used for kv event generation
    # TODO: add lora_request which used for gen lora_id/lora_name in kv event
    token_ids: list[int] | None = None
    original_block_size: int | None = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        load_spec: LoadSpec | None = None,
        skip_save: bool | None = False,
        block_hashes: list[BlockHash] | None = None,
        is_last_chunk: bool | None = None,
        discard_partial_chunks: bool = True,
        original_block_size: int | None = None,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM scheduler and AscendConnector.
                If context parallelism is enabled, block_size = block_size * pcp_size * dcp_size.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            skip_save (bool): whether to skip the save operation.
            discard_partial_chunks (bool): whether to discard partial chunks.
            original_block_size (int | None): the block size in vLLM worker. This is only used for kv event generation.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        if block_hashes is None:
            block_hashes = []
        input_token_len = tracker.token_len

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        chunk_boundary = cdiv(tracker.num_saved_tokens + 1, block_size) * block_size if discard_partial_chunks else 0
        # Calculate number of tokens to save based on discard_partial_chunks
        # setting
        num_tokens_to_save = (input_token_len // block_size * block_size) if discard_partial_chunks else input_token_len

        skip_save = skip_save or num_tokens_to_save < chunk_boundary
        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        # Get the token ids for kv event generation in kv_transfer
        token_ids = None
        if tracker.token_ids:
            token_ids = tracker.token_ids

        # # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens for request %s",
                load_spec.kvpool_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None
        logger.debug("request:%s, meta save spec:%s, meta load spec:%s", tracker.req_id, not skip_save, load_spec)
        return ReqMeta(
            req_id=tracker.req_id,
            token_len_chunk=num_tokens_to_save,
            block_ids=tracker.allocated_block_ids,
            can_save=not skip_save,
            load_spec=load_spec,
            block_hashes=block_hashes,
            is_last_chunk=is_last_chunk,
            token_ids=token_ids,
            original_block_size=original_block_size,
        )


class AscendConnectorMetadata(KVConnectorMetadata):
    def __init__(self, unfinished_request_ids, preempted_req_ids):
        self.requests = []
        self.unfinished_request_ids = unfinished_request_ids
        self.preempted_req_ids = preempted_req_ids

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


@dataclass
class LayerMultiBlockReqMeta:
    req_id: str
    keys: list[LayerPoolKey]
    starts: list[int]
    ends: list[int]
    block_ids: BlockIdGroups
    layer_id: int
    is_last_chunk: bool | None = True
    current_event: torch.npu.Event | None = None
