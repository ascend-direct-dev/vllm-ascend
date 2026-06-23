from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA, supports_hma
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import MooncakeLayerwiseConnector

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


class AscendMultiConnector(MultiConnector, SupportsHMA):
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        chosen_connector = self._requests_to_connector.get(request.request_id, -1)
        empty_blocks = blocks.new_empty()
        for i, c in enumerate(self._connectors):
            if i == chosen_connector or isinstance(c, MooncakeLayerwiseConnector):
                # Forward call to the chosen connector (if any).
                c.update_state_after_alloc(request, blocks, num_external_tokens)
            else:
                # Call with empty blocks for other connectors.
                c.update_state_after_alloc(request, empty_blocks, 0)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """HMA-aware variant of request_finished.

        ``block_ids`` holds the computed blocks organized per KV cache group
        (one inner list per group). It is aggregated across all sub-connectors:
        connectors that support HMA receive the grouped block ids, while
        connectors that do not are only valid when there is a single KV cache
        group and receive that group's flat block ids (matching the legacy
        request_finished path).
        """
        async_saves = 0
        kv_txfer_params: dict[str, Any] | None = None
        for c in self._connectors:
            if supports_hma(c):
                async_save, txfer_params = c.request_finished_all_groups(request, block_ids)
            else:
                assert len(block_ids) == 1, (
                    "HMA with multiple kv_cache_groups requires all sub-connectors to support HMA"
                )
                async_save, txfer_params = c.request_finished(request, block_ids[0])
            if async_save:
                async_saves += 1
            if txfer_params is not None:
                if kv_txfer_params is not None:
                    # TODO we can probably change this to merge the dicts here,
                    # checking for key clashes.
                    raise RuntimeError("Only one connector can produce KV transfer params")
                kv_txfer_params = txfer_params
        if async_saves > 1:
            self._extra_async_saves[request.request_id] = async_saves - 1

        # Clean up other state for this request.
        self._requests_to_connector.pop(request.request_id, None)

        return async_saves > 0, kv_txfer_params
