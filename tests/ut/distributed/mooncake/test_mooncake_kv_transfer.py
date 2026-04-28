import threading
import unittest
from types import SimpleNamespace

import torch

if not hasattr(torch, "npu"):
    torch.npu = SimpleNamespace(Event=object)  # type: ignore[attr-defined]

from vllm.distributed.kv_transfer.kv_connector.v1.base import supports_hma
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.config_data import (
    CacheRegion,
    ChunkedTokenDatabase,
    KeyMetadata,
    LayerMultiBlockReqMeta,
    ReqMeta,
    TransferItem,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerSendingThread,
    KVCacheStoreSendingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector


class _FakeKey:
    def __init__(self, value: str):
        self._value = value

    def to_string(self) -> str:
        return self._value


class _FakeStore:
    def __init__(self, exists_result: list[int]):
        self.exists_result = exists_result
        self.put_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []

    def set_device(self):
        return None

    def exists(self, keys: list[str]) -> list[int]:
        # Return exact number of states for requested keys.
        return self.exists_result[: len(keys)]

    def put(self, keys, addrs, sizes):
        self.put_calls.append((list(keys), list(addrs), list(sizes)))


class _FakeTokenDatabase:
    def process_tokens(self, token_len, block_hashes):
        for i, _ in enumerate(block_hashes):
            yield i * 16, (i + 1) * 16, _FakeKey(f"k{i}")

    def prepare_value(self, start, end, block_ids):
        block_id = start // 16
        return [1000 + block_id], [end - start], block_id

    def iter_transfer_items(self, token_len, block_hashes, block_ids, mask_num=0):
        for index, (start, end, key) in enumerate(self.process_tokens(token_len, block_hashes)):
            if start < mask_num:
                continue
            addrs, sizes, block_id = self.prepare_value(start, end, block_ids)
            yield TransferItem(start, end, key, addrs, sizes, block_id, index, 0)

    def prepare_value_layer(self, start, end, block_ids, layer_id):
        block_id = start // 16
        return [2000 + layer_id * 100 + block_id], [end - start]


class TestKVTransferMissingKeyPut(unittest.TestCase):
    def test_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            kv_role="kv_producer",
            ready_event=threading.Event(),
            enable_kv_event=False,
        )

        req_meta = ReqMeta(
            req_id="req-1",
            token_len_chunk=64,
            block_ids=[0, 1, 2, 3],
            block_hashes=[b"h0", b"h1", b"h2", b"h3"],  # type: ignore[arg-type]
            current_event=None,
        )
        thread.add_stored_request("req-1")
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["k1", "k3"])
        self.assertEqual(put_addrs, [[1001], [1003]])
        self.assertEqual(put_sizes, [[16], [16]])

    def test_layer_sending_thread_only_puts_missing_keys(self):
        store = _FakeStore(exists_result=[1, 0, 1, 0])
        token_db = _FakeTokenDatabase()
        thread = KVCacheStoreLayerSendingThread(
            m_store=store,
            token_database=token_db,
            block_size=16,
            tp_rank=0,
            dcp_size=1,
            put_step=1,
            ready_event=threading.Event(),
            num_layers=2,
            enable_kv_event=False,
        )

        req_meta = LayerMultiBlockReqMeta(
            req_id="req-2",
            keys=[_FakeKey("k0"), _FakeKey("k1"), _FakeKey("k2"), _FakeKey("k3")],  # type: ignore[arg-type]
            starts=[0, 16, 32, 48],
            ends=[16, 32, 48, 64],
            block_ids=[0, 1, 2, 3],
            layer_id=1,
            is_last_chunk=False,
            current_event=None,
        )
        thread.request_queue.put(req_meta)
        thread._handle_request(req_meta)

        self.assertEqual(len(store.put_calls), 1)
        put_keys, put_addrs, put_sizes = store.put_calls[0]
        self.assertEqual(put_keys, ["k1", "k3"])
        self.assertEqual(put_addrs, [[2101], [2103]])
        self.assertEqual(put_sizes, [[16], [16]])


class TestChunkedTokenDatabaseHMA(unittest.TestCase):
    def _make_db(self):
        db = ChunkedTokenDatabase(
            KeyMetadata("qwen3_6", head_or_tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
            block_size=16,
            partitions=None,
        )
        db.set_kv_cache_groups(["attention", "gdn_attention"], [16, 16])
        db.set_kv_cache_regions(
            {
                0: [CacheRegion(group_id=0, base_addr=1000, block_len=160)],
                1: [
                    CacheRegion(group_id=1, base_addr=2000, block_len=320),
                    CacheRegion(group_id=1, base_addr=3000, block_len=640),
                ],
            }
        )
        return db

    def test_hma_keys_use_group_namespace(self):
        db = self._make_db()
        items = list(db.iter_transfer_items(16, ["hash0"], [[3], [7]]))

        self.assertEqual([item.key.to_string() for item in items], [
            "qwen3_6@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@group:0:attention@hash0",
            "qwen3_6@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@group:1:gdn_attention@hash0",
        ])

    def test_linear_group_transfers_full_state_without_token_clipping(self):
        db = self._make_db()
        items = list(db.iter_transfer_items(8, ["hash0"], [[3], [7]]))
        attn_item, linear_item = items

        self.assertEqual(attn_item.sizes, [80])
        self.assertEqual(linear_item.sizes, [320, 640])
        self.assertEqual(linear_item.addrs, [2000 + 7 * 320, 3000 + 7 * 640])

    def test_linear_group_single_state_block_is_reused_for_later_chunks(self):
        db = self._make_db()
        items = list(db.iter_transfer_items(32, ["hash0", "hash1"], [[3, 4], [7]]))
        linear_items = [item for item in items if item.group_id == 1]

        self.assertEqual([item.block_id for item in linear_items], [7, 7])
        self.assertEqual([item.sizes for item in linear_items], [[320, 640], [320, 640]])

    def test_linear_load_uses_latest_state_only(self):
        db = self._make_db()
        items = list(db.iter_transfer_items(32, ["hash0", "hash1"], [[3, 4], [7]], latest_state_only=True))
        linear_items = [item for item in items if item.group_id == 1]

        self.assertEqual([item.key.chunk_hash for item in linear_items], ["hash1"])

    def test_single_group_keeps_legacy_namespace_and_prepare_value(self):
        db = ChunkedTokenDatabase(
            KeyMetadata("legacy", head_or_tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0),
            block_size=16,
            partitions=None,
        )
        db.set_kv_caches_base_addr([1000])
        db.set_block_len([160])
        _, _, key = next(db.process_tokens(16, ["hash0"]))

        self.assertEqual(key.to_string(), "legacy@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@hash0")
        self.assertEqual(db.prepare_value(0, 16, [2]), ([1320], [160], 2))


class TestAscendStoreHMAScheduler(unittest.TestCase):
    def test_connector_declares_hma_support(self):
        self.assertTrue(supports_hma(AscendStoreConnector))

    def test_request_finished_all_groups_delays_on_any_group_blocks(self):
        scheduler = object.__new__(KVPoolScheduler)
        scheduler.kv_role = "kv_producer"
        scheduler.consumer_is_to_put = False
        scheduler._request_trackers = {}
        request = SimpleNamespace(request_id="req-hma")

        delay, params = scheduler.request_finished_all_groups(request, ([], [7]))

        self.assertTrue(delay)
        self.assertIsNone(params)

    def test_request_finished_legacy_single_group_unchanged(self):
        scheduler = object.__new__(KVPoolScheduler)
        scheduler.kv_role = "kv_producer"
        scheduler.consumer_is_to_put = False
        scheduler._request_trackers = {}
        request = SimpleNamespace(request_id="req-legacy")

        delay, params = scheduler.request_finished(request, [1, 2])

        self.assertTrue(delay)
        self.assertIsNone(params)

    def test_layerwise_hma_fails_fast(self):
        kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])

        with self.assertRaisesRegex(ValueError, "use_layerwise=true with HMA"):
            KVPoolScheduler(SimpleNamespace(), use_layerwise=True, kv_cache_config=kv_cache_config)


if __name__ == "__main__":
    unittest.main()
