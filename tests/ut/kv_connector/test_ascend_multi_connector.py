import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

# fake mooncake.engine.TransferEngine so importing the layerwise connector
# (a dependency of AscendMultiConnector) does not require the real package.
fake_engine = types.ModuleType("mooncake.engine")
fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
sys.modules["mooncake.engine"] = fake_engine

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA, supports_hma  # noqa: E402

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import AscendMultiConnector  # noqa: E402


class _FakeHMAConnector(SupportsHMA):
    def __init__(self, ret):
        self.ret = ret
        self.all_groups_calls: list = []

    def request_finished_all_groups(self, request, block_ids):
        self.all_groups_calls.append(block_ids)
        return self.ret


class _FakeNonHMAConnector:
    def __init__(self, ret):
        self.ret = ret
        self.flat_calls: list = []

    def request_finished(self, request, block_ids):
        self.flat_calls.append(block_ids)
        return self.ret


def _bare_multi(connectors):
    multi = object.__new__(AscendMultiConnector)
    multi._connectors = connectors
    multi._extra_async_saves = {}
    multi._requests_to_connector = {}
    return multi


class TestAscendMultiConnectorHMA(unittest.TestCase):
    def test_supports_hma(self):
        self.assertTrue(supports_hma(AscendMultiConnector))

    def test_request_finished_all_groups_routes_by_capability(self):
        hma = _FakeHMAConnector((True, {"remote_block_ids": [[1], [2]]}))
        non_hma = _FakeNonHMAConnector((False, None))
        multi = _bare_multi([hma, non_hma])
        multi._requests_to_connector["r1"] = 0
        request = SimpleNamespace(request_id="r1")

        block_ids = ([1, 2, 3],)  # single group
        delay, params = multi.request_finished_all_groups(request, block_ids)

        self.assertTrue(delay)
        self.assertEqual(params, {"remote_block_ids": [[1], [2]]})
        # HMA connector gets the grouped block ids unchanged.
        self.assertEqual(hma.all_groups_calls, [([1, 2, 3],)])
        # Non-HMA connector gets the single group's flat block ids.
        self.assertEqual(non_hma.flat_calls, [[1, 2, 3]])
        # Request bookkeeping is cleaned up.
        self.assertNotIn("r1", multi._requests_to_connector)

    def test_non_hma_connector_with_multiple_groups_raises(self):
        non_hma = _FakeNonHMAConnector((False, None))
        multi = _bare_multi([non_hma])
        request = SimpleNamespace(request_id="r1")

        with self.assertRaises(AssertionError):
            multi.request_finished_all_groups(request, ([1, 2], [3]))

    def test_multiple_async_saves_tracked(self):
        hma1 = _FakeHMAConnector((True, None))
        hma2 = _FakeHMAConnector((True, None))
        multi = _bare_multi([hma1, hma2])
        request = SimpleNamespace(request_id="r1")

        delay, params = multi.request_finished_all_groups(request, ([1], [2]))

        self.assertTrue(delay)
        self.assertIsNone(params)
        # Two async saves -> one extra save tracked.
        self.assertEqual(multi._extra_async_saves["r1"], 1)

    def test_conflicting_transfer_params_raises(self):
        hma1 = _FakeHMAConnector((True, {"a": 1}))
        hma2 = _FakeHMAConnector((True, {"b": 2}))
        multi = _bare_multi([hma1, hma2])
        request = SimpleNamespace(request_id="r1")

        with self.assertRaises(RuntimeError):
            multi.request_finished_all_groups(request, ([1], [2]))

    def test_no_async_save_returns_false(self):
        non_hma = _FakeNonHMAConnector((False, None))
        multi = _bare_multi([non_hma])
        request = SimpleNamespace(request_id="r1")

        delay, params = multi.request_finished_all_groups(request, ([1, 2, 3],))

        self.assertFalse(delay)
        self.assertIsNone(params)


if __name__ == "__main__":
    unittest.main()
