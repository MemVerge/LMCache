# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import threading

# Third Party
import pytest
import torch

pytest.importorskip("pymvfs", reason="pymvfs package is required for gismo tests")

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.gismo_storage_backend import GismoStorageBackend
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger("GismoStorageBackend")


def create_key(chunk_hash: str):
    return CacheEngineKey(
        fmt="MemoryFormat.KV_2LTD",
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash=int(chunk_hash, base=16),
        dtype=torch.bfloat16,
    )


def run(
    config: LMCacheEngineConfig,
    shape: torch.Size,
    dtype: torch.dtype,
    device: str = "cuda",
):
    BACKEND_NAME = "GismoStorageBackend"
    keys = []
    objs = []
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406d268")
    )
    bad_key = create_key("deadbeefdeadbeef")

    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        metadata = LMCacheEngineMetadata(
            model_name="Llama-3.1-70B-Instruct",
            world_size=0,
            worker_id=0,
            fmt="MemoryFormat.KV_2LTD",
            kv_dtype=dtype,
            kv_shape=shape,
            use_mla=False,
        )
        dst_device = device
        logger.info(f"Destination device for testing: {dst_device}")
        backends = CreateStorageBackends(
            config=config,
            metadata=metadata,
            loop=thread_loop,
            dst_device=dst_device,
        )
        assert len(backends) == 2  # GismoBackend + LocalCPUBackend
        assert BACKEND_NAME in backends

        gismo_storage_backend = backends[BACKEND_NAME]
        assert isinstance(gismo_storage_backend, GismoStorageBackend)
        assert gismo_storage_backend is not None

        local_cpu_backend = backends["LocalCPUBackend"]
        assert isinstance(local_cpu_backend, LocalCPUBackend)
        assert local_cpu_backend is not None

        for key in keys:
            assert not gismo_storage_backend.contains(key, False)
            assert not gismo_storage_backend.exists_in_put_tasks(key)

            obj = local_cpu_backend.allocate(shapes=shape, dtypes=dtype)
            assert obj is not None
            assert obj.tensor is not None
            assert not obj.tensor.is_cuda
            objs.append(obj)

        # small tensor changes for data validation
        objs[0].tensor[100, 200] = 1e-3
        objs[0].tensor[200, 100] = 1e-4

        objs[1].tensor[300, 400] = 1e-2
        objs[1].tensor[400, 300] = 1e-5

        gismo_storage_backend.batched_submit_put_task(keys, objs)

        for key, obj in zip(keys, objs, strict=False):
            returned_memory_obj = gismo_storage_backend.get_blocking(key)
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            if device.startswith("cuda"):
                assert (
                    returned_memory_obj.tensor is not None
                    and returned_memory_obj.tensor.is_cuda
                )
                returned_cpu_tensor = returned_memory_obj.tensor.to("cpu")
                assert (
                    returned_cpu_tensor is not None and not returned_cpu_tensor.is_cuda
                )
                assert torch.equal(returned_cpu_tensor, obj.tensor)
            else:
                assert (
                    returned_memory_obj.tensor is not None
                    and not returned_memory_obj.tensor.is_cuda
                )
                assert returned_memory_obj.metadata.address != obj.metadata.address
                assert torch.equal(returned_memory_obj.tensor, obj.tensor)

        obj_list = asyncio.run(
            gismo_storage_backend.batched_get_non_blocking(lookup_id="test", keys=keys)
        )
        count = gismo_storage_backend.batched_contains(keys)
        assert count == len(keys)

        for i, obj in enumerate(objs):
            returned_memory_obj = obj_list[i]
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            if device.startswith("cuda"):
                assert (
                    returned_memory_obj.tensor is not None
                    and returned_memory_obj.tensor.is_cuda
                )
                returned_cpu_tensor = returned_memory_obj.tensor.to("cpu")
                assert (
                    returned_cpu_tensor is not None and not returned_cpu_tensor.is_cuda
                )
                assert torch.equal(returned_cpu_tensor, obj.tensor)
            else:
                assert (
                    returned_memory_obj.tensor is not None
                    and not returned_memory_obj.tensor.is_cuda
                )
                assert returned_memory_obj.metadata.address != obj.metadata.address
                assert torch.equal(returned_memory_obj.tensor, obj.tensor)

        bad_obj = gismo_storage_backend.get_blocking(bad_key)
        assert bad_obj is None
        gismo_storage_backend.batched_remove(keys)
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()


@pytest.mark.no_shared_allocator
def test_gismo_backend():
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/gismo.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([2048, 2048])
    logger.info("Running GismoStorageBackend test on CPU")
    run(config, shape, dtype, "cpu")
