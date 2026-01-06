# SPDX-License-Identifier: Apache-2.0
# Standard
from concurrent.futures import Future
from typing import Any, List, Optional, Sequence, Union
import asyncio
import concurrent
import os
import pickle
import threading
import time

# Third Party
import pymvfs
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryObj,
    MemoryObjMetadata,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_METADATA_CHUNK_SIZE = 4096  # 4KB
_DATA_CHUNK_SIZE = 2 * 1024 * 1024  # 2MB


class GismoStorageBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        metadata: LMCacheEngineMetadata,
        dst_device: str = "cuda",
    ):
        if dst_device.startswith("cuda") and torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.key_dict: dict[CacheEngineKey, MemoryObjMetadata] = dict()
        self.key_lock = threading.Lock()

        self.progress_set: set[CacheEngineKey] = set()
        self.progress_lock = threading.Lock()

        logger.info("Initializing Gismo Storage Backend")
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop
        self.metadata = metadata

        self.gismo_socket_path = config.extra_config.get(
            "gismo_socket_path", "dmo.daemon.sock.0"
        )
        self.gismo_directory = config.extra_config.get("gismo_directory", "/gismo")
        self.chunk_size = config.extra_config.get("gismo_chunk_size", _DATA_CHUNK_SIZE)

        self.client = pymvfs.Client()
        self.client.initialize_logging("/tmp/pymvfs_client.log", 200, 5, "Info")
        self.client.connect(self.gismo_socket_path, warmup=False)

        # read opened file descriptors
        self.fds_lock = threading.Lock()
        self.open_fds: dict[str, pymvfs.File] = dict()
        self.threadpool = concurrent.futures.ThreadPoolExecutor()

        # Ensure the directory exists
        self.client.mkdir(self.gismo_directory, 0o777)

        self.lock = threading.Lock()
        logger.info(
            "Gismo Storage Backend initialized at directory: "
            + f"{self.gismo_directory} with socket: {self.gismo_socket_path}"
            ", device: " + self.dst_device + f", chunk size: {self.chunk_size} bytes"
        )

    def __str__(self):
        return "GismoStorageBackend"

    def __del__(self):
        self.close()

    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(
            self.gismo_directory,
            key.model_name.replace(
                "/", "_"
            ),  # replace '/' to avoid directory issues in Gismo
            str(key.world_size),
            str(key.worker_id),
            str(abs(key.chunk_hash)),
        )

    def _exists_in_gismo(self, key: CacheEngineKey) -> bool:
        key_str = self._key_to_path(key) + _METADATA_FILE_SUFFIX
        try:
            return self.client.exists(key_str)
        except Exception as e:
            logger.debug(f"File not found in Gismo storage: {key_str}, exception: {e}")
            return False

    def _exists_in_memory_cache(self, key: CacheEngineKey) -> bool:
        if self.exists_in_put_tasks(key):
            return False
        with self.key_lock:
            return key in self.key_dict

    def _batch_contains_in_gismo(self, keys: Sequence[CacheEngineKey]) -> int:
        key_strs = [self._key_to_path(key) + _METADATA_FILE_SUFFIX for key in keys]
        try:
            return self.client.contains(key_strs)
        except Exception as e:
            logger.debug(f"Batch file existence check failed in Gismo storage: {e}")
            return 0

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # Check if the key exists in the backend from memory cache
        if self._exists_in_memory_cache(key):
            return True
        # Fallback to check existence in Gismo storage
        return self._exists_in_gismo(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.progress_lock:
            return key in self.progress_set

    def _put_fd_in_cache(self, path: str, fd: pymvfs.File) -> None:
        with self.fds_lock:
            self.open_fds[path] = fd

    def _get_fd_from_cache(self, path: str) -> Optional[pymvfs.File]:
        with self.fds_lock:
            return self.open_fds.get(path)

    def _remove_fd_from_cache(self, path: str) -> None:
        with self.fds_lock:
            if path in self.open_fds:
                self.open_fds[path].close()
                del self.open_fds[path]

    def _pack_metadata(self, mem_obj: MemoryObj) -> bytes:
        return pickle.dumps(mem_obj.meta)

    def _unpack_metadata(self, data: bytes) -> MemoryObjMetadata:
        return pickle.loads(data)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> Union[List[Future], None]:
        with self.progress_lock:
            for key in keys:
                self.progress_set.add(key)
        return self.write_objects_to_gismo(keys, objs)

    def _write_gismo_meta_file(
        self, key: CacheEngineKey, fpath: str, mem_obj: MemoryObj
    ) -> None:
        metadata_fpath = fpath + _METADATA_FILE_SUFFIX
        try:
            metadata_bytes = self._pack_metadata(mem_obj)
            file_option = pymvfs.FileOption.default()
            file_option.chunk_size = _METADATA_CHUNK_SIZE
            file_option.flags |= pymvfs.MVFS_FILE_ALLOW_REWRITE
            self.client.put(metadata_fpath, metadata_bytes, 0, 0, file_option)
            logger.debug(f"Wrote metadata to {metadata_fpath}")
        except Exception as e:
            logger.error(f"Error writing metadata to {metadata_fpath}: {e}")
            self._remove_fd_from_cache(fpath)

    def _write_gismo_file(self, key: CacheEngineKey, mem_obj: MemoryObj) -> None:
        fpath = self._key_to_path(key)
        file_option = pymvfs.FileOption.default()
        file_option.flags |= pymvfs.MVFS_FILE_ALLOW_REWRITE
        file_option.chunk_size = self.chunk_size
        fd = self.client.open(
            fpath, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o666, file_option
        )
        self._put_fd_in_cache(fpath, fd)
        logger.debug(f"Opened and cached file descriptor for {fpath}")
        try:
            fd.fallocate(0, 0, len(mem_obj.byte_array))
        except Exception as e:
            logger.error(f"fallocate exception for {fpath}: {e}")
            self._remove_fd_from_cache(fpath)
            mem_obj.ref_count_down()
            return
        chunk_size = 64 * self.chunk_size  # 64 chunks of 2MB = 128MB
        offset = 0
        data = mem_obj.byte_array
        vram_op = 0  # default to HOST_TO_HOST
        if mem_obj.tensor is not None and mem_obj.tensor.is_cuda:
            vram_op = 2  # DEVICE_TO_HOST
        while offset < len(data):
            write_size = min(chunk_size, len(data) - offset)
            chunk = data[offset : offset + write_size]
            try:
                fd.write_vram(chunk, offset, vram_op)
            except Exception as e:
                logger.error(f"Write exception for {fpath} at offset {offset}: {e}")
                self._remove_fd_from_cache(fpath)
                mem_obj.ref_count_down()
                return
            offset += write_size
        logger.debug(
            f"Wrote data to {fpath} in chunks of {offset} bytes with op {vram_op}"
        )
        # then write metadata
        self._write_gismo_meta_file(key, fpath, mem_obj)
        mem_obj.ref_count_down()
        with self.progress_lock:
            self.progress_set.discard(key)

    def _read_gismo_file(
        self,
        key: CacheEngineKey,
        metadata: MemoryObjMetadata,
        results: dict[CacheEngineKey, MemoryObj],
    ) -> None:
        dtype = metadata.dtype
        shape = metadata.shape
        fmt = metadata.fmt
        assert dtype is not None
        assert shape is not None
        assert fmt is not None
        mem_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
        if mem_obj is None:
            logger.error("Memory allocation failed during loading.")
            return
        vram_op = 0  # default to HOST_TO_HOST
        fpath = self._key_to_path(key)
        if mem_obj.tensor is not None and mem_obj.tensor.is_cuda:
            vram_op = 1  # HOST_TO_DEVICE
        try:
            fd = self._get_fd_from_cache(fpath)
            if fd is None:
                logger.info(f"No cached fd found for {fpath}, open directly")
                file_option = pymvfs.FileOption.default()
                fd = self.client.open(fpath, os.O_RDONLY, 0o666, file_option)
                self._put_fd_in_cache(fpath, fd)

            try:
                fd.readinto_vram(mem_obj.byte_array, 0, vram_op)
            except Exception as e:
                logger.error(f"Read exception for {fpath}: {e}")
                self._remove_fd_from_cache(fpath)
                mem_obj.ref_count_down()
                return
            logger.debug(f"Appended memory object for {fpath} from cached fd")
            results[key] = mem_obj
        except Exception as e:
            logger.error(f"Error: {e}")
            mem_obj.ref_count_down()

    def _read_metadata(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObjMetadata]:
        key_str = self._key_to_path(key)
        try:
            metadata_bytes = self.client.get(
                key_str + _METADATA_FILE_SUFFIX, _METADATA_CHUNK_SIZE, 0
            )
            if metadata_bytes is None:
                logger.warning(f"Metadata not found for {key}")
                return None
            metadata = self._unpack_metadata(metadata_bytes)
            return metadata
        except Exception as e:
            logger.warning(f"Failed to read metadata for {key}: {e}")
            return None

    def _read_metadata_with_retry(
        self,
        key: CacheEngineKey,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Optional[MemoryObjMetadata]:
        # During test, we found that sometimes reading metadata may arrive
        # Before the metadata file is fully written. So we add retry logic here.
        for attempt in range(max_retries):
            metadata = self._read_metadata(key)
            if metadata is not None:
                return metadata
            logger.warning(
                f"Retrying to read metadata for {key}, attempt {attempt + 1}"
            )
            asyncio.run_coroutine_threadsafe(asyncio.sleep(delay), self.loop).result()
        logger.error(f"Failed to read metadata for {key} after {max_retries} attempts")
        return None

    def _read_metadata_and_file(
        self,
        key: CacheEngineKey,
        results: dict[CacheEngineKey, MemoryObj],
    ):
        try:
            metadata = self._read_metadata_with_retry(key, max_retries=3, delay=0.1)
            if metadata is None:
                logger.error(f"Metadata not found for {key}")
                return
            # Store metadata in key_dict
            with self.key_lock:
                self.key_dict[key] = metadata
            self._read_gismo_file(key, metadata, results)
        except Exception as e:
            logger.error(f"Error reading metadata and file for {key}: {e}")
            return

    def write_objects_to_gismo(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
    ) -> List[Future]:
        futures = []
        for key, obj in zip(keys, objs, strict=False):
            obj.ref_count_up()
            with self.key_lock:
                assert key not in self.key_dict
                self.key_dict[key] = obj.meta
            f = self.threadpool.submit(self._write_gismo_file, key, obj)
            futures.append(f)
        return futures

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        timeout = 10  # seconds
        start_time = time.time()
        while key in self.progress_set:
            if time.time() - start_time > timeout:
                raise TimeoutError("Task timed out")
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.1), self.loop).result()

        obj_list = self.read_objects_from_gismo([key])
        if obj_list is None or len(obj_list) == 0:
            return None
        return obj_list[key]

    def read_objects_from_gismo(
        self,
        keys: Sequence[CacheEngineKey],
    ) -> dict[CacheEngineKey, MemoryObj]:
        results: dict[CacheEngineKey, MemoryObj] = {}
        futures = []
        if len(keys) == 0:
            return results
        metadata: MemoryObjMetadata | None = None
        if len(keys) == 1:
            with self.key_lock:
                metadata = self.key_dict.get(keys[0])
            if metadata is None:
                self._read_metadata_and_file(keys[0], results)
            else:
                self._read_gismo_file(keys[0], metadata, results)
            return results
        for key in keys:
            f: Future
            with self.key_lock:
                metadata = self.key_dict.get(key)
            # It means the key does not exist in cache
            # So we need to try read from Gismo storage
            if metadata is None:
                f = self.threadpool.submit(self._read_metadata_and_file, key, results)
            else:
                f = self.threadpool.submit(
                    self._read_gismo_file, key, metadata, results
                )
            futures.append(f)
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.ALL_COMPLETED
        )
        if len(done) != len(futures):
            logger.error("Some read tasks did not complete successfully")
        return results

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return self.batched_contains(keys, pin)

    def batched_contains(self, keys, pin=False) -> int:
        count = 0
        not_in_cache_keys = []
        for key in keys:
            if self._exists_in_memory_cache(key):
                count += 1
            else:
                not_in_cache_keys.append(key)
        # Fallback to check existence in Gismo storage
        if len(not_in_cache_keys) > 0:
            count += self._batch_contains_in_gismo(not_in_cache_keys)
        return count

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        obj_list = self.read_objects_from_gismo(keys)
        ret: list[MemoryObj] = []
        for key in keys:
            if key in obj_list:
                ret.append(obj_list[key])
            else:
                logger.warning(f"Key {key} not found in Gismo storage backend")
        return ret

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        obj_list = self.read_objects_from_gismo(keys)
        ret: List[Optional[MemoryObj]] = []
        for key in keys:
            if key in obj_list:
                ret.append(obj_list[key])
            else:
                logger.warning(f"Key {key} not found in Gismo storage backend")
                ret.append(None)
        return ret

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        with self.key_lock:
            if key in self.key_dict and not self.exists_in_put_tasks(key):
                del self.key_dict[key]

                path = self._key_to_path(key)
                try:
                    self._remove_fd_from_cache(path)
                    self.client.unlink(path)
                    self.client.unlink(path + _METADATA_FILE_SUFFIX)
                    logger.debug(
                        f"Removed file descriptor and unlinked file for {path}"
                    )
                    return True
                except Exception:
                    return False
            else:
                return False

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(
        self,
    ) -> None:
        # close all opened file descriptors
        logger.info("Closing all opened file descriptors")
        with self.fds_lock:
            for fd in self.open_fds.values():
                fd.close()
            self.open_fds.clear()
        self.client.disconnect()
        logger.info("GismoStorageBackend closed")
