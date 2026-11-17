"""Pin an existing host mapping for Intel GPU DMA.

The XPU counterpart of cudaHostRegister: Level Zero's zexDriverImportExternalPointer makes a host range that the
runtime did not allocate (here: the llmt-kvbroker pool file, mmap'd) usable for direct copies, as USM host memory
is. vLLM's pin_mmap_region() only knows CUDA, so on XPU the shared pool would be copied from pageable memory.
"""
import ctypes
import logging

logger = logging.getLogger("vllm.exl3xpu.hostreg")

_ze = None
_driver = None
_import = None
_release = None


def _init() -> bool:
    global _ze, _driver, _import, _release
    if _import is not None:
        return True
    try:
        _ze = ctypes.CDLL("libze_loader.so.1")
        _ze.zeInit(ctypes.c_uint32(0))
        n = ctypes.c_uint32(0)
        if _ze.zeDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
            return False
        drivers = (ctypes.c_void_p * n.value)()
        _ze.zeDriverGet(ctypes.byref(n), drivers)
        _driver = ctypes.c_void_p(drivers[0])
        fptr = ctypes.c_void_p()
        for name in ("zexDriverImportExternalPointer", "zexDriverReleaseImportedPointer"):
            r = _ze.zeDriverGetExtensionFunctionAddress(_driver, name.encode(), ctypes.byref(fptr))
            if r != 0 or not fptr.value:
                logger.warning("hostreg: %s unavailable (ze result 0x%x)", name, r)
                return False
            if name.startswith("zexDriverImport"):
                _import = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(fptr.value)
            else:
                _release = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(fptr.value)
        return True
    except OSError as e:
        logger.warning("hostreg: no Level Zero loader (%s)", e)
        return False


def register(ptr: int, size: int) -> bool:
    """Import [ptr, ptr+size) for direct GPU copies. False = not pinned (copies still work, slower)."""
    if not _init():
        return False
    r = _import(_driver, ctypes.c_void_p(ptr), ctypes.c_size_t(size))
    if r != 0:
        logger.warning("hostreg: zexDriverImportExternalPointer(%d B) failed: 0x%x", size, r)
        return False
    return True


def unregister(ptr: int) -> None:
    if _release is not None:
        _release(_driver, ctypes.c_void_p(ptr))
