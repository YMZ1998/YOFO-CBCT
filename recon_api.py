"""
Python wrapper for CT reconstruction API (reconAPI)

This module provides a Python interface to the CUDA-based CT reconstruction library.
It supports Filtered Back Projection (FBP) and Forward Projection (FP) operations.

Requirements:
    - numpy
    - The recon_tool shared library (DLL/SO)
    - CUDA runtime libraries
    - OpenSSL libraries
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ReconAPI:
    """
    Python wrapper for CT reconstruction C API

    Attributes:
        lib_path: Path to the shared library
        _lib: ctypes library handle
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the reconstruction API

        Args:
            lib_path: Path to the shared library. If None, will search in common locations.

        Raises:
            FileNotFoundError: If the library cannot be found
            OSError: If the library cannot be loaded
        """
        if lib_path is None:
            lib_path = self._find_library()

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found: {lib_path}")

        self.lib_path = str(Path(lib_path).resolve())
        self._dll_directory_handles = []

        if os.name == 'nt':
            lib_dir = str(Path(self.lib_path).parent)
            self._dll_directory_handles.append(os.add_dll_directory(lib_dir))

        try:
            self._lib = ctypes.CDLL(self.lib_path)
        except OSError as e:
            raise OSError(f"Failed to load library {self.lib_path}: {e}")

        # Define function signatures
        self._setup_function_signatures()

    def _find_library(self) -> str:
        """
        Search for the reconstruction library in common locations

        Returns:
            Path to the library

        Raises:
            FileNotFoundError: If library is not found
        """
        # Get the directory containing this script
        script_dir = Path(__file__).parent
        project_root = script_dir.parent

        # Common library names on different platforms
        if os.name == 'nt':  # Windows
            lib_names = ['YOFO_recon_tool.dll', 'recon_toold.dll']
        else:  # Linux/Unix
            lib_names = ['librecon_tool.so', 'librecon_toold.so']

        # Search paths
        search_paths = [
            script_dir,
            project_root / 'bin',
            project_root / 'build' / 'bin',
            project_root / 'debug',
            project_root / 'release',
            Path('.'),
        ]

        for search_path in search_paths:
            for lib_name in lib_names:
                lib_path = search_path / lib_name
                if lib_path.exists():
                    return str(lib_path.absolute())

        raise FileNotFoundError(
            f"Could not find reconstruction library. Searched: {search_paths}\n"
            f"Looking for: {lib_names}\n"
            f"Please build the library as a shared library (DLL/SO) or specify lib_path."
        )

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for all API functions"""

        # void getShape(unsigned int* pVolShape, unsigned int* pProjShape, const char* cfgFile)
        self._lib.getShape.argtypes = [
            ctypes.POINTER(ctypes.c_uint),  # pVolShape
            ctypes.POINTER(ctypes.c_uint),  # pProjShape
            ctypes.c_char_p  # cfgFile
        ]
        self._lib.getShape.restype = None

        # void FBP_API(float* pVol, const float* pProj, const char* cfgFile)
        self._lib.FBP_API.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pVol
            ctypes.POINTER(ctypes.c_float),  # pProj
            ctypes.c_char_p  # cfgFile
        ]
        self._lib.FBP_API.restype = None

        # void FP_API(float* pProj, const float* pVol, const char* cfgFile)
        self._lib.FP_API.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # pProj
            ctypes.POINTER(ctypes.c_float),  # pVol
            ctypes.c_char_p  # cfgFile
        ]
        self._lib.FP_API.restype = None

    @staticmethod
    def _encode_config_path(config_file: str) -> bytes:
        cfg_path = Path(config_file)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

        encoding = 'mbcs' if os.name == 'nt' else 'utf-8'
        return str(cfg_path.resolve()).encode(encoding)

    def get_shape(self, config_file: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get volume and projection data shapes from configuration file

        Args:
            config_file: Path to the encrypted configuration file (settings.bin)

        Returns:
            A tuple of (vol_shape, proj_shape) where:
                - vol_shape: (xNum, yNum, zNum) - volume dimensions
                - proj_shape: (uNum, vNum, noView) - projection dimensions

        Example:
            >>> api = ReconAPI()
            >>> vol_shape, proj_shape = api.get_shape("test_data/settings.bin")
            >>> print(f"Volume: {vol_shape}, Projection: {proj_shape}")
        """
        vol_shape = (ctypes.c_uint * 3)()
        proj_shape = (ctypes.c_uint * 3)()

        cfg_bytes = self._encode_config_path(config_file)
        self._lib.getShape(vol_shape, proj_shape, cfg_bytes)

        vol_tuple = (vol_shape[0], vol_shape[1], vol_shape[2])
        proj_tuple = (proj_shape[0], proj_shape[1], proj_shape[2])

        return vol_tuple, proj_tuple

    def fbp(self, proj_data: np.ndarray, config_file: str) -> np.ndarray:
        """
        Perform Filtered Back Projection (FBP) reconstruction

        Args:
            proj_data: Projection data as numpy array with shape (noView, vNum, uNum)
                      dtype should be float32
                      Memory layout: all pixels of projection 1, then projection 2, etc.
            config_file: Path to the encrypted configuration file

        Returns:
            Reconstructed volume data as numpy array with shape (xNum, yNum, zNum)
            dtype is float32

        Example:
            >>> api = ReconAPI()
            >>> proj = load_projections_from_raw_files("path/to/raw", noView, uNum, vNum)
            >>> volume = api.fbp(proj, "settings.bin")
        """
        # Ensure input is float32 and C-contiguous
        if proj_data.dtype != np.float32:
            proj_data = proj_data.astype(np.float32)

        if not proj_data.flags['C_CONTIGUOUS']:
            proj_data = np.ascontiguousarray(proj_data)

        # Get expected shapes
        vol_shape, proj_shape = self.get_shape(config_file)

        # Verify input shape - C++ API expects shape (noView, vNum, uNum) in memory
        # proj_shape from config is (uNum, vNum, noView)
        expected_proj_shape = (proj_shape[2], proj_shape[1], proj_shape[0])  # (noView, vNum, uNum)
        if proj_data.shape != expected_proj_shape:
            # Try to reshape if total elements match
            if proj_data.size == np.prod(proj_shape):
                proj_data = proj_data.reshape(expected_proj_shape)
            else:
                raise ValueError(
                    f"Projection data shape mismatch. "
                    f"Expected {expected_proj_shape}, got {proj_data.shape}"
                )

        # Allocate output volume
        vol_data = np.zeros((vol_shape[2],vol_shape[1],vol_shape[0]), dtype=np.float32)

        # Call C API
        cfg_bytes = self._encode_config_path(config_file)
        self._lib.FBP_API(
            vol_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            proj_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cfg_bytes
        )

        return vol_data

    def fp(self, vol_data: np.ndarray, config_file: str) -> np.ndarray:
        """
        Perform Forward Projection (FP)

        Args:
            vol_data: Volume data as numpy array with shape (xNum, yNum, zNum)
                     dtype should be float32
            config_file: Path to the encrypted configuration file

        Returns:
            Projection data as numpy array with shape (noView, vNum, uNum)
            dtype is float32
            Memory layout: all pixels of projection 1, then projection 2, etc.

        Example:
            >>> api = ReconAPI()
            >>> volume = np.fromfile("volume.raw", dtype=np.float32)
            >>> volume = volume.reshape((zNum, yNum, xNum))
            >>> proj = api.fp(volume, "settings.bin")
        """
        # Ensure input is float32 and C-contiguous
        if vol_data.dtype != np.float32:
            vol_data = vol_data.astype(np.float32)

        if not vol_data.flags['C_CONTIGUOUS']:
            vol_data = np.ascontiguousarray(vol_data)

        # Get expected shapes
        vol_shape, proj_shape = self.get_shape(config_file)

        # Verify input shape
        expected_vol_shape = (vol_shape[2], vol_shape[1], vol_shape[0])
        if vol_data.shape != expected_vol_shape:
            # Try to reshape if total elements match
            if vol_data.size == np.prod(expected_vol_shape):
                vol_data = vol_data.reshape(expected_vol_shape)
            else:
                raise ValueError(
                    f"Volume data shape mismatch. "
                    f"Expected {expected_vol_shape}, got {vol_data.shape}"
                )

        # Allocate output projection with shape (noView, vNum, uNum) to match C++ API
        # proj_shape from config is (uNum, vNum, noView)
        output_proj_shape = (proj_shape[2], proj_shape[1], proj_shape[0])  # (noView, vNum, uNum)
        proj_data = np.zeros(output_proj_shape, dtype=np.float32)

        # Call C API
        cfg_bytes = self._encode_config_path(config_file)
        self._lib.FP_API(
            proj_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vol_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cfg_bytes
        )

        return proj_data


def load_projections_from_raw_files(raw_dir: str, num_views: int,
                                    u_num: int, v_num: int) -> np.ndarray:
    """
    Load projection data from individual raw files

    Args:
        raw_dir: Directory containing numbered raw files (00001.raw, 00002.raw, ...)
        num_views: Number of projection views to load
        u_num: Number of detector pixels in u direction
        v_num: Number of detector pixels in v direction

    Returns:
        Projection data as numpy array with shape (num_views, v_num, u_num)
        Memory layout matches C++ API expectations: all pixels of projection 1,
        then all pixels of projection 2, etc.

    Example:
        >>> proj = load_projections_from_raw_files("test_data/processed_raw", 720, 640, 640)
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Projection directory not found: {raw_dir}")

    # IMPORTANT: Shape is (num_views, v_num, u_num) to match C++ API memory layout
    # Each projection's data is stored contiguously in memory
    proj_data = np.zeros((num_views, v_num, u_num), dtype=np.float32)

    for i in range(num_views):
        # Files are typically numbered as 00001.raw, 00002.raw, etc.
        filename = raw_dir / f"{i + 1:05d}.raw"

        if not filename.exists():
            raise FileNotFoundError(f"Projection file not found: {filename}")

        # Load single projection
        single_proj = np.fromfile(filename, dtype=np.float32)

        # Reshape to (v_num, u_num) and assign to i-th projection
        expected_size = u_num * v_num
        if single_proj.size != expected_size:
            raise ValueError(
                f"File {filename} has {single_proj.size} elements, "
                f"expected {expected_size}"
            )

        proj_data[i, :, :] = single_proj.reshape((v_num, u_num))

    return proj_data


def save_raw(data: np.ndarray, filename: str):
    """
    Save numpy array to raw binary file

    Args:
        data: Numpy array to save (will be converted to float32)
        filename: Output filename

    Example:
        >>> volume = api.fbp(proj, "settings.bin")
        >>> save_raw(volume, "output_volume.raw")
    """
    data_f32 = data.astype(np.float32)
    data_f32.tofile(filename)


def save_mhd(data: np.ndarray, filename: str, spacing=None, origin=None):
    """
    Save numpy array as a MetaImage .mhd header plus a .raw data file.

    The CT volume arrays in this wrapper use numpy shape (z, y, x), while
    MetaImage DimSize is written as (x, y, z).
    """
    mhd_path = Path(filename)
    if mhd_path.suffix.lower() != '.mhd':
        mhd_path = mhd_path.with_suffix('.mhd')

    mhd_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = mhd_path.with_suffix('.raw')

    data_f32 = np.ascontiguousarray(data.astype(np.float32))
    data_f32.tofile(raw_path)

    ndims = data_f32.ndim
    dim_size = list(reversed(data_f32.shape))
    spacing = spacing if spacing is not None else [1.0] * ndims
    origin = origin if origin is not None else [0.0] * ndims

    if len(spacing) != ndims:
        raise ValueError(f"spacing length must be {ndims}, got {len(spacing)}")
    if len(origin) != ndims:
        raise ValueError(f"origin length must be {ndims}, got {len(origin)}")

    header = [
        "ObjectType = Image",
        f"NDims = {ndims}",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        "CompressedData = False",
        f"DimSize = {' '.join(str(int(v)) for v in dim_size)}",
        f"ElementSpacing = {' '.join(str(float(v)) for v in spacing)}",
        f"Offset = {' '.join(str(float(v)) for v in origin)}",
        "ElementType = MET_FLOAT",
        f"ElementDataFile = {raw_path.name}",
        "",
    ]
    mhd_path.write_text("\n".join(header), encoding='ascii')


def load_raw(filename: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """
    Load raw binary file as numpy array

    Args:
        filename: Input filename
        shape: Shape to reshape the data
        dtype: Data type (default: float32)

    Returns:
        Numpy array with specified shape and dtype

    Example:
        >>> volume = load_raw("volume.raw", (512, 512, 512))
    """
    data = np.fromfile(filename, dtype=dtype)
    return data.reshape(shape)


if __name__ == "__main__":
    # Example usage
    print(" ReconAPI Python Wrapper")
    print()

    # Initialize API
    try:
        api = ReconAPI(r'YOFO_recon_tool.dll')
        print(f"✓ Loaded library: {api.lib_path}")
    except Exception as e:
        print(f"✗ Failed to load library: {e}")
        print("\nNote: You need to build the library as a shared library (DLL).")
        print("Modify CMakeLists.txt: change 'STATIC' to 'SHARED' in add_library()")
        exit(1)

    # Data path. Pass it as the first argument, or edit the fallback path below.
    data_path = sys.argv[1] if len(sys.argv) > 1 else r'E:\code\YOFO_重建工具\data\YOFO_No_Metal_001_Jirox_CT1613_00042_73078'
    if not os.path.isdir(data_path):
        print(f"\n✗ Data path not found: {data_path}")
        print("Usage: python recon_api.py <data_path>")
        exit(1)

    # Configuration file
    config_file = os.path.join(data_path, "YOFO_config.bin")
    raw_dir = os.path.join(data_path, "YOFO_raw")

    # Get shapes
    try:
        vol_shape, proj_shape = api.get_shape(config_file)
        print(f"\n✓ Configuration loaded:")
        print(f"  Volume shape: {vol_shape}")
        print(f"  Projection shape: {proj_shape}")
    except Exception as e:
        print(f"\n✗ Failed to read configuration: {e}")
        exit(1)
    proj = load_projections_from_raw_files(raw_dir, proj_shape[2], proj_shape[0], proj_shape[1])
    volume = api.fbp(proj, config_file)
    print(volume.shape)
    save_mhd(volume, 'reconstructed_volume.mhd')
    mask = volume > 1.0
    mask = mask.astype(np.float32)
    save_mhd(mask, os.path.join(data_path, 'vol_mask.mhd'))
    proj_mask = api.fp(mask, config_file)
    save_mhd(proj_mask, os.path.join(data_path, 'proj_mask.mhd'))
    proj_mask_path = os.path.join(data_path, 'proj_mask')
    os.makedirs(proj_mask_path, exist_ok=True)
    for i in range(proj_mask.shape[0]):
        save_raw(proj_mask[i, :, :], os.path.join(proj_mask_path, f'{i+1:05d}.raw'))

    # volume2 = api.fbp(proj_new, config_file)
    # save_raw(volume2, 'reconstructed_volume2.raw')
