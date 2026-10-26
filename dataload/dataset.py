"""Paired TIFF inputs for self-supervised pansharpening."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_image(path, normalization="minmax", max_value=1.0):
    """Read TIFF as float32 CHW, preserving every spectral channel."""
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        array = series.asarray()
        axes = series.axes
    if array.ndim == 2:
        array = array[None, ...]
    elif array.ndim == 3:
        if axes.endswith("S") or axes.endswith("C"):
            array = np.moveaxis(array, -1, 0)
        elif not axes.endswith("YX"):
            raise ValueError(f"Unsupported TIFF axes {axes!r} in {path}; use CYX or YXS.")
    else:
        raise ValueError(f"Expected a 2D/3D TIFF in {path}, got {array.shape}.")
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite pixels in {path}.")
    if normalization == "minmax":
        low = array.min(axis=(1, 2), keepdims=True)
        span = array.max(axis=(1, 2), keepdims=True) - low
        array = (array - low) / np.maximum(span, np.finfo(np.float32).eps)
    elif normalization == "scale":
        if not np.isfinite(max_value) or max_value <= 0:
            raise ValueError("max_value must be finite and positive.")
        array = array / max_value
    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization}")
    return torch.from_numpy(np.ascontiguousarray(array))


class FusionDataset(Dataset):
    """Match <sample_id>_{ms,pan,mul,pri}.tif by their complete sample ID.

    Default folders are MS, PAN, UPMS, PRIOR. Folder names are configurable
    to read existing datasets without moving their images.
    """

    def __init__(self, root, ms_channels=4, ms_dir="MS", pan_dir="PAN",
                 upms_dir="UPMS", prior_dir="PRIOR", normalization="minmax",
                 max_value=1.0):
        self.root = Path(root).expanduser().resolve()
        self.ms_channels = ms_channels
        self.normalization = normalization
        self.max_value = max_value
        folders = {"ms": ms_dir, "pan": pan_dir, "mul": upms_dir, "pri": prior_dir}
        indexed = {}
        for suffix, folder in folders.items():
            directory = self.root / folder
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing input directory: {directory}")
            files = {}
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in (".tif", ".tiff"):
                    continue
                ending = f"_{suffix}"
                if not path.stem.endswith(ending):
                    continue
                sample_id = path.stem[:-len(ending)]
                if not sample_id or sample_id in files:
                    raise ValueError(f"Empty or duplicate sample ID in {directory}: {sample_id}")
                files[sample_id] = path
            indexed[suffix] = files
        self.sample_ids = sorted(indexed["ms"])
        if not self.sample_ids:
            raise ValueError(f"No *_ms.tif / *_ms.tiff samples found in {self.root / ms_dir}")
        expected = set(self.sample_ids)
        for suffix, files in indexed.items():
            missing = sorted(expected - files.keys())
            extra = sorted(files.keys() - expected)
            if missing or extra:
                raise ValueError(f"Unpaired {suffix} files: missing={missing[:5]}, extra={extra[:5]}")
        self.samples = [tuple(indexed[key][sample_id] for key in folders)
                        for sample_id in self.sample_ids]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ms, pan, upms, prior = (
            load_image(path, self.normalization, self.max_value)
            for path in self.samples[index]
        )
        if pan.shape[0] != 1:
            raise ValueError(f"PAN must have one channel: {self.samples[index][1]}")
        for name, image in (("MS", ms), ("UPMS", upms), ("PRIOR", prior)):
            if image.shape[0] != self.ms_channels:
                raise ValueError(f"{name} has {image.shape[0]} channels; expected {self.ms_channels}.")
        if upms.shape[1:] != pan.shape[1:] or prior.shape != upms.shape:
            raise ValueError(f"PAN, UPMS and PRIOR must share spatial dimensions: {self.sample_ids[index]}")
        return ms, pan, upms, prior
