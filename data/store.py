"""Trusted access to project Zarr stores.

All local project stores must carry a provenance record and a content hash.
This guard is deliberately strict so an untracked or quarantined store cannot
silently enter training, validation, or evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import xarray as xr


REQUIRED_PROVENANCE = {
    "source",
    "request_parameters",
    "download_date",
    "content_hash",
}


class UntrustedStoreError(RuntimeError):
    """Raised when a Zarr store has no complete trusted provenance record."""


def content_hash(path: str | Path) -> str:
    """Hash all Zarr payload/coordinate files, excluding mutable metadata."""
    path = Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        if item.name in {".zattrs", ".zmetadata"}:
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


def stamp_store(path: str | Path, *, source: str, request_parameters: dict,
                download_date: str, validity: str = "valid") -> str:
    """Write provenance attrs and return the content hash."""
    import zarr

    path = Path(path)
    digest = content_hash(path)
    attrs = {
        "source": source,
        "request_parameters": json.dumps(request_parameters, sort_keys=True),
        "download_date": download_date,
        "content_hash": digest,
        "validity": validity,
    }
    group = zarr.open_group(str(path), mode="a")
    group.attrs.update(attrs)
    try:
        zarr.consolidate_metadata(str(path))
    except AttributeError:  # pragma: no cover - older zarr compatibility
        pass
    return digest


def open_project_zarr(path: str | Path, **kwargs) -> xr.Dataset:
    """Open a local project store only after checking provenance metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr store does not exist: {path}")
    ds = xr.open_zarr(path, **kwargs)
    missing = sorted(REQUIRED_PROVENANCE - set(ds.attrs))
    if missing:
        ds.close()
        raise UntrustedStoreError(
            f"Refusing unprovenanced Zarr store {path}; missing attrs: {', '.join(missing)}"
        )
    if ds.attrs.get("validity") == "invalid_synthetic":
        ds.close()
        raise UntrustedStoreError(f"Refusing quarantined synthetic store: {path}")
    for key in REQUIRED_PROVENANCE:
        if not ds.attrs[key]:
            ds.close()
            raise UntrustedStoreError(f"Refusing store {path}: empty provenance attr {key}")
    expected_hash = content_hash(path)
    if ds.attrs["content_hash"] != expected_hash:
        recorded_hash = ds.attrs["content_hash"]
        ds.close()
        raise UntrustedStoreError(
            f"Refusing store {path}: content hash mismatch "
            f"(recorded {recorded_hash}, calculated {expected_hash})"
        )
    return ds
