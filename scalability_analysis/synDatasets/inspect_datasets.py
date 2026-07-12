"""
inspect_datasets.py
--------------------
Inspects the binary synthetic datasets from:
  https://zenodo.org/records/8433873/files/synDatasets.tar.gz

Run this BEFORE any benchmarking to understand:
  - File formats (pickle, parquet, HDF5, etc.)
  - Column names and dtypes
  - Dataset sizes (rows, memory)
  - Ground truth / duplicate structure
  - Any nested or multi-file structure inside the archive

Usage:
  python inspect_datasets.py --root /path/to/synDatasets
  python inspect_datasets.py --root /path/to/synDatasets --verbose
  python inspect_datasets.py --tarball /path/to/synDatasets.tar.gz   # inspect without extracting first
"""

import os
import sys
import struct
import argparse
import traceback
from pathlib import Path


# ── Optional imports (graceful degradation) ───────────────────────────────────
try:
    import pickle
    HAS_PICKLE = True
except ImportError:
    HAS_PICKLE = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not found. Install with: pip install pandas")

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    import tarfile
    HAS_TAR = True
except ImportError:
    HAS_TAR = False


# ── Helpers ───────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 70

def section(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def subsection(title):
    print(f"\n  ── {title} ──")

def fmt_size(n_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"

def sniff_magic(filepath):
    """Detect file format from magic bytes."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(16)
    except Exception:
        return "unreadable"

    # Pickle: starts with 0x80 followed by protocol byte (0x02–0x05)
    if header[:2] in [b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05']:
        return "pickle"
    # Parquet: PAR1 magic
    if header[:4] == b'PAR1':
        return "parquet"
    # HDF5: \x89HDF
    if header[:4] == b'\x89HDF':
        return "hdf5"
    # Feather/Arrow IPC: ARROW1 or continuation marker
    if header[:6] == b'ARROW1':
        return "feather"
    if header[:4] == b'\xff\xff\xff\xff':
        return "arrow-ipc"
    # SQLite
    if header[:6] == b'SQLite':
        return "sqlite"
    # Gzip
    if header[:2] == b'\x1f\x8b':
        return "gzip"
    # CSV / plain text
    try:
        header.decode("utf-8")
        return "text/csv"
    except UnicodeDecodeError:
        pass
    return "unknown-binary"


def summarize_dataframe(df, label="", verbose=False):
    """Print a tidy summary of a pandas DataFrame."""
    print(f"\n    Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    mem = df.memory_usage(deep=True).sum()
    print(f"    Memory      : {fmt_size(mem)}")
    print(f"    Columns     : {list(df.columns)}")
    print(f"    Dtypes      :")
    for col, dtype in df.dtypes.items():
        null_pct = df[col].isna().mean() * 100
        print(f"      {col:<30} {str(dtype):<15}  nulls: {null_pct:.1f}%")
    if verbose:
        print(f"\n    First 3 rows:")
        print(df.head(3).to_string(index=True))


def inspect_pickle(filepath, verbose=False):
    if not HAS_PICKLE or not HAS_PANDAS:
        print("    [SKIP] pickle/pandas not available")
        return
    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        print(f"    Pickle type : {type(obj).__name__}")
        if HAS_PANDAS and isinstance(obj, pd.DataFrame):
            summarize_dataframe(obj, verbose=verbose)
        elif isinstance(obj, dict):
            print(f"    Dict keys   : {list(obj.keys())}")
            for k, v in obj.items():
                print(f"      '{k}' → {type(v).__name__}", end="")
                if HAS_PANDAS and isinstance(v, pd.DataFrame):
                    print(f" shape={v.shape}  cols={list(v.columns)}")
                elif isinstance(v, list):
                    print(f" len={len(v)}")
                else:
                    print()
        elif isinstance(obj, (list, tuple)):
            print(f"    List/tuple  : {len(obj)} items, first item type: {type(obj[0]).__name__}")
            if HAS_PANDAS and isinstance(obj[0], pd.DataFrame):
                for i, df in enumerate(obj[:3]):
                    print(f"    Item {i}:")
                    summarize_dataframe(df, verbose=verbose)
        else:
            print(f"    Value       : {repr(obj)[:200]}")
    except Exception as e:
        print(f"    [ERROR] Could not load pickle: {e}")


def inspect_parquet(filepath, verbose=False):
    if not HAS_PARQUET or not HAS_PANDAS:
        print("    [SKIP] pyarrow/pandas not available")
        return
    try:
        df = pd.read_parquet(filepath)
        summarize_dataframe(df, verbose=verbose)
    except Exception as e:
        print(f"    [ERROR] Could not read parquet: {e}")


def inspect_hdf5(filepath, verbose=False):
    if not HAS_HDF5:
        print("    [SKIP] h5py not available")
        return
    try:
        with h5py.File(filepath, "r") as f:
            print(f"    HDF5 keys   : {list(f.keys())}")
            for key in f.keys():
                item = f[key]
                print(f"      '{key}' → {type(item).__name__}  shape={getattr(item, 'shape', '?')}  dtype={getattr(item, 'dtype', '?')}")
                if HAS_PANDAS and hasattr(item, 'shape') and len(item.shape) == 2:
                    df = pd.DataFrame(item[()])
                    summarize_dataframe(df, label=key, verbose=verbose)
    except Exception as e:
        print(f"    [ERROR] Could not read HDF5: {e}")


def inspect_csv(filepath, verbose=False):
    if not HAS_PANDAS:
        print("    [SKIP] pandas not available")
        return
    try:
        df = pd.read_csv(filepath, nrows=5 if not verbose else None)
        if not verbose:
            # Re-read for shape
            df_full = pd.read_csv(filepath)
            summarize_dataframe(df_full, verbose=verbose)
        else:
            summarize_dataframe(df, verbose=verbose)
    except Exception as e:
        print(f"    [ERROR] Could not read CSV: {e}")


def inspect_file(filepath, verbose=False):
    """Dispatch to the right inspector based on magic bytes and extension."""
    path = Path(filepath)
    size = path.stat().st_size
    fmt = sniff_magic(filepath)
    ext = path.suffix.lower()

    print(f"\n  File   : {path.name}")
    print(f"  Size   : {fmt_size(size)}")
    print(f"  Magic  : {fmt}  |  Extension: {ext}")

    # Try extension override if magic is ambiguous
    if fmt == "unknown-binary" and ext in [".pkl", ".pickle"]:
        fmt = "pickle"
    if ext in [".parquet", ".pq"]:
        fmt = "parquet"
    if ext in [".h5", ".hdf5"]:
        fmt = "hdf5"
    if ext in [".csv", ".tsv"]:
        fmt = "text/csv"
    if ext in [".feather"]:
        fmt = "feather"

    if fmt == "pickle":
        inspect_pickle(filepath, verbose)
    elif fmt == "parquet":
        inspect_parquet(filepath, verbose)
    elif fmt == "hdf5":
        inspect_hdf5(filepath, verbose)
    elif fmt in ["text/csv"]:
        inspect_csv(filepath, verbose)
    elif fmt == "feather":
        if HAS_PANDAS:
            try:
                df = pd.read_feather(filepath)
                summarize_dataframe(df, verbose=verbose)
            except Exception as e:
                print(f"    [ERROR] Could not read feather: {e}")
        else:
            print("    [SKIP] pandas not available for feather")
    else:
        print(f"    [INFO] Unknown format — showing raw hex header:")
        with open(filepath, "rb") as f:
            raw = f.read(64)
        print(f"    Hex   : {raw.hex()}")
        print(f"    ASCII : {repr(raw)}")


def list_tar_contents(tarball_path):
    """List contents of the tar.gz without extracting."""
    section(f"TAR CONTENTS: {Path(tarball_path).name}")
    if not HAS_TAR:
        print("  [SKIP] tarfile not available")
        return
    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            members = tar.getmembers()
            print(f"  Total entries : {len(members)}")
            dirs = set()
            files = []
            for m in members:
                if m.isdir():
                    dirs.add(m.name)
                else:
                    files.append(m)
            print(f"  Directories   : {len(dirs)}")
            print(f"  Files         : {len(files)}")
            print()
            total_size = 0
            for m in sorted(files, key=lambda x: x.size, reverse=True):
                total_size += m.size
                print(f"    {fmt_size(m.size):>10}  {m.name}")
            print(f"\n  Total uncompressed size: {fmt_size(total_size)}")
    except Exception as e:
        print(f"  [ERROR] Could not open tarball: {e}")


def inspect_directory(root, verbose=False):
    """Walk the extracted directory and inspect every file."""
    root = Path(root)
    if not root.exists():
        print(f"[ERROR] Path does not exist: {root}")
        sys.exit(1)

    section(f"DIRECTORY SCAN: {root}")

    all_files = sorted(root.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    print(f"  Total files found: {len(files)}")

    # Group by extension
    from collections import defaultdict
    by_ext = defaultdict(list)
    for f in files:
        by_ext[f.suffix.lower()].append(f)

    subsection("Files by extension")
    for ext, flist in sorted(by_ext.items()):
        total = sum(f.stat().st_size for f in flist)
        print(f"    {ext or '(no ext)':>12}  ×{len(flist)}  total {fmt_size(total)}")

    subsection("Individual file inspection")
    for filepath in files:
        try:
            inspect_file(filepath, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] inspecting {filepath.name}: {e}")
            if verbose:
                traceback.print_exc()

    # ── Heuristic: look for ground truth ──────────────────────────────────────
    section("GROUND TRUTH DETECTION")
    gt_keywords = ["gt", "ground", "truth", "label", "match", "duplicate", "gold"]
    found_gt = []
    for f in files:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in gt_keywords):
            found_gt.append(f)
    if found_gt:
        print(f"  Likely ground truth files ({len(found_gt)}):")
        for f in found_gt:
            print(f"    {f}")
            try:
                inspect_file(f, verbose=True)
            except Exception as e:
                print(f"    [ERROR]: {e}")
    else:
        print("  No files with obvious ground-truth names found.")
        print("  Tip: ground truth may be embedded inside the main data files.")
        print("       Check for columns like: 'id', 'entity_id', 'cluster_id', 'label'")

    # ── Summary table ─────────────────────────────────────────────────────────
    section("SCALABILITY PROFILE (dataset sizes)")
    if HAS_PANDAS:
        rows = []
        for filepath in files:
            size = filepath.stat().st_size
            fmt = sniff_magic(filepath)
            rows.append({"file": filepath.name, "size_bytes": size,
                         "size_human": fmt_size(size), "format": fmt})
        df = pd.DataFrame(rows).sort_values("size_bytes")
        print(df[["file", "size_human", "format"]].to_string(index=False))
    else:
        for f in files:
            print(f"  {fmt_size(f.stat().st_size):>10}  {f.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect binary synthetic ER datasets (Zenodo 8433873)"
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Path to the extracted synDatasets directory"
    )
    parser.add_argument(
        "--tarball", type=str, default=None,
        help="Path to synDatasets.tar.gz (lists contents without extracting)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print first rows of each dataset"
    )
    args = parser.parse_args()

    if args.tarball:
        list_tar_contents(args.tarball)

    if args.root:
        inspect_directory(args.root, verbose=args.verbose)

    if not args.tarball and not args.root:
        # Try to auto-detect common paths
        candidates = [
            Path("synDatasets"),
            Path("./synDatasets"),
            Path(os.path.expanduser("~/synDatasets")),
            Path("/data/synDatasets"),
            Path("/scratch/synDatasets"),
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            print(f"[AUTO] Found dataset dir at: {found}")
            inspect_directory(found, verbose=args.verbose)
        else:
            print("Usage examples:")
            print("  python inspect_datasets.py --tarball synDatasets.tar.gz")
            print("  python inspect_datasets.py --root ./synDatasets")
            print("  python inspect_datasets.py --root ./synDatasets --verbose")
            sys.exit(0)


if __name__ == "__main__":
    main()