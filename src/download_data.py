"""
Download the real PDEBench 2D Darcy Flow (beta=1.0) dataset.

PDEBench ships ONE HDF5 file per beta value containing all 10,000 samples
(there is no separate "Test" file, unlike Burgers/Advection) - the official
list of dataset files/URLs is published in the PDEBench repo:
https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/pdebench_data_urls.csv

The previous version of this script pointed at a nonexistent HuggingFace
path and silently failed; the URL below is the real DaRUS (Uni Stuttgart)
download link taken directly from that CSV.
"""
import hashlib
import h5py
import requests
from tqdm import tqdm
from pathlib import Path

DATA_URL = "https://darus.uni-stuttgart.de/api/access/datafile/133219"
FILENAME = "2D_DarcyFlow_beta1.0_Train.hdf5"
# Official MD5 published in PDEBench's data manifest (pdebench_data_urls.csv) for
# 2D_DarcyFlow_beta1.0_Train.hdf5. Verifying against this guarantees we have the exact,
# unmodified benchmark file - not a truncated download or a look-alike.
EXPECTED_MD5 = "81694ed31306ff2e5f6b76349b0b4389"

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_pdebench"


def download_file(url: str, filepath: Path, chunk_size: int = 1 << 20):
    """Download file with progress bar."""
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=filepath.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def verify_md5(filepath: Path, expected: str = EXPECTED_MD5) -> bool:
    """Verify the file's MD5 matches PDEBench's published manifest hash (byte-for-byte)."""
    h = hashlib.md5()
    size = filepath.stat().st_size
    with open(filepath, "rb") as f, tqdm(
        total=size, unit="B", unit_scale=True, desc=f"md5 {filepath.name}"
    ) as pbar:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            pbar.update(len(chunk))
    digest = h.hexdigest()
    if digest != expected:
        print(f"  MD5 MISMATCH: got {digest}, expected {expected}")
        return False
    print(f"  MD5 OK ({digest}) - matches PDEBench manifest")
    return True


def verify_hdf5(filepath: Path) -> bool:
    """Verify HDF5 file can be opened and has expected PDEBench Darcy structure."""
    try:
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            print(f"Keys in {filepath.name}: {keys}")
            print(f"  attrs: {dict(f.attrs)}")
            if "nu" not in keys or "tensor" not in keys:
                print("  Missing expected 'nu'/'tensor' datasets")
                return False
            print(f"  nu (permeability) shape: {f['nu'].shape}, dtype: {f['nu'].dtype}")
            print(f"  tensor (pressure) shape: {f['tensor'].shape}, dtype: {f['tensor'].dtype}")
        return True
    except Exception as e:
        print(f"Verification failed for {filepath}: {e}")
        return False


def verify(filepath: Path) -> bool:
    """Full verification: byte-for-byte checksum + HDF5 structure."""
    return verify_md5(filepath) and verify_hdf5(filepath)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / FILENAME

    if filepath.exists():
        print(f"{filepath.name} already exists, verifying...")
        if verify(filepath):
            print("  Verified OK")
            return
        print("  Verification failed, re-downloading...")
        filepath.unlink()

    print(f"Downloading {FILENAME} (~1.25 GB) from {DATA_URL} ...")
    try:
        download_file(DATA_URL, filepath)
        if verify(filepath):
            print("  Downloaded and verified")
        else:
            raise RuntimeError("Downloaded file failed verification")
    except Exception as e:
        print(f"  Download failed: {e}")
        if filepath.exists():
            filepath.unlink()
        raise

    print(
        "\nDone. This single file contains all 10,000 PDEBench samples "
        "(no separate test file); src/preprocess.py performs the train/val/test split.\n"
        "Next step: run `python src/preprocess.py`."
    )


if __name__ == "__main__":
    main()
