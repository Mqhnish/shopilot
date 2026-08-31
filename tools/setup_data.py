"""Fetch and verify the frozen competition catalog.

``data/catalog.jsonl`` is 60 MB of the organizer's frozen data. It is not
committed -- it is not ours to redistribute, and a 60 MB blob does not belong in
a source repository -- so a fresh clone needs this one command before anything
can be scored.

Standard library only, like everything else here. Downloads the release asset,
verifies it against the organizer's published SHA256SUMS, decompresses it, and
checks the row count matches the documented 50,000.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ("https://github.com/TechJam2026/techjam-conversational-search"
           "/releases/download/participant-kit")
ARCHIVE = "catalog.jsonl.gz"
CHECKSUMS = "SHA256SUMS"
EXPECTED_ROWS = 50_000

# Published by the organizer alongside the release. Checked against the live
# SHA256SUMS as well; a mismatch between the two means the release moved and
# is worth stopping for rather than silently trusting either one.
PINNED_SHA256 = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def verify(catalog: Path, quiet: bool = False) -> bool:
    if not catalog.exists():
        if not quiet:
            print(f"missing: {catalog}")
        return False
    rows = count_rows(catalog)
    ok = rows == EXPECTED_ROWS
    if not quiet:
        print(f"{catalog}: {rows:,} rows "
              f"({'as expected' if ok else f'EXPECTED {EXPECTED_ROWS:,}'})")
    return ok


def fetch(url: str, destination: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    print(f"  -> {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def published_checksum() -> str:
    """The SHA256 the organizer publishes for the archive, right now."""
    with urllib.request.urlopen(f"{RELEASE}/{CHECKSUMS}", timeout=60) as response:
        text = response.read().decode("utf-8", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == ARCHIVE:
            return parts[0]
    raise RuntimeError(f"{ARCHIVE} not listed in {CHECKSUMS}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify what is already on disk and exit")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the catalog is already valid")
    args = parser.parse_args()

    data = ROOT / "data"
    catalog = data / "catalog.jsonl"

    if args.check:
        return 0 if verify(catalog) else 1
    if catalog.exists() and not args.force:
        if verify(catalog):
            print("catalog already present and valid; nothing to do")
            return 0
        print("catalog present but wrong size; re-downloading")

    data.mkdir(parents=True, exist_ok=True)
    archive = data / ARCHIVE
    try:
        fetch(f"{RELEASE}/{ARCHIVE}", archive)
        expected = published_checksum()
    except Exception as error:
        print(f"\ndownload failed: {error}\n\n"
              f"Fetch it manually instead:\n"
              f"  curl -L -o data/{ARCHIVE} {RELEASE}/{ARCHIVE}\n"
              f"  gunzip -c data/{ARCHIVE} > data/catalog.jsonl", file=sys.stderr)
        return 1

    if expected != PINNED_SHA256:
        print(f"\nthe published checksum has changed since this was written:\n"
              f"  published: {expected}\n  pinned:    {PINNED_SHA256}\n"
              f"The release may have been re-cut. Stopping rather than guessing.",
              file=sys.stderr)
        return 1

    actual = sha256_of(archive)
    if actual != expected:
        print(f"\nchecksum mismatch -- the download is corrupt:\n"
              f"  expected {expected}\n  got      {actual}", file=sys.stderr)
        archive.unlink(missing_ok=True)
        return 1
    print(f"sha256 verified: {actual}")

    print("decompressing…")
    with gzip.open(archive, "rb") as source, catalog.open("wb") as out:
        shutil.copyfileobj(source, out)
    archive.unlink(missing_ok=True)

    if not verify(catalog):
        return 1
    print("\nready. Run `make verify`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
