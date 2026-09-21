"""Read live POLAR status with short-lived, delete-sharing Windows handles.

This is a read-only observer, outside the hash-locked training/evaluation code.
It neither opens test predictions nor launches, retries, or alters any fit.
Use with the recorded status-I/O retry adapter: delete sharing alone does not
make MoveFileEx replacement of an open destination reliable on this machine.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path


def open_shared_binary(path: Path):
    """Open read-only; permit other handles to read, write, rename, or delete."""
    if os.name != "nt":
        return Path(path).open("rb")

    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ | WRITE | DELETE; OPEN_EXISTING.
    # No write/delete permission is requested for this observer itself.
    handle = create_file(str(Path(path).resolve()), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def read_shared_json(path: Path, *, optional=False):
    try:
        with open_shared_binary(path) as stream:
            content = stream.read()
    except FileNotFoundError:
        if optional:
            return None
        raise
    # Close the handle before parsing/rendering/printing the snapshot.
    return json.loads(content.decode("utf-8-sig"))


def contained(run: Path, value: str) -> Path:
    path = (run / value).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError("Status artifact escapes the selected run directory")
    return path


def snapshot(run: Path) -> dict:
    run = run.resolve(strict=True)
    queue = read_shared_json(run / "queue_status.json")
    receipts = read_shared_json(run / "queue_receipts.json", optional=True) or {"jobs": []}
    plan = read_shared_json(run / "queue_plan.json")
    counts = {"startup_checks": 0, "frozen_heads": 0, "polar4_neural": 0, "polar9_neural": 0}
    for receipt in receipts["jobs"]:
        if receipt["status"] != "COMPLETE":
            continue
        name = receipt["job"]
        if name.startswith("smoke_"):
            counts["startup_checks"] += 1
        elif "_head_" in name:
            counts["frozen_heads"] += 1
        elif name.startswith("polar4_neural_"):
            counts["polar4_neural"] += 1
        elif name.startswith("polar9_neural_"):
            counts["polar9_neural"] += 1
    active = next((job for job in plan["jobs"] if job["id"] == queue.get("job")), None)
    progress = progress_path = None
    if active:
        progress_path = contained(run, active["marker"]).parent / "progress.json"
        progress = read_shared_json(progress_path, optional=True)
    return {
        "observed_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run),
        "queue": queue,
        "completed": counts,
        "active_progress_path": str(progress_path) if progress_path else None,
        "active_progress": progress,
        "test_gate_open": (run / "test_access_gate.json").exists(),
        "completion": read_shared_json(run / "completion.json", optional=True),
        "reader_policy": "read_only_FILE_SHARE_READ_WRITE_DELETE_close_before_parse",
        "snapshot_atomic_across_multiple_files": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
