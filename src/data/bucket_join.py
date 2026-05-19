from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TextIO

from src.data.ali_ccp_format import FEATURE_SEP, iter_lines, resolve_dataset_paths
from src.data.feature_stream import stable_hash


def open_bucket_writers(directory: Path, prefix: str, bucket_count: int) -> list[TextIO]:
    directory.mkdir(parents=True, exist_ok=True)
    return [
        (directory / f"{prefix}_{bucket_id:04d}.csv").open("w", encoding="utf-8", newline="")
        for bucket_id in range(bucket_count)
    ]


def close_writers(writers: list[TextIO]) -> None:
    for writer in writers:
        writer.close()


def split_skeleton(
    skeleton_path: Path,
    output_dir: Path,
    bucket_count: int,
    max_lines: int | None,
) -> dict[str, int]:
    writers = open_bucket_writers(output_dir, "skeleton", bucket_count)
    rows = 0
    errors = 0
    try:
        for line in iter_lines(skeleton_path, max_lines=max_lines):
            parts = line.rstrip("\n\r").split(",", 5)
            if len(parts) != 6:
                errors += 1
                continue
            bucket_id = stable_hash(parts[3], bucket_count)
            writers[bucket_id].write(line)
            rows += 1
    finally:
        close_writers(writers)
    return {"skeleton_rows": rows, "skeleton_split_errors": errors}


def split_common(
    common_path: Path,
    output_dir: Path,
    bucket_count: int,
    max_lines: int | None,
) -> dict[str, int]:
    writers = open_bucket_writers(output_dir, "common", bucket_count)
    rows = 0
    errors = 0
    try:
        for line in iter_lines(common_path, max_lines=max_lines):
            parts = line.rstrip("\n\r").split(",", 2)
            if len(parts) != 3:
                errors += 1
                continue
            bucket_id = stable_hash(parts[0], bucket_count)
            writers[bucket_id].write(line)
            rows += 1
    finally:
        close_writers(writers)
    return {"common_rows": rows, "common_split_errors": errors}


def join_bucket(skeleton_bucket: Path, common_bucket: Path, output_file: TextIO) -> dict[str, int]:
    common_map: dict[str, tuple[int, str]] = {}
    common_rows = 0
    for line in iter_lines(common_bucket):
        parts = line.rstrip("\n\r").split(",", 2)
        if len(parts) != 3:
            continue
        common_map[parts[0]] = (int(parts[1]), parts[2])
        common_rows += 1

    skeleton_rows = 0
    joined_rows = 0
    missing_common = 0
    for line in iter_lines(skeleton_bucket):
        parts = line.rstrip("\n\r").split(",", 5)
        if len(parts) != 6:
            continue
        skeleton_rows += 1
        common = common_map.get(parts[3])
        if common is None:
            missing_common += 1
            continue
        common_count, common_features = common
        ad_count = int(parts[4])
        if parts[5] and common_features:
            features = f"{parts[5]}{FEATURE_SEP}{common_features}"
        else:
            features = parts[5] or common_features
        output_file.write(
            f"{parts[0]},{parts[1]},{parts[2]},{parts[3]},{ad_count + common_count},{features}\n"
        )
        joined_rows += 1

    return {
        "bucket_common_rows": common_rows,
        "bucket_skeleton_rows": skeleton_rows,
        "bucket_joined_rows": joined_rows,
        "bucket_missing_common": missing_common,
    }


def bucket_join(args: argparse.Namespace) -> dict[str, object]:
    data_root = Path(args.data_root)
    skeleton_path, common_path = resolve_dataset_paths(data_root, args.split)
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "_tmp_buckets"
    skeleton_bucket_dir = temp_dir / "skeleton"
    common_bucket_dir = temp_dir / "common"
    joined_path = output_dir / f"joined_{args.split}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "split": args.split,
        "bucket_count": args.bucket_count,
        "max_skeleton_lines": args.max_skeleton_lines,
        "max_common_lines": args.max_common_lines,
        "joined_path": str(joined_path),
    }
    report.update(split_skeleton(skeleton_path, skeleton_bucket_dir, args.bucket_count, args.max_skeleton_lines))
    report.update(split_common(common_path, common_bucket_dir, args.bucket_count, args.max_common_lines))

    totals = {
        "common_rows_loaded": 0,
        "skeleton_rows_seen": 0,
        "joined_rows": 0,
        "missing_common": 0,
    }
    with joined_path.open("w", encoding="utf-8", newline="") as output_file:
        for bucket_id in range(args.bucket_count):
            bucket_report = join_bucket(
                skeleton_bucket_dir / f"skeleton_{bucket_id:04d}.csv",
                common_bucket_dir / f"common_{bucket_id:04d}.csv",
                output_file,
            )
            totals["common_rows_loaded"] += bucket_report["bucket_common_rows"]
            totals["skeleton_rows_seen"] += bucket_report["bucket_skeleton_rows"]
            totals["joined_rows"] += bucket_report["bucket_joined_rows"]
            totals["missing_common"] += bucket_report["bucket_missing_common"]

    skeleton_rows = totals["skeleton_rows_seen"]
    report.update(totals)
    report["join_coverage"] = totals["joined_rows"] / skeleton_rows if skeleton_rows else 0.0
    if not args.keep_temp:
        for path in temp_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(temp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        if temp_dir.exists():
            temp_dir.rmdir()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket join Ali-CCP skeleton and common feature files.")
    parser.add_argument("--data-root", default="D:\\Ali-CCP")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--bucket-count", type=int, default=128)
    parser.add_argument("--max-skeleton-lines", type=int, default=None)
    parser.add_argument("--max-common-lines", type=int, default=None)
    parser.add_argument("--output-dir", default="D:\\Ali-CCP\\processed\\joined")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()

    report = bucket_join(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_output:
        output_path = Path(args.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

