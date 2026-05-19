from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.ali_ccp_format import FEATURE_SEP, iter_lines, resolve_dataset_paths


def collect_skeleton_rows(skeleton_path: Path, max_skeleton_lines: int) -> tuple[list[list[str]], set[str]]:
    rows: list[list[str]] = []
    common_ids: set[str] = set()
    for line in iter_lines(skeleton_path, max_lines=max_skeleton_lines):
        parts = line.rstrip("\n\r").split(",", 5)
        if len(parts) != 6:
            continue
        rows.append(parts)
        common_ids.add(parts[3])
    return rows, common_ids


def collect_required_common(common_path: Path, required_ids: set[str]) -> tuple[dict[str, tuple[int, str]], int]:
    common_map: dict[str, tuple[int, str]] = {}
    scanned_rows = 0
    for line in iter_lines(common_path):
        scanned_rows += 1
        parts = line.rstrip("\n\r").split(",", 2)
        if len(parts) != 3:
            continue
        if parts[0] in required_ids:
            common_map[parts[0]] = (int(parts[1]), parts[2])
            if len(common_map) == len(required_ids):
                break
    return common_map, scanned_rows


def selective_join(args: argparse.Namespace) -> dict[str, object]:
    skeleton_path, common_path = resolve_dataset_paths(Path(args.data_root), args.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skeleton_rows, required_ids = collect_skeleton_rows(skeleton_path, args.max_skeleton_lines)
    common_map, scanned_common_rows = collect_required_common(common_path, required_ids)

    joined_rows = 0
    missing_common = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        for parts in skeleton_rows:
            common = common_map.get(parts[3])
            if common is None:
                missing_common += 1
                continue
            common_count, common_features = common
            ad_count = int(parts[4])
            features = f"{parts[5]}{FEATURE_SEP}{common_features}" if parts[5] and common_features else parts[5] or common_features
            output_file.write(
                f"{parts[0]},{parts[1]},{parts[2]},{parts[3]},{ad_count + common_count},{features}\n"
            )
            joined_rows += 1

    return {
        "split": args.split,
        "skeleton_rows": len(skeleton_rows),
        "required_common_ids": len(required_ids),
        "matched_common_ids": len(common_map),
        "scanned_common_rows": scanned_common_rows,
        "joined_rows": joined_rows,
        "missing_common": missing_common,
        "join_coverage": joined_rows / len(skeleton_rows) if skeleton_rows else 0.0,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective join for small Ali-CCP validation samples.")
    parser.add_argument("--data-root", default="D:\\Ali-CCP")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-skeleton-lines", type=int, default=1000)
    parser.add_argument("--output", default="D:\\Ali-CCP\\processed\\joined_stage02_selective\\joined_train.csv")
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()

    report = selective_join(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_output:
        output_path = Path(args.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

