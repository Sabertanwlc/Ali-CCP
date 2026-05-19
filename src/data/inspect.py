from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.ali_ccp_format import (
    format_rate,
    iter_lines,
    parse_common_line,
    parse_skeleton_line,
    resolve_dataset_paths,
)


def inspect_skeleton(path: Path, max_lines: int) -> dict[str, object]:
    counters: Counter[str] = Counter()
    ad_feature_counts: list[int] = []
    field_counts: Counter[str] = Counter()
    first_error: str | None = None

    for line in iter_lines(path, max_lines=max_lines):
        counters["rows"] += 1
        try:
            record = parse_skeleton_line(line)
        except Exception as exc:
            counters["parse_errors"] += 1
            if first_error is None:
                first_error = str(exc)
            continue

        counters["click_1"] += record.click == 1
        counters["conversion_1"] += record.conversion == 1
        counters["ctcvr_1"] += record.ctcvr_label == 1
        counters["conversion_without_click"] += record.conversion == 1 and record.click == 0
        ad_feature_counts.append(len(record.ad_features))
        for feature in record.ad_features:
            field_counts[feature.field_id] += 1

    parsed_rows = counters["rows"] - counters["parse_errors"]
    return {
        "path": str(path),
        "rows_scanned": counters["rows"],
        "parsed_rows": parsed_rows,
        "parse_errors": counters["parse_errors"],
        "first_error": first_error,
        "click_1": counters["click_1"],
        "conversion_1": counters["conversion_1"],
        "ctcvr_1": counters["ctcvr_1"],
        "conversion_without_click": counters["conversion_without_click"],
        "click_rate": format_rate(counters["click_1"], parsed_rows),
        "conversion_rate": format_rate(counters["conversion_1"], parsed_rows),
        "ctcvr_rate": format_rate(counters["ctcvr_1"], parsed_rows),
        "avg_ad_features": sum(ad_feature_counts) / len(ad_feature_counts) if ad_feature_counts else 0.0,
        "top_ad_fields": field_counts.most_common(20),
    }


def inspect_common(path: Path, max_lines: int) -> dict[str, object]:
    counters: Counter[str] = Counter()
    common_feature_counts: list[int] = []
    field_counts: Counter[str] = Counter()
    first_error: str | None = None

    for line in iter_lines(path, max_lines=max_lines):
        counters["rows"] += 1
        try:
            record = parse_common_line(line)
        except Exception as exc:
            counters["parse_errors"] += 1
            if first_error is None:
                first_error = str(exc)
            continue

        common_feature_counts.append(len(record.common_features))
        for feature in record.common_features:
            field_counts[feature.field_id] += 1

    parsed_rows = counters["rows"] - counters["parse_errors"]
    return {
        "path": str(path),
        "rows_scanned": counters["rows"],
        "parsed_rows": parsed_rows,
        "parse_errors": counters["parse_errors"],
        "first_error": first_error,
        "avg_common_features": sum(common_feature_counts) / len(common_feature_counts)
        if common_feature_counts
        else 0.0,
        "top_common_fields": field_counts.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Ali-CCP raw files without loading them into memory.")
    parser.add_argument("--data-root", default="D:\\Ali-CCP")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-lines", type=int, default=100_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    skeleton_path, common_path = resolve_dataset_paths(data_root, args.split)
    report = {
        "split": args.split,
        "max_lines": args.max_lines,
        "skeleton": inspect_skeleton(skeleton_path, args.max_lines),
        "common": inspect_common(common_path, args.max_lines),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

