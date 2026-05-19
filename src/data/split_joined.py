from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.ali_ccp_format import iter_lines, parse_joined_line
from src.data.feature_stream import stable_hash


def split_joined(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input)
    train_path = Path(args.train_output)
    valid_path = Path(args.valid_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": 0,
        "parse_errors": 0,
        "train_rows": 0,
        "valid_rows": 0,
        "train_clicks": 0,
        "valid_clicks": 0,
        "train_conversions": 0,
        "valid_conversions": 0,
    }
    threshold = int(args.valid_ratio * args.hash_modulo)

    with train_path.open("w", encoding="utf-8", newline="") as train_file, valid_path.open(
        "w", encoding="utf-8", newline=""
    ) as valid_file:
        for line in iter_lines(input_path):
            stats["rows"] += 1
            try:
                record = parse_joined_line(line)
            except Exception:
                stats["parse_errors"] += 1
                continue

            bucket = stable_hash(record.sample_id, args.hash_modulo)
            if bucket < threshold:
                valid_file.write(line)
                stats["valid_rows"] += 1
                stats["valid_clicks"] += record.click
                stats["valid_conversions"] += record.conversion
            else:
                train_file.write(line)
                stats["train_rows"] += 1
                stats["train_clicks"] += record.click
                stats["train_conversions"] += record.conversion

    stats["valid_ratio"] = args.valid_ratio
    stats["train_click_rate"] = stats["train_clicks"] / stats["train_rows"] if stats["train_rows"] else 0.0
    stats["valid_click_rate"] = stats["valid_clicks"] / stats["valid_rows"] if stats["valid_rows"] else 0.0
    stats["train_conversion_rate"] = (
        stats["train_conversions"] / stats["train_rows"] if stats["train_rows"] else 0.0
    )
    stats["valid_conversion_rate"] = (
        stats["valid_conversions"] / stats["valid_rows"] if stats["valid_rows"] else 0.0
    )
    stats["input"] = str(input_path)
    stats["train_output"] = str(train_path)
    stats["valid_output"] = str(valid_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable train/valid split for joined Ali-CCP samples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--valid-output", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--hash-modulo", type=int, default=10000)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()

    report = split_joined(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report_output:
        output_path = Path(args.report_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

