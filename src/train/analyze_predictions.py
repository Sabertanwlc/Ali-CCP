from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.train.metrics import binary_classification_report


def read_predictions(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            parsed: dict[str, object] = {
                "sample_id": row["sample_id"],
                "click": int(row["click"]),
                "conversion": int(row["conversion"]),
                "ctcvr": int(row["ctcvr"]),
                "common_feature_id": row["common_feature_id"],
                "feature_count": int(row["feature_count"]),
                "pctr": float(row["pctr"]),
            }
            parsed["pcvr"] = float(row["pcvr"]) if row.get("pcvr") else None
            parsed["pctcvr"] = float(row["pctcvr"]) if row.get("pctcvr") else None
            rows.append(parsed)
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def top_rows(rows: list[dict[str, object]], key: str, reverse: bool, limit: int) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:limit]


def analyze_predictions(args: argparse.Namespace) -> dict[str, object]:
    rows = read_predictions(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click_labels = [int(row["click"]) for row in rows]
    pctr = [float(row["pctr"]) for row in rows]
    report: dict[str, object] = {
        "input": args.input,
        "rows": len(rows),
        "clicks": sum(click_labels),
        "conversions": sum(int(row["conversion"]) for row in rows),
        "ctcvr_positives": sum(int(row["ctcvr"]) for row in rows),
        "avg_feature_count": sum(int(row["feature_count"]) for row in rows) / len(rows) if rows else 0.0,
    }
    report.update(binary_classification_report(click_labels, pctr, prefix="ctr"))

    write_rows(
        output_dir / "high_pctr_not_clicked.csv",
        top_rows([row for row in rows if int(row["click"]) == 0], "pctr", True, args.top_n),
    )
    write_rows(
        output_dir / "low_pctr_clicked.csv",
        top_rows([row for row in rows if int(row["click"]) == 1], "pctr", False, args.top_n),
    )

    if rows and rows[0].get("pctcvr") is not None:
        ctcvr_labels = [int(row["ctcvr"]) for row in rows]
        pctcvr = [float(row["pctcvr"]) for row in rows]
        report.update(binary_classification_report(ctcvr_labels, pctcvr, prefix="ctcvr"))

        clicked_rows = [row for row in rows if int(row["click"]) == 1]
        clicked_conversion = [int(row["conversion"]) for row in clicked_rows]
        clicked_pcvr = [float(row["pcvr"]) for row in clicked_rows]
        report["clicked_rows"] = len(clicked_rows)
        report.update(binary_classification_report(clicked_conversion, clicked_pcvr, prefix="cvr_on_clicked"))

        write_rows(
            output_dir / "high_pctcvr_not_converted.csv",
            top_rows([row for row in rows if int(row["ctcvr"]) == 0], "pctcvr", True, args.top_n),
        )
        write_rows(
            output_dir / "low_pctcvr_converted.csv",
            top_rows([row for row in rows if int(row["ctcvr"]) == 1], "pctcvr", False, args.top_n),
        )
        write_rows(
            output_dir / "low_pcvr_clicked_converted.csv",
            top_rows([row for row in clicked_rows if int(row["conversion"]) == 1], "pcvr", False, args.top_n),
        )

    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exported Ali-CCP validation predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    report = analyze_predictions(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

