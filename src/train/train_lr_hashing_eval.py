from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines.lr_hashing import OnlineLogisticRegression
from src.data.feature_stream import iter_hashed_joined_examples
from src.train.metrics import binary_classification_report


def train_and_eval(args: argparse.Namespace) -> dict[str, object]:
    model = OnlineLogisticRegression(
        hash_size=args.hash_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )

    train_rows = 0
    train_clicks = 0
    for example in iter_hashed_joined_examples(
        Path(args.train_path),
        hash_size=args.hash_size,
        max_lines=args.max_train_lines,
        value_clip=args.value_clip,
        normalize_l2=args.normalize_l2,
    ):
        model.update_one(example.indices, example.values, example.click)
        train_rows += 1
        train_clicks += example.click

    valid_labels: list[int] = []
    valid_predictions: list[float] = []
    valid_rows = 0
    valid_clicks = 0
    for example in iter_hashed_joined_examples(
        Path(args.valid_path),
        hash_size=args.hash_size,
        max_lines=args.max_valid_lines,
        value_clip=args.value_clip,
        normalize_l2=args.normalize_l2,
    ):
        valid_predictions.append(model.predict_one(example.indices, example.values))
        valid_labels.append(example.click)
        valid_rows += 1
        valid_clicks += example.click

    report = {
        "model": "lr_hashing",
        "task": "ctr",
        "feature_source": "joined",
        "train_path": args.train_path,
        "valid_path": args.valid_path,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "train_click_rate": train_clicks / train_rows if train_rows else 0.0,
        "valid_click_rate": valid_clicks / valid_rows if valid_rows else 0.0,
        "hash_size": args.hash_size,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "value_clip": args.value_clip,
        "normalize_l2": args.normalize_l2,
    }
    report.update(binary_classification_report(valid_labels, valid_predictions, prefix="valid_ctr"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LR Hashing on joined train and evaluate on joined valid.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--max-valid-lines", type=int, default=None)
    parser.add_argument("--hash-size", type=int, default=262_144)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument("--normalize-l2", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = train_and_eval(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
