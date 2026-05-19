from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines.lr_hashing import OnlineLogisticRegression
from src.data.ali_ccp_format import resolve_dataset_paths
from src.data.feature_stream import iter_hashed_joined_examples, iter_hashed_skeleton_examples
from src.train.metrics import binary_auc, binary_log_loss


def train(args: argparse.Namespace) -> dict[str, object]:
    skeleton_path, _ = resolve_dataset_paths(Path(args.data_root), args.split)
    model = OnlineLogisticRegression(
        hash_size=args.hash_size,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )

    labels: list[int] = []
    predictions: list[float] = []
    rows = 0
    parse_errors = 0

    if args.feature_source == "joined":
        if not args.joined_path:
            raise ValueError("--joined-path is required when --feature-source joined")
        stream = iter_hashed_joined_examples(
            Path(args.joined_path),
            hash_size=args.hash_size,
            max_lines=args.max_lines,
            value_clip=args.value_clip,
            normalize_l2=args.normalize_l2,
        )
    else:
        stream = iter_hashed_skeleton_examples(
            skeleton_path,
            hash_size=args.hash_size,
            max_lines=args.max_lines,
            value_clip=args.value_clip,
            normalize_l2=args.normalize_l2,
        )
    for example in stream:
        rows += 1
        prediction = model.update_one(example.indices, example.values, example.click)
        labels.append(example.click)
        predictions.append(prediction)

    return {
        "model": "lr_hashing",
        "task": "ctr",
        "feature_source": args.feature_source,
        "split": args.split,
        "rows": rows,
        "parse_errors": parse_errors,
        "hash_size": args.hash_size,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "value_clip": args.value_clip,
        "normalize_l2": args.normalize_l2,
        "click_rate": sum(labels) / len(labels) if labels else 0.0,
        "auc": binary_auc(labels, predictions),
        "log_loss": binary_log_loss(labels, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a streaming LR Hashing CTR baseline.")
    parser.add_argument("--data-root", default="D:\\Ali-CCP")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--max-lines", type=int, default=100_000)
    parser.add_argument("--feature-source", choices=["ad", "joined"], default="ad")
    parser.add_argument("--joined-path", default=None)
    parser.add_argument("--hash-size", type=int, default=262_144)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument("--normalize-l2", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = train(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
