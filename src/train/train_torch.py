from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ali_ccp_format import resolve_dataset_paths
from src.data.torch_dataset import AliCcpHashedDataset, collate_hashed_batch
from src.models.deepfm import DeepFM
from src.models.esmm_dcnv2 import EsmmDcnV2
from src.train.metrics import binary_classification_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_binary_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float = 1.0,
) -> torch.Tensor:
    eps = 1e-7
    predictions = torch.clamp(predictions, eps, 1.0 - eps)
    loss = -(targets * torch.log(predictions) * positive_weight + (1.0 - targets) * torch.log(1.0 - predictions))
    return loss.mean()


def build_model(args: argparse.Namespace) -> nn.Module:
    hidden_units = tuple(int(item) for item in args.hidden_units.split(",") if item)
    if args.model == "deepfm":
        return DeepFM(
            num_embeddings=args.hash_size,
            embedding_dim=args.embedding_dim,
            hidden_units=hidden_units,
            dropout=args.dropout,
        )
    if args.model == "esmm_dcnv2":
        return EsmmDcnV2(
            num_embeddings=args.hash_size,
            embedding_dim=args.embedding_dim,
            cross_layers=args.cross_layers,
            hidden_units=hidden_units,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model: {args.model}")


def make_dataloader(path: Path, args: argparse.Namespace) -> DataLoader:
    dataset = AliCcpHashedDataset(
        path,
        hash_size=args.hash_size,
        max_lines=args.max_lines,
        value_clip=args.value_clip,
        normalize_l2=args.normalize_l2,
        feature_source=args.feature_source,
    )
    return DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_hashed_batch)


def predict_for_tasks(
    model: nn.Module,
    args: argparse.Namespace,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, list[float] | list[int]]:
    model.eval()
    click_labels: list[int] = []
    conversion_labels: list[int] = []
    ctcvr_labels: list[int] = []
    pctr_predictions: list[float] = []
    pcvr_predictions: list[float] = []
    pctcvr_predictions: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            indices = batch["indices"].to(device)
            values = batch["values"].to(device)
            mask = batch["mask"].to(device)
            click = batch["click"]
            conversion = batch["conversion"]
            ctcvr = batch["ctcvr"]
            if args.model == "deepfm":
                output = model(indices, values, mask)
                pctr_predictions.extend(float(item) for item in output.detach().cpu().tolist())
            else:
                output = model(indices, values, mask)
                pctr_predictions.extend(float(item) for item in output["pctr"].detach().cpu().tolist())
                pcvr_predictions.extend(float(item) for item in output["pcvr"].detach().cpu().tolist())
                pctcvr_predictions.extend(float(item) for item in output["pctcvr"].detach().cpu().tolist())
            click_labels.extend(int(item) for item in click.cpu().tolist())
            conversion_labels.extend(int(item) for item in conversion.cpu().tolist())
            ctcvr_labels.extend(int(item) for item in ctcvr.cpu().tolist())
    model.train()
    return {
        "click_labels": click_labels,
        "conversion_labels": conversion_labels,
        "ctcvr_labels": ctcvr_labels,
        "pctr": pctr_predictions,
        "pcvr": pcvr_predictions,
        "pctcvr": pctcvr_predictions,
    }


def add_validation_metrics(report: dict[str, object], predictions: dict[str, list[float] | list[int]], model_name: str) -> None:
    click_labels = [int(item) for item in predictions["click_labels"]]
    conversion_labels = [int(item) for item in predictions["conversion_labels"]]
    ctcvr_labels = [int(item) for item in predictions["ctcvr_labels"]]
    pctr = [float(item) for item in predictions["pctr"]]

    report["valid_rows"] = len(click_labels)
    report["valid_clicks"] = sum(click_labels)
    report["valid_conversions"] = sum(conversion_labels)
    report["valid_ctcvr_positives"] = sum(ctcvr_labels)
    report["valid_click_rate"] = sum(click_labels) / len(click_labels) if click_labels else 0.0
    report["valid_conversion_rate"] = (
        sum(conversion_labels) / len(conversion_labels) if conversion_labels else 0.0
    )
    report.update(binary_classification_report(click_labels, pctr, prefix="valid_ctr"))

    if model_name != "deepfm":
        pcvr = [float(item) for item in predictions["pcvr"]]
        pctcvr = [float(item) for item in predictions["pctcvr"]]
        report.update(binary_classification_report(ctcvr_labels, pctcvr, prefix="valid_ctcvr"))
        clicked_labels: list[int] = []
        clicked_pcvr: list[float] = []
        for click, conversion, pcvr_prediction in zip(click_labels, conversion_labels, pcvr, strict=True):
            if click == 1:
                clicked_labels.append(conversion)
                clicked_pcvr.append(pcvr_prediction)
        report["valid_clicked_rows"] = len(clicked_labels)
        report["valid_clicked_conversions"] = sum(clicked_labels)
        report.update(binary_classification_report(clicked_labels, clicked_pcvr, prefix="valid_cvr_on_clicked"))


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    if args.train_path:
        train_path = Path(args.train_path)
    else:
        skeleton_path, _ = resolve_dataset_paths(Path(args.data_root), args.split)
        train_path = Path(args.joined_path) if args.feature_source == "joined" else skeleton_path
    if args.feature_source == "joined" and not (args.joined_path or args.train_path):
        raise ValueError("--joined-path or --train-path is required when --feature-source joined")
    dataloader = make_dataloader(train_path, args)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    bce = nn.BCELoss()

    history: list[dict[str, object]] = []
    best_score: float | None = None
    best_epoch: int | None = None
    best_report: dict[str, object] | None = None
    patience_used = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_rows = 0
        for batch in dataloader:
            indices = batch["indices"].to(device)
            values = batch["values"].to(device)
            mask = batch["mask"].to(device)
            click = batch["click"].to(device)
            ctcvr = batch["ctcvr"].to(device)

            optimizer.zero_grad()
            if args.model == "deepfm":
                prediction = model(indices, values, mask)
                loss = bce(prediction, click)
            else:
                outputs = model(indices, values, mask)
                loss = args.ctr_loss_weight * bce(outputs["pctr"], click) + args.ctcvr_loss_weight * weighted_binary_loss(
                    outputs["pctcvr"],
                    ctcvr,
                    positive_weight=args.ctcvr_positive_weight,
                )
            loss.backward()
            optimizer.step()

            batch_rows = indices.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_rows
            total_rows += batch_rows

        epoch_report: dict[str, object] = {
            "epoch": epoch + 1,
            "rows": total_rows,
            "avg_loss": total_loss / total_rows if total_rows else 0.0,
        }
        if args.valid_path:
            valid_dataloader = make_dataloader(Path(args.valid_path), args)
            predictions = predict_for_tasks(model, args, valid_dataloader, device)
            add_validation_metrics(epoch_report, predictions, args.model)

        score = metric_value(epoch_report, args.early_stop_metric)
        if score is not None:
            improved = best_score is None or score_better(score, best_score, args.early_stop_mode, args.min_delta)
            if improved:
                best_score = score
                best_epoch = epoch + 1
                best_report = dict(epoch_report)
                patience_used = 0
                if args.checkpoint_output:
                    save_checkpoint(model, args, Path(args.checkpoint_output), epoch + 1, epoch_report)
            else:
                patience_used += 1
        history.append(epoch_report)

        if args.early_stop_patience is not None and best_score is not None and patience_used >= args.early_stop_patience:
            break

    final_epoch_report = history[-1] if history else {}
    report: dict[str, object] = {
        "model": args.model,
        "feature_source": args.feature_source,
        "split": args.split,
        "seed": args.seed,
        "rows": final_epoch_report.get("rows", 0),
        "epochs": args.epochs,
        "epochs_completed": len(history),
        "hash_size": args.hash_size,
        "embedding_dim": args.embedding_dim,
        "avg_loss": final_epoch_report.get("avg_loss", 0.0),
        "device": str(device),
        "ctr_loss_weight": args.ctr_loss_weight,
        "ctcvr_loss_weight": args.ctcvr_loss_weight,
        "ctcvr_positive_weight": args.ctcvr_positive_weight,
        "early_stop_metric": args.early_stop_metric,
        "early_stop_mode": args.early_stop_mode,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
    }
    if best_report:
        for key, value in best_report.items():
            if key != "epoch":
                report[f"best_{key}"] = value
    elif final_epoch_report:
        for key, value in final_epoch_report.items():
            if key != "epoch":
                report[key] = value
    if args.checkpoint_output:
        report["checkpoint_output"] = args.checkpoint_output
    return report


def metric_value(report: dict[str, object], metric_name: str) -> float | None:
    value = report.get(metric_name)
    if isinstance(value, int | float):
        return float(value)
    return None


def score_better(score: float, best_score: float, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return score < best_score - min_delta
    return score > best_score + min_delta


def save_checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    checkpoint_path: Path,
    epoch: int,
    report: dict[str, object],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "report": report,
        },
        checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeepFM or ESMM+DCN-V2 on hashed Ali-CCP features.")
    parser.add_argument("--data-root", default="D:\\Ali-CCP")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model", choices=["deepfm", "esmm_dcnv2"], required=True)
    parser.add_argument("--feature-source", choices=["ad", "joined"], default="ad")
    parser.add_argument("--joined-path", default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--hash-size", type=int, default=262_144)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-units", default="256,128,64")
    parser.add_argument("--cross-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ctr-loss-weight", type=float, default=1.0)
    parser.add_argument("--ctcvr-loss-weight", type=float, default=1.0)
    parser.add_argument("--ctcvr-positive-weight", type=float, default=1.0)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument("--normalize-l2", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint-output", default=None)
    parser.add_argument("--early-stop-metric", default="valid_ctr_auc")
    parser.add_argument("--early-stop-mode", choices=["max", "min"], default="max")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
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
