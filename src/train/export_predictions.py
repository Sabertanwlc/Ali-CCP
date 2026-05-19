from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.ali_ccp_format import iter_lines, parse_joined_line
from src.data.torch_dataset import AliCcpHashedDataset, collate_hashed_batch
from src.train.train_torch import build_model


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    model = build_model(checkpoint_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    return model


def iter_joined_metadata(path: Path, max_lines: int | None = None):
    for line in iter_lines(path, max_lines=max_lines):
        record = parse_joined_line(line)
        yield {
            "sample_id": record.sample_id,
            "click": record.click,
            "conversion": record.conversion,
            "ctcvr": record.ctcvr_label,
            "common_feature_id": record.common_feature_id,
            "feature_count": len(record.ad_features),
        }


def export_predictions(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_model(args)
    dataset = AliCcpHashedDataset(
        Path(args.valid_path),
        hash_size=args.hash_size,
        max_lines=args.max_lines,
        value_clip=args.value_clip,
        normalize_l2=args.normalize_l2,
        feature_source="joined",
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_hashed_batch)
    metadata_iter = iter(iter_joined_metadata(Path(args.valid_path), max_lines=args.max_lines))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "sample_id",
            "click",
            "conversion",
            "ctcvr",
            "common_feature_id",
            "feature_count",
            "pctr",
            "pcvr",
            "pctcvr",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for batch in dataloader:
                indices = batch["indices"].to(device)
                values = batch["values"].to(device)
                mask = batch["mask"].to(device)
                if args.model == "deepfm":
                    pctr = model(indices, values, mask).detach().cpu().tolist()
                    pcvr = ["" for _ in pctr]
                    pctcvr = ["" for _ in pctr]
                else:
                    outputs = model(indices, values, mask)
                    pctr = outputs["pctr"].detach().cpu().tolist()
                    pcvr = outputs["pcvr"].detach().cpu().tolist()
                    pctcvr = outputs["pctcvr"].detach().cpu().tolist()

                for pctr_value, pcvr_value, pctcvr_value in zip(pctr, pcvr, pctcvr, strict=True):
                    metadata = next(metadata_iter)
                    metadata["pctr"] = pctr_value
                    metadata["pcvr"] = pcvr_value
                    metadata["pctcvr"] = pctcvr_value
                    writer.writerow(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model predictions for joined Ali-CCP validation data.")
    parser.add_argument("--model", choices=["deepfm", "esmm_dcnv2"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hash-size", type=int, default=1_048_576)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--normalize-l2", action="store_true")
    parser.add_argument("--value-clip", type=float, default=10.0)
    parser.add_argument("--max-lines", type=int, default=None)
    args = parser.parse_args()
    export_predictions(args)


if __name__ == "__main__":
    main()

