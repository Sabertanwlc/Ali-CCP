from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from src.data.feature_stream import iter_hashed_joined_examples, iter_hashed_skeleton_examples


class AliCcpHashedDataset(IterableDataset):
    def __init__(
        self,
        skeleton_path: Path,
        hash_size: int,
        max_lines: int | None = None,
        value_clip: float | None = None,
        normalize_l2: bool = False,
        feature_source: str = "ad",
    ) -> None:
        self.skeleton_path = skeleton_path
        self.hash_size = hash_size
        self.max_lines = max_lines
        self.value_clip = value_clip
        self.normalize_l2 = normalize_l2
        self.feature_source = feature_source

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.feature_source == "joined":
            stream = iter_hashed_joined_examples(
                self.skeleton_path,
                hash_size=self.hash_size,
                max_lines=self.max_lines,
                value_clip=self.value_clip,
                normalize_l2=self.normalize_l2,
            )
        else:
            stream = iter_hashed_skeleton_examples(
                self.skeleton_path,
                hash_size=self.hash_size,
                max_lines=self.max_lines,
                value_clip=self.value_clip,
                normalize_l2=self.normalize_l2,
            )

        for example in stream:
            yield {
                "indices": torch.as_tensor(example.indices, dtype=torch.long),
                "values": torch.as_tensor(example.values, dtype=torch.float32),
                "click": torch.tensor(example.click, dtype=torch.float32),
                "conversion": torch.tensor(example.conversion, dtype=torch.float32),
                "ctcvr": torch.tensor(example.ctcvr, dtype=torch.float32),
            }


def collate_hashed_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_features = max(item["indices"].numel() for item in batch)
    batch_size = len(batch)
    indices = torch.zeros((batch_size, max_features), dtype=torch.long)
    values = torch.zeros((batch_size, max_features), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_features), dtype=torch.float32)

    for row, item in enumerate(batch):
        size = item["indices"].numel()
        indices[row, :size] = item["indices"]
        values[row, :size] = item["values"]
        mask[row, :size] = 1.0

    return {
        "indices": indices,
        "values": values,
        "mask": mask,
        "click": torch.stack([item["click"] for item in batch]),
        "conversion": torch.stack([item["conversion"] for item in batch]),
        "ctcvr": torch.stack([item["ctcvr"] for item in batch]),
    }
