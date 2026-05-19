from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FEATURE_SEP = "\x01"
FIELD_FEATURE_SEP = "\x02"
FEATURE_VALUE_SEP = "\x03"


@dataclass(frozen=True)
class SparseFeature:
    field_id: str
    feature_id: str
    value: float

    @property
    def token(self) -> str:
        return f"{self.field_id}:{self.feature_id}"


@dataclass(frozen=True)
class SkeletonRecord:
    sample_id: str
    click: int
    conversion: int
    common_feature_id: str
    ad_feature_count: int
    ad_features: tuple[SparseFeature, ...]

    @property
    def ctcvr_label(self) -> int:
        return int(self.click == 1 and self.conversion == 1)


@dataclass(frozen=True)
class CommonFeatureRecord:
    common_feature_id: str
    common_feature_count: int
    common_features: tuple[SparseFeature, ...]


def parse_sparse_features(raw_features: str) -> tuple[SparseFeature, ...]:
    if not raw_features:
        return ()

    features: list[SparseFeature] = []
    for part in raw_features.split(FEATURE_SEP):
        if not part:
            continue
        field_and_feature, sep, raw_value = part.partition(FEATURE_VALUE_SEP)
        if not sep:
            raise ValueError(f"missing feature-value separator in feature: {part!r}")
        field_id, sep, feature_id = field_and_feature.partition(FIELD_FEATURE_SEP)
        if not sep:
            raise ValueError(f"missing field-feature separator in feature: {part!r}")
        if not field_id or not feature_id:
            raise ValueError(f"empty field_id or feature_id in feature: {part!r}")
        features.append(SparseFeature(field_id=field_id, feature_id=feature_id, value=float(raw_value)))
    return tuple(features)


def parse_skeleton_line(line: str) -> SkeletonRecord:
    parts = line.rstrip("\n\r").split(",", 5)
    if len(parts) != 6:
        raise ValueError(f"skeleton line should have 6 columns, got {len(parts)}")

    sample_id, raw_click, raw_conversion, common_feature_id, raw_count, raw_features = parts
    return SkeletonRecord(
        sample_id=sample_id,
        click=int(raw_click),
        conversion=int(raw_conversion),
        common_feature_id=common_feature_id,
        ad_feature_count=int(raw_count),
        ad_features=parse_sparse_features(raw_features),
    )


def parse_joined_line(line: str) -> SkeletonRecord:
    parts = line.rstrip("\n\r").split(",", 5)
    if len(parts) != 6:
        raise ValueError(f"joined line should have 6 columns, got {len(parts)}")

    sample_id, raw_click, raw_conversion, common_feature_id, raw_count, raw_features = parts
    return SkeletonRecord(
        sample_id=sample_id,
        click=int(raw_click),
        conversion=int(raw_conversion),
        common_feature_id=common_feature_id,
        ad_feature_count=int(raw_count),
        ad_features=parse_sparse_features(raw_features),
    )


def parse_common_line(line: str) -> CommonFeatureRecord:
    parts = line.rstrip("\n\r").split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"common line should have 3 columns, got {len(parts)}")

    common_feature_id, raw_count, raw_features = parts
    return CommonFeatureRecord(
        common_feature_id=common_feature_id,
        common_feature_count=int(raw_count),
        common_features=parse_sparse_features(raw_features),
    )


def iter_lines(path: Path, max_lines: int | None = None) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        for index, line in enumerate(file, start=1):
            if max_lines is not None and index > max_lines:
                break
            yield line


def resolve_dataset_paths(data_root: Path, split: str) -> tuple[Path, Path]:
    if split == "train":
        skeleton = data_root / "sample_train" / "sample_skeleton_train.csv"
        common = data_root / "sample_train" / "common_features_train.csv"
    elif split == "test":
        skeleton = data_root / "sample_test" / "sample_skeleton_test.csv"
        common = data_root / "sample_test" / "common_features_test.csv"
    else:
        raise ValueError("split must be 'train' or 'test'")
    return skeleton, common


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.000000"
    return f"{numerator / denominator:.6f}"
