from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.data.ali_ccp_format import SkeletonRecord, iter_lines, parse_joined_line, parse_skeleton_line


@dataclass(frozen=True)
class HashedExample:
    sample_id: str
    click: int
    conversion: int
    ctcvr: int
    indices: np.ndarray
    values: np.ndarray


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % modulo


def skeleton_to_hashed_example(
    record: SkeletonRecord,
    hash_size: int,
    value_clip: float | None = None,
    normalize_l2: bool = False,
) -> HashedExample:
    feature_values: dict[int, float] = {}
    for feature in record.ad_features:
        index = stable_hash(feature.token, hash_size)
        value = feature.value
        if value_clip is not None:
            value = max(min(value, value_clip), -value_clip)
        feature_values[index] = feature_values.get(index, 0.0) + value

    indices = np.fromiter(feature_values.keys(), dtype=np.int64)
    values = np.fromiter(feature_values.values(), dtype=np.float32)
    if normalize_l2 and values.size:
        norm = float(np.linalg.norm(values))
        if norm > 0.0:
            values = values / norm
    return HashedExample(
        sample_id=record.sample_id,
        click=record.click,
        conversion=record.conversion,
        ctcvr=record.ctcvr_label,
        indices=indices,
        values=values,
    )


def iter_hashed_skeleton_examples(
    skeleton_path: Path,
    hash_size: int,
    max_lines: int | None = None,
    value_clip: float | None = None,
    normalize_l2: bool = False,
) -> Iterable[HashedExample]:
    for line in iter_lines(skeleton_path, max_lines=max_lines):
        record = parse_skeleton_line(line)
        yield skeleton_to_hashed_example(record, hash_size, value_clip=value_clip, normalize_l2=normalize_l2)


def iter_hashed_joined_examples(
    joined_path: Path,
    hash_size: int,
    max_lines: int | None = None,
    value_clip: float | None = None,
    normalize_l2: bool = False,
) -> Iterable[HashedExample]:
    for line in iter_lines(joined_path, max_lines=max_lines):
        record = parse_joined_line(line)
        yield skeleton_to_hashed_example(record, hash_size, value_clip=value_clip, normalize_l2=normalize_l2)
