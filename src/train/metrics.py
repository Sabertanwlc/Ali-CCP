from __future__ import annotations

import math


def binary_log_loss(labels: list[int], predictions: list[float]) -> float:
    if not labels:
        return 0.0
    eps = 1e-15
    total = 0.0
    for label, prediction in zip(labels, predictions, strict=True):
        probability = min(max(prediction, eps), 1.0 - eps)
        total += label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
    return -total / len(labels)


def binary_auc(labels: list[int], predictions: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    ranked = sorted(zip(predictions, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_count = sum(label for _, label in ranked[index:end])
        rank_sum += positive_count * average_rank
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_pr_auc(labels: list[int], predictions: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0

    ranked = sorted(zip(predictions, labels, strict=True), key=lambda item: item[0], reverse=True)
    true_positives = 0
    previous_recall = 0.0
    area = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positives += 1
        recall = true_positives / positives
        precision = true_positives / index
        area += precision * (recall - previous_recall)
        previous_recall = recall
    return area


def lift_at_k(labels: list[int], predictions: list[float], fractions: tuple[float, ...] = (0.01, 0.05, 0.10)) -> dict[str, float]:
    if not labels:
        return {f"lift_at_{fraction:g}": 0.0 for fraction in fractions}
    base_rate = sum(labels) / len(labels)
    ranked = sorted(zip(predictions, labels, strict=True), key=lambda item: item[0], reverse=True)
    result: dict[str, float] = {}
    for fraction in fractions:
        top_n = max(1, int(len(ranked) * fraction))
        top_rate = sum(label for _, label in ranked[:top_n]) / top_n
        result[f"rate_at_{fraction:g}"] = top_rate
        result[f"lift_at_{fraction:g}"] = top_rate / base_rate if base_rate > 0.0 else 0.0
    return result


def calibration_bins(labels: list[int], predictions: list[float], num_bins: int = 10) -> list[dict[str, float | int]]:
    if not labels:
        return []
    bins: list[dict[str, float | int]] = []
    for bin_index in range(num_bins):
        lower = bin_index / num_bins
        upper = (bin_index + 1) / num_bins
        rows = [
            (label, prediction)
            for label, prediction in zip(labels, predictions, strict=True)
            if lower <= prediction < upper or (bin_index == num_bins - 1 and prediction == 1.0)
        ]
        if not rows:
            bins.append(
                {
                    "bin": bin_index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "avg_prediction": 0.0,
                    "positive_rate": 0.0,
                }
            )
            continue
        positives = sum(label for label, _ in rows)
        bins.append(
            {
                "bin": bin_index,
                "lower": lower,
                "upper": upper,
                "count": len(rows),
                "avg_prediction": sum(prediction for _, prediction in rows) / len(rows),
                "positive_rate": positives / len(rows),
            }
        )
    return bins


def binary_classification_report(
    labels: list[int],
    predictions: list[float],
    prefix: str,
    include_calibration: bool = True,
) -> dict[str, object]:
    report: dict[str, object] = {
        f"{prefix}_positives": sum(labels),
        f"{prefix}_positive_rate": sum(labels) / len(labels) if labels else 0.0,
        f"{prefix}_auc": binary_auc(labels, predictions),
        f"{prefix}_pr_auc": binary_pr_auc(labels, predictions),
        f"{prefix}_log_loss": binary_log_loss(labels, predictions),
    }
    for key, value in lift_at_k(labels, predictions).items():
        report[f"{prefix}_{key}"] = value
    if include_calibration:
        report[f"{prefix}_calibration_bins"] = calibration_bins(labels, predictions)
    return report
