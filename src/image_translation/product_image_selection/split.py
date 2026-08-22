import hashlib
from collections import defaultdict
import numpy as np
from .exceptions import ManifestError

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupUnit:
    group_id: str
    total_count: int
    taken_count: int
    not_taken_count: int


def pixel_hash(pixels):
    pixels = np.ascontiguousarray(pixels, dtype="uint8")
    return hashlib.sha256(str(pixels.shape).encode() + b"\0" + pixels.tobytes()).hexdigest()


class SplitResult(dict):
    """Mapping-compatible split result with auditable assignment diagnostics."""

    def __init__(self, *args, diagnostics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics = diagnostics or {}


def grouped_split(records, seed=0, fractions=(.7, .15, .15), minimum_groups_per_class=0):
    if abs(sum(fractions) - 1) > 1e-6 or any(x <= 0 for x in fractions):
        raise ManifestError("split fractions must be positive and sum to 1")
    group_counts=defaultdict(lambda: defaultdict(int))
    for r in records:
        group_counts[r.product_group_id][r.label] += 1
    labels = tuple(sorted({label for counts in group_counts.values() for label in counts}, key=lambda x: x.value))
    if set(x.value for x in labels) != {"taken", "not_taken"}:
        raise ManifestError("both classes are required")
    required_groups = max(1, minimum_groups_per_class)
    units = {
        group: GroupUnit(group, sum(counts.values()), counts.get(type(labels[0])("taken"), 0),
                         counts.get(type(labels[0])("not_taken"), 0))
        for group, counts in group_counts.items()
    }
    if any(sum(unit.taken_count > 0 if label.value == "taken" else unit.not_taken_count > 0
               for unit in units.values()) < required_groups * 3 for label in labels):
        raise ManifestError("insufficient groups per class for configured splits")
    split_names = ("train", "validation", "test")
    groups = sorted(units, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    total_targets = [len(records) * f for f in fractions]
    class_totals = {label.value: sum(getattr(u, f"{label.value}_count") for u in units.values()) for label in labels}
    class_targets = {label: [class_totals[label] * f for f in fractions] for label in class_totals}
    assignment = {}
    totals = [0, 0, 0]
    classes = [defaultdict(int) for _ in range(3)]

    def can_still_support(candidate, index, remaining):
        trial = [dict(x) for x in classes]
        trial[index]["taken"] = trial[index].get("taken", 0) + candidate.taken_count
        trial[index]["not_taken"] = trial[index].get("not_taken", 0) + candidate.not_taken_count
        for target in range(3):
            for label in ("taken", "not_taken"):
                have = sum(1 for group, split in assignment.items() if split == split_names[target]
                           and getattr(units[group], f"{label}_count") > 0)
                if target == index and getattr(candidate, f"{label}_count") > 0:
                    have += 1
                available = sum(getattr(units[g], f"{label}_count") > 0 for g in remaining)
                if have + available < required_groups:
                    return False
        return True

    # Establish support with indivisible units first. This is one assignment
    # pass: a mixed-label unit satisfies both counters without being reused.
    for index in range(3):
        for label in ("taken", "not_taken"):
            candidates = [g for g in groups if g not in assignment
                          and getattr(units[g], f"{label}_count") > 0]
            if not candidates:
                raise ManifestError("grouped stratification cannot satisfy class support")
            group = min(candidates, key=lambda g: (
                -int(units[g].taken_count > 0 and units[g].not_taken_count > 0),
                hashlib.sha256(f"{seed}:{g}:{index}:{label}".encode()).hexdigest()))
            unit = units[group]
            assignment[group] = split_names[index]
            totals[index] += unit.total_count
            classes[index]["taken"] += unit.taken_count
            classes[index]["not_taken"] += unit.not_taken_count

    remaining_groups = [g for g in groups if g not in assignment]
    for position, group in enumerate(remaining_groups):
        unit = units[group]
        remaining = remaining_groups[position + 1:]
        choices=[]
        for index in range(3):
            trial_totals = totals.copy()
            trial_totals[index] += unit.total_count
            trial_classes = [dict(x) for x in classes]
            trial_classes[index]["taken"] = trial_classes[index].get("taken", 0) + unit.taken_count
            trial_classes[index]["not_taken"] = trial_classes[index].get("not_taken", 0) + unit.not_taken_count
            deviation = sum(abs(trial_totals[i] - total_targets[i]) for i in range(3))
            deviation += sum(abs(trial_classes[i].get(label, 0) - class_targets[label][i])
                             for i in range(3) for label in class_targets)
            choices.append((deviation, index))
        if not choices: raise ManifestError("grouped stratification cannot satisfy class support")
        index=min(choices)[1]
        assignment[group] = split_names[index]
        totals[index] += unit.total_count
        classes[index]["taken"] += unit.taken_count
        classes[index]["not_taken"] += unit.not_taken_count
    if set(assignment) != set(groups) or len(set(assignment.values())) < 3:
        raise ManifestError("every group must be assigned once to non-empty splits")
    result = SplitResult({s: [r for r in records if assignment[r.product_group_id] == s] for s in split_names})
    if any(len(result[s]) == 0 or {r.label.value for r in result[s]} != {"taken", "not_taken"} for s in result):
        raise ManifestError("every split must be non-empty")
    if any(sum(r.label.value == label for r in result[s]) == 0 for s in split_names for label in ("taken", "not_taken")):
        raise ManifestError("every split must contain both labels")
    if any(sum(units[g].__getattribute__(f"{label}_count") > 0 for g, split in assignment.items() if split == s) < required_groups
           for s in split_names for label in ("taken", "not_taken")):
        raise ManifestError("configured minimum groups per class is not satisfied")
    result.diagnostics = {
        "target_totals": dict(zip(split_names, total_targets)),
        "actual_totals": dict(zip(split_names, totals)),
        "target_class_counts": {label: dict(zip(split_names, target)) for label, target in class_targets.items()},
        "actual_class_counts": {label: {s: classes[i].get(label, 0) for i, s in enumerate(split_names)} for label in class_targets},
        "group_assignments": dict(assignment),
    }
    return result

def validate_pixel_leakage(records, decoded_pixels, splits=None):
    seen={}
    for record, pixels in zip(records, decoded_pixels):
        digest=pixel_hash(pixels)
        previous=seen.get(digest)
        if previous and previous.label != record.label:
            raise ManifestError("identical decoded pixels have conflicting labels")
        if previous and previous.product_group_id != record.product_group_id:
            raise ManifestError("identical decoded pixels cross product groups")
        seen[digest]=record
    if splits is not None:
        owners = {}
        hashes_by_id = {r.image_id: pixel_hash(p) for r, p in zip(records, decoded_pixels)}
        for name, rows in splits.items():
            for row in rows:
                if row.product_group_id in owners and owners[row.product_group_id] != name:
                    raise ManifestError("product group crosses split")
                owners[row.product_group_id] = name
                digest = hashes_by_id[row.image_id]
                if digest in owners and owners[digest] != name:
                    raise ManifestError("identical pixels cross split")
                owners[digest] = name
    return True

def validate_no_leakage(splits, pixel_hashes=None):
    owners={}
    for name, rows in splits.items():
        for row in rows:
            if row.product_group_id in owners and owners[row.product_group_id] != name:
                raise ValueError("product group crosses split")
            owners[row.product_group_id]=name
    if pixel_hashes:
        seen={}
        for name, rows in splits.items():
            for row in rows:
                digest=pixel_hashes[row.image_id]
                if digest in seen and seen[digest] != name: raise ValueError("identical pixels cross split")
                seen[digest]=name
