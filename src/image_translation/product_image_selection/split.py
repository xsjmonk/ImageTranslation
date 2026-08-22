import hashlib
from collections import defaultdict
import numpy as np
from .exceptions import ManifestError
from collections import defaultdict
def pixel_hash(pixels):
    pixels = np.ascontiguousarray(pixels, dtype="uint8")
    return hashlib.sha256(str(pixels.shape).encode() + b"\0" + pixels.tobytes()).hexdigest()
def grouped_split(records, seed=0, fractions=(.7,.15,.15), minimum_groups_per_class=0):
    if abs(sum(fractions)-1) > 1e-6 or any(x <= 0 for x in fractions):
        raise ManifestError("split fractions must be positive and sum to 1")
    group_counts=defaultdict(lambda: defaultdict(int))
    for r in records:
        group_counts[r.product_group_id][r.label] += 1
    labels = tuple(sorted({label for counts in group_counts.values() for label in counts}, key=lambda x: x.value))
    required_groups=max(1, minimum_groups_per_class)
    if any(sum(label in counts for counts in group_counts.values()) < required_groups * 3 for label in labels):
        raise ManifestError("insufficient groups per class for configured splits")
    groups=sorted(group_counts, key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    total_targets=[len(records)*f for f in fractions]
    class_targets={label: sum(c[label] for c in group_counts.values()) for label in labels}
    class_targets={label:[class_targets[label]*f for f in fractions] for label in labels}
    assignment={}; totals=[0,0,0]; classes=[defaultdict(int) for _ in range(3)]
    reserved=set()
    for label in labels:
        candidates=[g for g in groups if label in group_counts[g]]
        if len(candidates) < 3: raise ManifestError("each class needs groups in every split")
        for index, group in enumerate(candidates[:3]):
            assignment[group]=("train","validation","test")[index]; reserved.add(group)
            totals[index]+=sum(group_counts[group].values())
            for item,count in group_counts[group].items(): classes[index][item]+=count
    remaining_groups=[g for g in groups if g not in reserved]
    for position, group in enumerate(remaining_groups):
        remaining=remaining_groups[position+1:]
        choices=[]
        for index in range(3):
            trial_totals=totals.copy(); trial_totals[index] += sum(group_counts[group].values())
            trial_classes=[dict(x) for x in classes]; 
            for label, count in group_counts[group].items(): trial_classes[index][label]=trial_classes[index].get(label,0)+count
            feasible=True
            for split in range(3):
                for label in labels:
                    available=sum(group_counts[g].get(label,0)>0 for g in remaining)
                    need=minimum_groups_per_class - sum(group_counts[g].get(label,0)>0 for g,s in assignment.items() if s==("train","validation","test")[split])
                    if split == index: need -= int(group_counts[group].get(label,0)>0)
                    if need > available: feasible=False
            if feasible:
                deviation=sum(abs(trial_totals[i]-total_targets[i]) for i in range(3))
                deviation += sum(abs(trial_classes[i].get(label,0)-class_targets[label][i]) for i in range(3) for label in labels)
                choices.append((deviation,index))
        if not choices: raise ManifestError("grouped stratification cannot satisfy class support")
        index=min(choices)[1]
        assignment[group]=("train","validation","test")[index]; totals[index]+=sum(group_counts[group].values())
        for label,count in group_counts[group].items(): classes[index][label]+=count
    if len(set(assignment.values())) < 3:
        raise ManifestError("split could not create non-empty train, validation, and test sets")
    result={s:[r for r in records if assignment[r.product_group_id]==s] for s in ("train","validation","test")}
    if any(len(result[s]) == 0 for s in result):
        raise ManifestError("every split must be non-empty")
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
        owners={}
        for name, rows in splits.items():
            for row in rows:
                if row.product_group_id in owners and owners[row.product_group_id] != name:
                    raise ManifestError("product group crosses split")
                if pixel_hashes := {pixel_hash(p) for r,p in zip(records, decoded_pixels) if r.image_id == row.image_id}:
                    for digest in pixel_hashes:
                        if digest in owners and owners[digest] != name: raise ManifestError("identical pixels cross split")
                        owners[digest]=name
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
