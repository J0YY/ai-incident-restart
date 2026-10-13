#!/usr/bin/env python3
"""Create abstract slots only. DRAFT/UNRUN; no participants or outcome data.

24 independent future case families; 48 slots/arm; 8 cases/slot.
This is a budget-scenario schedule, not a randomized participant allocation.
Run: python3 studies/restart_review_validation_v1/scripts/make_schedule.py
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026100701


def build_schedule():
    rng = random.Random(SEED)
    # Labels are planning slots; actual case mapping requires human validation.
    cases = []
    for i in range(8):
        for prefix, truth in (("S", "sufficient"), ("I", "insufficient"), ("U", "unresolved")):
            cases.append({
                "case_id": f"MAIN_{prefix}{i + 1:02d}",
                "family_id": f"FUTURE_{prefix}{i + 1:02d}",
                "planned_truth_class": truth,
                "planned_advice_status": "correct" if i % 2 == 0 else "misleading",
            })
    rows = []
    for arm in ("A", "B"):
        offsets = list(range(24)) * 2
        rng.shuffle(offsets)
        for slot, offset in enumerate(offsets, 1):
            selected = [cases[(offset + j) % 24] for j in range(8)]
            rng.shuffle(selected)
            for order, case in enumerate(selected, 1):
                rows.append({"status": "DRAFT_ABSTRACT_SLOT_NO_PARTICIPANT",
                             "seed": SEED, "arm": arm,
                             "schedule_slot": f"{arm}{slot:03d}",
                             "case_order": order, **case})
    return rows


def validate(rows):
    by_slot = defaultdict(list)
    exposure = Counter()
    for row in rows:
        by_slot[row["schedule_slot"]].append(row)
        exposure[(row["arm"], row["case_id"])] += 1
    assert len(rows) == 768 and len(by_slot) == 96
    assert len(exposure) == 48 and set(exposure.values()) == {16}
    for packet in by_slot.values():
        assert len(packet) == 8
        assert len({r["family_id"] for r in packet}) == 8
        assert sorted(Counter(r["planned_truth_class"] for r in packet).values()) == [2, 3, 3]
        assert set(r["planned_advice_status"] for r in packet) == {"correct", "misleading"}
        assert sorted(r["case_order"] for r in packet) == list(range(1, 9))


if __name__ == "__main__":
    rows = build_schedule()
    validate(rows)
    output = ROOT / "data_templates" / "main_schedule_DRAFT.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")
    print("Verified 96 abstract slots, 768 case assignments, no repeated family per slot,")
    print("and 16 exposures per case per arm. No participants allocated; no results created.")
