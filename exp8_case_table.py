"""Print per-question results of a completed experiment 8 run (standard library only)."""

import argparse
import json
from collections import Counter
from pathlib import Path


def _correct(reading, probe):
    probes = ("qa", "qb") if probe == "both" else (probe,)
    return all(reading[q]["decision"] == "target" for q in probes)


def case_table(folder, seed=7):
    lines = []
    for variant in ("initial", "replay_prepost", "competitive_prepost"):
        model = f"{variant}-s{seed}"
        path = folder / model
        val = json.loads((path / "validation.json").read_text(encoding="utf-8"))
        pairs = json.loads((path / "view_pairs.json").read_text(encoding="utf-8"))
        rows = val["rows"]
        lines.extend((f"\n## {model}", f"First exact: {val['aggregate']['first_exact']}",
                      "First step: route | true car | kind | photo | decision | logits A / B / Null"))
        for r in rows:
            if r["step"] == 0 and r["kind"] in ("self", "new_view"):
                s, car = r["after"], r["identity"].split(":")[-1]
                a, b = (s["target"], s["other"]) if car == "A" else (s["other"], s["target"])
                lines.append(f"{r['route']} {car} {r['kind']} {r['probe']} {s['decision']} "
                             f"{a:.3f} / {b:.3f} / {s['null']:.3f}")
        lines.append("Per-route predictions: true identity -> (chosen A, B, Null)")
        matrix = {}
        for row in rows:
            car = row["identity"].split(":")[-1]
            kinds = [row["kind"]]
            if row["kind"] == "new_view" and row.get("before_frame", {}).get("margin") is not None:
                kinds.append("paired_new_post")
            chosen = {"target": car, "other": "B" if car == "A" else "A", "null": "Null"}[row["after"]["decision"]]
            for kind in kinds:
                matrix.setdefault((row["route"], kind, car), Counter())[chosen] += 1
        for key, counts in sorted(matrix.items()):
            lines.append(f"{' '.join(key)} -> {counts['A']}, {counts['B']}, {counts['Null']}")
        lines.append("Controlled histories: qA / qB decisions and target-minus-other margins")
        changes = {}
        for case in pairs:
            result = case["result"]
            if result is None:
                lines.append(f"{case['identity']}: no eligible pair")
                continue
            scores = result["scores"]
            lines.append(f"{case['identity']}: A={result['a']} B={result['b']} qA={result['qa']} "
                         f"qB={result['qb']} extraB={result['extra_b']} rival={result['rival']}")
            for name, reading in scores.items():
                cells = [f"{reading[q]['decision']} ({reading[q]['margin']:+.3f})" for q in ("qa", "qb")]
                lines.append(f"  {name}: {' / '.join(cells)}")
            for src, dst in (("A", "AB"), ("B", "AB"), ("BB", "AB"), ("A", "ABB")):
                if src not in scores or dst not in scores:
                    continue
                for probe in ("qa", "qb", "both"):
                    key = (src, dst, probe)
                    transition = (_correct(scores[src], probe), _correct(scores[dst], probe))
                    changes.setdefault(key, Counter())[transition] += 1
        lines.append("Same case + query: wrong->right, right->wrong, unchanged, total")
        for (src, dst, q), counts in changes.items():
            lines.append(f"  {src}->{dst} {q}: {counts[False, True]}, {counts[True, False]}, "
                         f"{counts[True, True]+counts[False, False]}, {sum(counts.values())}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bbox", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(case_table(args.bbox, args.seed))
