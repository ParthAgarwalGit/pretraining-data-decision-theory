"""Recompute the cross-scale experiment using only Python's standard library.

Input is the audited final-checkpoint score table included in the repository.
Run from any working directory. Optional output is written to a new directory.
"""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "empirical_study"
AGGREGATE = "olmes_10_macro_avg"


def average_ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in ordered[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def load_scores():
    groups = defaultdict(list)
    keys = set()
    with (RESULTS / "analysis_observations.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["task"], int(row["params_num"]), row["recipe"])
            observation = (*key, row["seed"])
            if observation in keys:
                raise ValueError(f"Duplicate observation: {observation}")
            keys.add(observation)
            score = float(row["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid score: {observation}")
            groups[key].append(score)
    if len(keys) != 11550 or any(len(values) != 3 for values in groups.values()):
        raise ValueError("Expected the audited complete panel of 11,550 scores.")
    # Match the analysis pipeline's compensated summation, including exact ties.
    # Different last-bit rounding can otherwise split tied recipe means.
    means = {}
    for key, values in groups.items():
        total = 0.0
        correction = 0.0
        for value in values:
            adjusted = value - correction
            updated = total + adjusted
            correction = (updated - total) - adjusted
            total = updated
        means[key] = total / len(values)
    return means


def compare(scores, task, source, target, recipes):
    source_values = [scores[task, source, recipe] for recipe in recipes]
    target_values = [scores[task, target, recipe] for recipe in recipes]
    selected = max(range(len(recipes)), key=source_values.__getitem__)
    target_best = max(target_values)
    return {
        "task": task,
        "source": source,
        "target": target,
        "spearman": statistics.correlation(average_ranks(source_values), average_ranks(target_values)),
        "selected_recipe": recipes[selected],
        "target_winners": " | ".join(recipe for recipe, value in zip(recipes, target_values) if value == target_best),
        "selected_target_score": target_values[selected],
        "best_target_score": target_best,
        "top1": int(target_values[selected] == target_best),
        "regret_pp": 100 * (target_best - target_values[selected]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="530M", help="Smaller model size, e.g. 4M, 530M or 750M")
    parser.add_argument("--task", help="Show a single task, e.g. arc_easy or boolq. Default shows ten individual tasks.")
    parser.add_argument("--output", type=Path, help="New directory for a CSV, JSON and browser-readable summary")
    args = parser.parse_args()
    scores = load_scores()
    sizes = sorted({key[1] for key in scores})
    labels = {f"{size // 1000000}M": size for size in sizes[:-1]}
    tasks = sorted({key[0] for key in scores})
    recipes = sorted({key[2] for key in scores})
    if args.source not in labels:
        parser.error(f"Choose a source from {', '.join(labels)}. Target is fixed at 1B.")
    if args.task and args.task not in tasks:
        parser.error(f"Choose a task from {', '.join(tasks)}")
    rows = [compare(scores, task, size, sizes[-1], recipes) for task in tasks for size in sizes[:-1]]
    by_key = {(row["task"], row["source"]): row for row in rows}
    with (RESULTS / "cross_scale_predictability.csv").open(encoding="utf-8", newline="") as stream:
        expected = list(csv.DictReader(stream))
    if len(expected) != len(rows):
        raise ValueError("Reference result coverage differs from the observed panel.")
    for reference in expected:
        actual = by_key[reference["task"], int(reference["params_num"])]
        for calculated, stored in [("spearman", "spearman"), ("top1", "top1")]:
            if not math.isclose(actual[calculated], float(reference[stored]), abs_tol=1e-10):
                raise ValueError(f"Reference mismatch: {reference['task']}, {reference['params_num']}, {stored}")
        if not math.isclose(actual["regret_pp"], 100 * float(reference["regret"]), abs_tol=1e-10):
            raise ValueError("Regret differs from the saved reference.")
    selected_tasks = [args.task] if args.task else [task for task in tasks if task != AGGREGATE]
    selected = [by_key[task, labels[args.source]] for task in selected_tasks]
    summary = {
        "observations_read": 11550,
        "comparisons_verified": len(rows),
        "source": args.source,
        "target": "1B",
        "tasks": len(selected),
        "mean_spearman": statistics.mean(row["spearman"] for row in selected),
        "winner_matches": sum(row["top1"] for row in selected),
        "mean_regret_pp": statistics.mean(row["regret_pp"] for row in selected),
    }
    print("Read 11,550 seed-level scores; recomputed and verified all 143 task/size comparisons.")
    print(f"\nSource {args.source}; fixed target 1B; {len(selected)} task(s); 25 recipes; 3 seeds per recipe.")
    print(f"{'Task':22} {'Spearman':>10} {'Winner match':>14} {'Regret (pp)':>13}")
    for row in selected:
        print(f"{row['task']:22} {row['spearman']:10.3f} {row['top1']:14} {row['regret_pp']:13.3f}")
    print(f"\nMean correlation {summary['mean_spearman']:.3f}; winners {summary['winner_matches']}/{len(selected)}; mean regret {summary['mean_regret_pp']:.3f} percentage points.")
    for row in selected:
        print(f"\n{row['task']}: selected {row['selected_recipe']}; target best {row['target_winners']}")
    print("\nTarget scores are three-seed reference means. Task comparisons share models and are dependent.")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        with (args.output / "comparisons.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        body = "".join(f"<tr><td>{html.escape(row['task'])}</td><td>{row['spearman']:.3f}</td><td>{row['top1']}</td><td>{row['regret_pp']:.3f}</td><td>{html.escape(row['selected_recipe'])}</td></tr>" for row in selected)
        document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>DataDecide live demonstration</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef2f5}}</style>
<h1>Recipe selection from {html.escape(args.source)} to 1B</h1>
<p>Recomputed from 11,550 seed-level observations. All 143 comparisons matched the reference results.</p>
<p>Mean Spearman {summary['mean_spearman']:.3f}. Winner matches {summary['winner_matches']}/{len(selected)}. Mean regret {summary['mean_regret_pp']:.3f} percentage points.</p>
<table><tr><th>Task</th><th>Spearman</th><th>Winner match</th><th>Regret (pp)</th><th>Selected recipe</th></tr>{body}</table>
<p>Source and target rankings use the same 25 recipes and three-seed means. Target scores are noisy reference measurements. The default average excludes the aggregate task.</p></html>"""
        (args.output / "demo.html").write_text(document, encoding="utf-8")
        print(f"Saved {args.output.resolve() / 'demo.html'}")


if __name__ == "__main__":
    main()
