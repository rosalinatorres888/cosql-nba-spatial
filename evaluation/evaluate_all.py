"""
CoSQL NBA — Cross-Model Evaluation
Runs all three models against a common test set and prints a comparison table.

Usage:
    python evaluation/evaluate_all.py

Requirements:
    - ANTHROPIC_API_KEY in environment (for few-shot pipeline)
    - PostgreSQL nba_spatial database running (for execution accuracy)
    - pip install -r requirements.txt
"""

import os
import sys

# ── Results from each model (pre-run outputs) ────────────────────────────────
# Each model's evaluate.py writes its results here.
# This script reads them and prints the comparison table.

RESULTS = {
    "Few-shot LLM (WOZ held-out)": {
        "file": "evaluation/results/few_shot_woz_results.txt",
        "test_set": "WOZ 28-pair held-out",
        "n": 28,
    },
    "Few-shot LLM (GRU test set, cold)": {
        "file": "evaluation/results/few_shot_craig_results.txt",
        "test_set": "GRU test set (cold)",
        "n": 220,
    },
    "GRU Seq2Seq": {
        "file": "evaluation/results/gru_results.txt",
        "test_set": "GRU test set",
        "n": 220,
    },
    "Transformer": {
        "file": "evaluation/results/transformer_results.txt",
        "test_set": "GRU test set",
        "n": 220,
    },
}


def parse_accuracy(filepath):
    """Read a result file and extract execution accuracy."""
    if not os.path.exists(filepath):
        return None, None
    correct, total = 0, 0
    with open(filepath) as f:
        for line in f:
            if line.startswith("✓"):
                correct += 1
                total += 1
            elif line.startswith("✗"):
                total += 1
    return correct, total


def main():
    print("\n" + "=" * 70)
    print("CoSQL NBA — Cross-Model Evaluation Results")
    print("=" * 70)
    print(f"{'Model':<35} {'Test Set':<25} {'N':>4}  {'Accuracy':>12}")
    print("-" * 70)

    for model_name, cfg in RESULTS.items():
        correct, total = parse_accuracy(cfg["file"])
        if correct is None:
            accuracy_str = "NOT RUN"
        else:
            pct = correct / total * 100
            accuracy_str = f"{correct}/{total} ({pct:.1f}%)"
        print(f"{model_name:<35} {cfg['test_set']:<25} {cfg['n']:>4}  {accuracy_str:>12}")

    print("=" * 70)
    print("\nTo regenerate results:")
    print("  python models/few_shot_pipeline/evaluate.py")
    print("  python models/gru_seq2seq/evaluate.py")
    print("  python models/transformer/evaluate.py")
    print("Then re-run this script.\n")


if __name__ == "__main__":
    main()
