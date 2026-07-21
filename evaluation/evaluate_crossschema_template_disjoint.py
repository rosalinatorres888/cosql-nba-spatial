"""
Template-disjoint cross-schema evaluation — few-shot pipeline on the 880-pair
boxscores corpus, with WHOLE TEMPLATES held out.

Motivation (docs/TEMPLATE_OVERLAP.md): the shared per-template 80/20 split
places every test pair's exact gold SQL in training, so it measures
paraphrase robustness. This script instead holds out 20% of template_ids
entirely — the model has never seen the test questions' SQL patterns among
its in-context examples — measuring generalization to novel query structures.

Protocol:
  - Split: 20% of the 110 template_ids held out whole (seed 42)
    -> 88 train templates (704 pairs) / 22 test templates (176 pairs)
  - Few-shot: 3 examples drawn from TRAIN templates only
  - Prompt: boxscores/player_boxscores schema (as in the strict evaluator)
  - Metrics: execution accuracy (strict matcher), SQL validity rate,
    and corpus BLEU (sacreBLEU, case-sensitive) against gold SQL strings

Usage:  python evaluation/evaluate_crossschema_template_disjoint.py
"""

import os
import sys
import random
from pathlib import Path

import pandas as pd
import psycopg2
import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.few_shot_pipeline.evaluate import results_match, run_query
from evaluation.evaluate_crossschema_strict import build_prompt

load_dotenv()

DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "nba_spatial")
DB_USER     = os.getenv("DB_USER", "rosalinatorres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CRAIG_CSV   = Path(__file__).parent.parent / "Data/Inputs/sql_training_full.csv"
OUTPUT_FILE = Path(__file__).parent / "results/crossschema_template_disjoint_results.txt"
RANDOM_SEED = 42
N_EXAMPLES  = 3
HOLDOUT_FRACTION = 0.2


def split_templates(csv_path):
    """Hold out whole templates: no test template appears in the train pool."""
    df = pd.read_csv(csv_path)
    templates = sorted(df["template_id"].unique())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(templates)
    n_test = int(len(templates) * HOLDOUT_FRACTION)
    test_templates = set(templates[:n_test])
    train_df = df[~df["template_id"].isin(test_templates)].reset_index(drop=True)
    test_df = df[df["template_id"].isin(test_templates)].reset_index(drop=True)
    return train_df, test_df, test_templates


def main():
    print("=" * 60)
    print("TEMPLATE-DISJOINT CROSS-SCHEMA EVALUATION")
    print("Few-shot pipeline on the 880-pair boxscores corpus")
    print("=" * 60)

    train_df, test_df, test_templates = split_templates(CRAIG_CSV)
    train_pool = list(zip(train_df["question"], train_df["sql_query"]))
    test_pairs = list(zip(test_df["question"], test_df["sql_query"]))
    print(f"Train: {train_df['template_id'].nunique()} templates / {len(train_pool)} pairs")
    print(f"Test:  {len(test_templates)} templates / {len(test_pairs)} pairs (all templates unseen)\n")

    client = anthropic.Anthropic()
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME,
                            user=DB_USER, password=DB_PASSWORD)
    print("✅ Connected to nba_spatial\n")

    rng = random.Random(RANDOM_SEED)
    valid = match = 0
    predictions, references = [], []
    lines = []

    for i, (question, gold_sql) in enumerate(test_pairs):
        examples = rng.sample(train_pool, N_EXAMPLES)
        try:
            resp = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=512,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": build_prompt(question, examples)}],
            )
            predicted = next((b.text.strip() for b in resp.content if b.type == "text"), "")
            if predicted.startswith("```"):
                predicted = predicted.split("```")[1].strip()
                if predicted.lower().startswith("sql"):
                    predicted = predicted[3:].strip()
        except Exception as e:
            predicted = ""
            print(f"  ⚠️  API error on Q{i+1}: {e}")

        predictions.append(predicted)
        references.append(gold_sql)

        gold_rows = run_query(conn, gold_sql)
        pred_rows = run_query(conn, predicted)
        is_valid = pred_rows is not None
        is_match = results_match(gold_rows, pred_rows)
        valid += is_valid
        match += is_match

        status = "✅" if is_match else ("⚠️ valid, wrong result" if is_valid else "❌ execution error")
        line = (f"[{i+1:3d}/{len(test_pairs)}] {status}\n"
                f"  NL:   {question}\n  Gold: {gold_sql}\n  Pred: {predicted}\n")
        print(line)
        lines.append(line)

    n = len(test_pairs)
    bleu_line = "  BLEU: sacrebleu not installed — pip install sacrebleu"
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        bleu_line = f"  Corpus BLEU (sacreBLEU): {bleu.score:.2f}"
    except ImportError:
        pass

    summary = (
        f"\n{'=' * 60}\nTEMPLATE-DISJOINT RESULTS\n{'=' * 60}\n"
        f"  Test: {len(test_templates)} held-out templates / {n} pairs\n"
        f"  SQL validity rate:   {valid}/{n} ({valid/n:.1%})\n"
        f"  Execution accuracy:  {match}/{n} ({match/n:.1%})\n"
        f"{bleu_line}\n"
    )
    print(summary)
    lines.append(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"📄 Results written to: {OUTPUT_FILE}")
    conn.close()


if __name__ == "__main__":
    main()
