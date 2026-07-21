"""
Independent verification of the supervised models' committed predictions.

Re-executes the GRU's and Transformer's committed prediction files against the
live nba_spatial database and reports, per model:

  1. Exact match          — normalized string equality (lowercase, collapsed
                            whitespace, trailing semicolon stripped)
  2. Execution accuracy (as emitted)
                          — predicted SQL executed verbatim; result must match
                            the gold SQL's result under the strict matcher
  3. Execution accuracy (literal-case-fixed)
                          — same, after mechanically restoring case inside
                            string literals (three-letter alphabetic literals
                            to uppercase team codes, others to title case);
                            isolates the lowercasing defect of Error Class 4b
  4. Corpus BLEU          — sacreBLEU, both sides lowercased

Prediction files verified (committed to this repo):
  - evaluation/results/sean_transformer_predictions_32dim.txt
  - evaluation/results/sean_transformer_predictions_64dim.txt
  - Data/Results/cosql_nmt_gru_seq2seq_model2_results.txt

Usage:  python evaluation/verify_committed_predictions.py
"""

import os
import re
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.few_shot_pipeline.evaluate import results_match, run_query

load_dotenv()

REPO = Path(__file__).parent.parent

FILES = [
    ("Transformer (32-dim)", REPO / "evaluation/results/sean_transformer_predictions_32dim.txt", "sean"),
    ("Transformer (64-dim)", REPO / "evaluation/results/sean_transformer_predictions_64dim.txt", "sean"),
    ("GRU seq2seq (fixed)",  REPO / "Data/Results/cosql_nmt_gru_seq2seq_model2_results.txt",     "craig"),
]


def parse_sean(path):
    """Blocks of: '✓/✗ Q: ...' / 'GT: ...' / 'PRED: ...'"""
    pairs = []
    gt = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("GT:"):
            gt = s[3:].strip()
        elif s.startswith("PRED:"):
            pairs.append((gt, s[5:].strip()))
    return pairs


def parse_craig(path):
    """Blocks of: 'Translating sentence N: ...' / 'expected output: ...' / 'Translated: ...'"""
    pairs = []
    gt = None
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.lower().startswith("expected output:"):
            gt = s.split(":", 1)[1].strip()
        elif s.lower().startswith("translated:"):
            pairs.append((gt, s.split(":", 1)[1].strip()))
    return pairs


def normalize_sql(sql):
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower())


def fix_literal_case(sql):
    """Restore case inside single-quoted string literals: 3-letter alphabetic
    literals -> uppercase (team codes); others -> title case (player names)."""
    def repl(m):
        lit = m.group(1)
        if len(lit) == 3 and lit.isalpha():
            return "'" + lit.upper() + "'"
        return "'" + lit.title() + "'"
    return re.sub(r"'([^']*)'", repl, sql)


def main():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "nba_spatial"),
        user=os.getenv("DB_USER", "rosalinatorres"), password=os.getenv("DB_PASSWORD", ""),
    )
    print("✅ Connected to nba_spatial\n")

    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None
        print("⚠️  sacrebleu not installed — BLEU will be skipped\n")

    for label, path, fmt in FILES:
        pairs = parse_sean(path) if fmt == "sean" else parse_craig(path)
        n = len(pairs)
        exact = raw_exec = fixed_exec = 0
        preds, refs = [], []

        for gt, pred in pairs:
            preds.append(pred.lower())
            refs.append(gt.lower())
            if normalize_sql(gt) == normalize_sql(pred):
                exact += 1
            gold_rows = run_query(conn, gt)
            if results_match(gold_rows, run_query(conn, pred)):
                raw_exec += 1
            if results_match(gold_rows, run_query(conn, fix_literal_case(pred))):
                fixed_exec += 1

        bleu = sacrebleu.corpus_bleu(preds, [refs]).score if sacrebleu else float("nan")
        print(f"{label}  (n={n}, {path.relative_to(REPO)})")
        print(f"  Exact match:                    {exact}/{n}  ({exact/n:.1%})")
        print(f"  Execution accuracy (as emitted): {raw_exec}/{n}  ({raw_exec/n:.1%})")
        print(f"  Execution accuracy (case-fixed): {fixed_exec}/{n}  ({fixed_exec/n:.1%})")
        print(f"  Corpus BLEU (lowercased):        {bleu:.1f}\n")

    conn.close()


if __name__ == "__main__":
    main()
