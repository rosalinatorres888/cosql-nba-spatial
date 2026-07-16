"""
Cross-schema evaluation — few-shot pipeline on Craig's boxscores corpus,
measured as TRUE EXECUTION ACCURACY with the strict matcher.

Supersedes evaluate_my_model_on_craig_data.py, whose reported 83.2% was an
execution *rate* (predicted SQL ran without error) — it never compared the
predicted result to the gold result, and the prompt contained the wrong
(shot_charts) schema, so syntactically-valid-but-unrelated SQL counted as
success.

Protocol (kept comparable to Craig's GRU evaluation):
  - Same split: 80/20 per template_id, seed 42 (Craig's split_data logic)
  - Few-shot: 3 examples drawn from the TRAIN split only
  - Prompt contains the boxscores/player_boxscores schema (the schema under
    test), never the shot_charts schema
  - Metric: execution accuracy — predicted SQL result must match gold SQL
    result under the strict matcher from models/few_shot_pipeline/evaluate.py

Caveat shared with the supervised models: the per-template split means every
test question's gold SQL also appears in training (paraphrase-level test, not
template-level). See docs/TEMPLATE_OVERLAP.md.

Usage:  python evaluation/evaluate_crossschema_strict.py
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

load_dotenv()

DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "nba_spatial")
DB_USER     = os.getenv("DB_USER", "rosalinatorres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CRAIG_CSV   = Path(__file__).parent.parent / "Data/Inputs/sql_training_full.csv"
OUTPUT_FILE = Path(__file__).parent / "results/crossschema_strict_results.txt"
RANDOM_SEED = 42
N_EXAMPLES  = 3

BOXSCORES_SCHEMA = """
Tables in PostgreSQL database 'nba_spatial' (boxscore slice):

boxscores(id SERIAL PK, game_id TEXT, game_date DATE, season TEXT,
          season_type TEXT, team_id BIGINT, team_abbreviation TEXT,
          player_id BIGINT, player_name TEXT, pts INTEGER, reb INTEGER,
          ast INTEGER, stl INTEGER, blk INTEGER, to_ INTEGER,
          fgm INTEGER, fga INTEGER, fg3m INTEGER, fg3a INTEGER,
          ftm INTEGER, fta INTEGER, min TEXT)
  -- one row per player per game; to_ = turnovers (trailing underscore)
  -- season_type values: 'Regular' or 'Playoff' (exact strings, no plural)

player_boxscores(id SERIAL PK, game_id TEXT, game_date DATE, season TEXT,
                 season_type TEXT, player_id BIGINT, player_name TEXT,
                 team_abbreviation TEXT, pts INTEGER, reb INTEGER,
                 ast INTEGER, stl INTEGER, blk INTEGER, fgm INTEGER,
                 fga INTEGER, fg3m INTEGER, fg3a INTEGER, ftm INTEGER,
                 fta INTEGER, min TEXT)
"""


def split_data(csv_path):
    """Craig's split: 80/20 per template_id, seed 42 — kept for comparability."""
    df = pd.read_csv(csv_path)
    train_rows, test_rows = [], []
    random.seed(RANDOM_SEED)
    for template_id in df["template_id"].unique():
        group = df[df["template_id"] == template_id]
        if len(group) >= 2:
            train = group.sample(n=min(6, len(group) - 2), random_state=RANDOM_SEED)
            test = group.drop(train.index)
        else:
            train = group
            test = pd.DataFrame()
        train_rows.append(train)
        test_rows.append(test)
    return (pd.concat(train_rows).reset_index(drop=True),
            pd.concat(test_rows).reset_index(drop=True))


def build_prompt(question, examples):
    examples_text = "\n".join(
        f"NL: {q}\nSQL: {s}" for q, s in examples
    )
    return f"""You are a Text-to-SQL system for NBA boxscore data.
Generate a single executable PostgreSQL SELECT query for the given question.
Return ONLY the SQL — no explanation, no markdown, no semicolon.

{BOXSCORES_SCHEMA}

Examples:
{examples_text}

Question: {question}
SQL:"""


def main():
    print("=" * 60)
    print("CROSS-SCHEMA STRICT EVALUATION")
    print("Few-shot pipeline on Craig's boxscores corpus")
    print("=" * 60)

    train_df, test_df = split_data(CRAIG_CSV)
    train_pool = list(zip(train_df["question"], train_df["sql_query"]))
    test_pairs = list(zip(test_df["question"], test_df["sql_query"]))
    print(f"Train pool: {len(train_pool)}   Test pairs: {len(test_pairs)}\n")

    client = anthropic.Anthropic()
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, database=DB_NAME,
                            user=DB_USER, password=DB_PASSWORD)
    print("✅ Connected to nba_spatial\n")

    rng = random.Random(RANDOM_SEED)
    valid = match = 0
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
    summary = (
        f"\n{'=' * 60}\nCROSS-SCHEMA STRICT RESULTS\n{'=' * 60}\n"
        f"  SQL validity rate:   {valid}/{n} ({valid/n:.1%})\n"
        f"  Execution accuracy:  {match}/{n} ({match/n:.1%})\n"
        f"  (prior 83.2% figure was validity-only, wrong-schema prompt)\n"
    )
    print(summary)
    lines.append(summary)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"📄 Results written to: {OUTPUT_FILE}")
    conn.close()


if __name__ == "__main__":
    main()
