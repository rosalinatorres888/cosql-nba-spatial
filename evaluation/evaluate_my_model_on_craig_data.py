"""
Rosalina's NL2SQL model — evaluated on Craig's 880-pair dataset
Tests whether the few-shot LLM approach generalizes to a different
training distribution (Craig's auto-generated paraphrases) and a
different schema slice (boxscores / player_boxscores), not just
the 138 hand-annotated pairs it was designed around.

Does NOT touch model/nl2sql.py or model/evaluate.py — additive only.
Uses the same train/test split logic as Craig's scripts (80/20 per
template_id, seed=42) so results are directly comparable.
"""

import os
import random
import psycopg2
import pandas as pd
from dotenv import load_dotenv

from model.nl2sql import NL2SQL

load_dotenv()

DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "nba_spatial")
DB_USER     = os.getenv("DB_USER", "rosalinatorres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CRAIG_CSV   = "my teams models/CoSQL-NBA/Data/Inputs/sql_training_full.csv"
OUTPUT_FILE = "evaluation/results/my_model_on_craig_data_results.txt"
RANDOM_SEED = 42


def split_data(csv_path):
    df = pd.read_csv(csv_path)
    train_rows, test_rows = [], []
    random.seed(RANDOM_SEED)

    for template_id in df['template_id'].unique():
        group = df[df['template_id'] == template_id]
        if len(group) >= 2:
            train = group.sample(n=min(6, len(group) - 2), random_state=RANDOM_SEED)
            test  = group.drop(train.index)
        else:
            train = group
            test  = pd.DataFrame()
        train_rows.append(train)
        test_rows.append(test)

    return pd.concat(train_rows).reset_index(drop=True), \
           pd.concat(test_rows).reset_index(drop=True)


def connect():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )


def try_execute(conn, sql):
    sql = sql.strip().rstrip(';')
    if not sql:
        return False, "empty prediction"
    try:
        cur = conn.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        cur.close()
        return True, result
    except Exception as e:
        conn.rollback()
        return False, str(e)


def main():
    print("=" * 60)
    print("Rosalina's NL2SQL model — evaluated on Craig's dataset")
    print("=" * 60)

    _, test_df = split_data(CRAIG_CSV)
    test_pairs = list(zip(test_df['question'], test_df['sql_query']))
    print(f"Test pairs: {len(test_pairs)} (same split logic as Craig's evals)\n")

    model = NL2SQL()
    conn = connect()
    print("✅ Connected to nba_spatial\n")

    os.makedirs("evaluation/results", exist_ok=True)

    executed, failed = 0, 0
    lines = []
    sep = "=" * 50

    for i, (question, gold_sql) in enumerate(test_pairs):
        try:
            predicted = model.predict(question)
        except Exception as e:
            predicted = ""
            print(f"  ⚠️  Prediction error on Q{i+1}: {e}")

        ok, result = try_execute(conn, predicted)
        if ok:
            executed += 1
        else:
            failed += 1

        status = f"✅ SUCCESS — {result[:2] if isinstance(result, list) else result}" \
                 if ok else f"❌ FAILED  — {result}"

        block = (
            f"\n{sep}\n"
            f"Question {i+1}: {question}\n"
            f"Gold SQL (Craig's):  {gold_sql}\n"
            f"Predicted SQL (mine): {predicted}\n"
            f"DB execution:        {status}\n"
            f"{sep}"
        )
        lines.append(block)
        print(block)

    total = len(test_pairs)
    summary = (
        f"\n{'=' * 60}\n"
        f"ROSALINA'S MODEL ON CRAIG'S DATA — RESULTS SUMMARY\n"
        f"{'=' * 60}\n"
        f"  Test pairs:         {total}\n"
        f"  DB execution rate:  {executed}/{total} ({executed/total*100:.1f}%)\n"
        f"  Failed execution:   {failed}/{total}\n"
        f"{'=' * 60}\n"
    )
    print(summary)
    lines.append(summary)

    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(lines))
    print(f"📄 Results written to: {OUTPUT_FILE}")

    conn.close()


if __name__ == "__main__":
    main()
