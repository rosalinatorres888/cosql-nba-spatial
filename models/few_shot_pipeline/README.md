# Few-Shot LLM Pipeline (Rosalina Torres)

DIN-SQL-style conversational NL→SQL inference using Claude Opus 4.8.

## Results

| Test Set | N | Execution Accuracy |
|---|---|---|
| WOZ held-out (own schema) | 28 | **100% (28/28)** |
| GRU test set (cold, unseen schema) | 220 | **83.2% (183/220)** |

## Architecture

- Schema-aware few-shot prompting (DDL format)
- k=3 in-context examples selected by query class keyword matching
- Multi-turn coreference resolution via prior-SQL carry-forward
- Spatial predicate grounding (court zone coordinate ranges)

## Run

```bash
# Single query
python models/few_shot_pipeline/nl2sql.py

# Evaluation (requires PostgreSQL nba_spatial)
python models/few_shot_pipeline/evaluate.py
```

## Files

- `nl2sql.py` — inference pipeline
- `evaluate.py` — execution accuracy evaluator
