# Boston Celtics — Conversational Text-to-SQL (CoSQL)

**IE7500 Natural Language Processing · Northeastern University COE · Summer 2026**

Team: Rosalina Torres · Sean Costello · Craig Hobel

---

This project investigates multiple architectural approaches for conversational Text-to-SQL over an NBA spatial shot-chart database. The primary implementation is a CoSQL-inspired conversational pipeline that combines schema-aware prompting, few-shot in-context learning, spatial reasoning, and multi-turn coreference resolution. To provide comparative baselines, the project also implements two neural sequence-generation architectures: (1) a GRU encoder-decoder with Bahdanau attention and (2) a Transformer encoder-decoder, both trained on the same annotated corpus. All architectures are evaluated against a common execution-verified dataset, enabling direct comparison using SQL validity, execution accuracy, and BLEU-4.

---

## Project Summary

Three architectural approaches to conversational Text-to-SQL over NBA shot-chart data,
evaluated head-to-head on execution accuracy and BLEU-4.

| Model | Architecture | Test Set | Execution Accuracy |
|---|---|---|---|
| Few-shot LLM pipeline | DIN-SQL + Claude Opus 4.8 | WOZ 28-pair held-out | **100% (28/28)** |
| Few-shot LLM pipeline | DIN-SQL + Claude Opus 4.8 | GRU test set (cold) | **83.2% (183/220)** |
| GRU Seq2Seq | Encoder-decoder (Bahdanau attention) | GRU test set | 97.3% (214/220)* |
| Transformer | Encoder-decoder (multi-head attention) | GRU test set | 92.7% (204/220) |

*97.3% reflects fixed preprocessing. Original model: 0% — see `evaluation/results/`.

---

## Repository Structure

```
├── models/
│   ├── few_shot_pipeline/   # Rosalina Torres — DIN-SQL few-shot inference
│   ├── gru_seq2seq/         # Craig Hobel — GRU encoder-decoder baseline
│   └── transformer/         # Sean Costello — Transformer encoder-decoder baseline
├── evaluation/
│   ├── evaluate_all.py      # Run all three models, print comparison table
│   └── results/             # Per-model result files
├── annotation/              # 139 WOZ NL/SQL pairs across 8 query classes
├── docs/
│   ├── EVALUATION_RESULTS.md
│   ├── ANNOTATION_PROTOCOL.md
│   └── BUG_REPORT.md
├── schema.sql               # PostgreSQL schema (nba_spatial)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/rosalinatorres888/Boston-Celtics-conversational-Text-to-SQL-CoSQL-
cd Boston-Celtics-conversational-Text-to-SQL-CoSQL-
pip install -r requirements.txt
```

### Run individual models

```bash
# Few-shot LLM pipeline (requires ANTHROPIC_API_KEY)
python models/few_shot_pipeline/nl2sql.py

# GRU Seq2Seq baseline
python models/gru_seq2seq/train.py
python models/gru_seq2seq/evaluate.py

# Transformer baseline
python models/transformer/train.py
python models/transformer/evaluate.py
```

### Run full comparison evaluation

```bash
python evaluation/evaluate_all.py
```

Prints a side-by-side results table for all three models.

---

## Environment Setup

```bash
# PostgreSQL (required for few-shot pipeline execution accuracy)
createdb nba_spatial
psql nba_spatial < schema.sql

# Environment variables
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env
```

---

## Branch Strategy

| Branch | Owner | Purpose |
|---|---|---|
| `main` | Protected | Reviewed, passing code only |
| `rosalina/pipeline` | Rosalina Torres | Few-shot LLM pipeline |
| `craig/gru` | Craig Hobel | GRU encoder-decoder |
| `sean/transformer` | Sean Costello | Transformer baseline |

All merges to `main` require a pull request. No direct pushes to `main`.

---

## Team Contributions

- **Rosalina Torres** — conversational pipeline, coreference resolution, WOZ annotation corpus (139 pairs), evaluation framework, cross-model evaluation, documentation
- **Sean Costello** — Transformer encoder-decoder baseline
- **Craig Hobel** — GRU encoder-decoder baseline

---

## References

- Pourreza & Rafiei (2023). DIN-SQL: Decomposed In-Context Learning of Text-to-SQL. *NeurIPS 2023*.
- Yu et al. (2019). CoSQL: A Conversational Text-to-SQL Challenge. *EMNLP 2019*.
- Sutskever et al. (2014). Sequence to Sequence Learning with Neural Networks. *NeurIPS 2014*.
