# Finding: Template-Level Train/Test Overlap in the 880-Pair Corpus

**Author:** Rosalina Torres · **Date:** 2026-07-14 · **Status:** For team discussion — no teammate numbers changed.

## What the corpus is

`Data/Inputs/sql_training_full.csv` contains **880 rows = 110 templates × 8 NL
paraphrases**. Within a template, all 8 paraphrases map to the **identical**
gold SQL string.

## What the split does

The shared split protocol (`split_data`, seed 42) samples 6 of the 8
paraphrases per template for training and holds out 2 for test (660/220).
Measured on the reproduced split:

| Check | Result |
|---|---|
| Test pairs whose template appears in training | **100%** (220/220) |
| Test pairs whose *exact gold SQL string* appears in training | **100%** (220/220) |
| Test pairs whose exact question wording appears in training | 0% |

## What this means

Every test question's correct answer — the full SQL string, verbatim — was
seen during training. The models are never asked to produce SQL they haven't
memorized; only the *question wording* is new.

- **What the GRU 97.3% and Transformer 92.7% measure:** paraphrase
  robustness — can the model map a reworded question onto a memorized SQL
  string? That is a legitimate and reportable capability, but it is not
  SQL generation generalization.
- **What they do not measure:** performance on novel query structures,
  unseen column/filter combinations, or compositional generalization —
  the things "execution accuracy" usually implies in Text-to-SQL papers
  (Spider/CoSQL report template-disjoint or database-disjoint splits for
  exactly this reason).
- **Symmetry note:** the few-shot pipeline's own-schema number (88.5%) is
  measured on a leakage-free conversation-level split where no test
  (utterance, SQL) pair appears among its in-context examples. The
  cross-schema strict number uses the shared per-template split for
  comparability and therefore carries the same overlap caveat as the
  supervised numbers — this is stated where reported.

## Recommendation (team decision)

1. Report the current numbers explicitly as **paraphrase-level accuracy**.
2. Add a **template-disjoint split** (hold out ~22 whole templates ≈ 176
   test pairs; train on the remaining 88 templates) and report both numbers
   for all three architectures. This is the single most informative change
   for the M3 comparison — it is where the in-context vs supervised
   generalization difference should actually appear.
3. Either way, describe the split construction in the paper. A reviewer who
   reproduces the split will find the 100% SQL overlap in minutes.

## Reproduce this analysis

```python
import pandas as pd, random
df = pd.read_csv("Data/Inputs/sql_training_full.csv")
random.seed(42)
train, test = [], []
for tid in df["template_id"].unique():
    g = df[df["template_id"] == tid]
    tr = g.sample(n=min(6, len(g) - 2), random_state=42)
    train.append(tr); test.append(g.drop(tr.index))
train, test = pd.concat(train), pd.concat(test)
print((test.sql_query.isin(set(train.sql_query))).mean())   # -> 1.0
```
