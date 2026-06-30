# GRU Seq2Seq Baseline (Craig Hobel)

GRU encoder-decoder with Bahdanau attention, trained on 880 NL/SQL pairs.

## Results

| Test Set | N | Execution Accuracy |
|---|---|---|
| GRU test set (fixed preprocessing) | 220 | 97.3% (214/220) |
| WOZ schema (cold) | 28 | 0% (0/28) |

## Run

```bash
python models/gru_seq2seq/train.py
python models/gru_seq2seq/evaluate.py
```
