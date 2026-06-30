# Transformer Baseline (Sean Costello)

Transformer encoder-decoder with multi-head attention, trained on 880 NL/SQL pairs.

## Results

| Test Set | N | Execution Accuracy |
|---|---|---|
| GRU test set | 220 | 92.7% (204/220) |

## Run

```bash
python models/transformer/train.py
python models/transformer/evaluate.py
```
