# Cross-Attention Transformer Baseline (Sean Costello)

Cross-Attention Transformer with multi-head attention, trained on 880 NL/SQL pairs.

My thoughts for using a Multi-Headed attention Transformer was that if we make it Multi-Headed it allows the model to be able to capture multiple relationships from each piece of Data. The reasoning behind this is to think of it like having a single researcher with a single specialty trying to solve a problem (Single Attention) or we could have several researchers with different specialties to solve the same problem (Multi-Headed). Because we are trying to convert Natural Language of different ranges into the same/similar SQL queries if we run a M-H Attention it can look at different relationships at the same time for each individual word, instead of one blanket relationship. I believe this will be much more useful when you consider slang and different words/sentences meaning the same thing./

M-H Cross-Attention Transformers are often used in Language Translation, which at it's most basic for that is what we are trying to solve here but instead of English to Spanish we are converting English to SQL. Thus far given our train/test set the results have looked promising. Even though the initial set was 880 unique NL sentences to 110 SQL queries (8 NL per 1 SQL), I think we may need to produce more data to train on for better performance and to prevent overfitting (which I believe may be happening here).

## Results

| Test Set | N | Execution Accuracy |
|---|---|---|
| 32 Key Dim Test set | 220 | 92.7% (204/220) |
| 64 Key Dim Test set | 220 | 93.2% (205/220) |

## Run

```bash
python models/transformer/sc_sql_transformer_v1.ipynb
```

## Sources
https://www.tensorflow.org/text/tutorials/transformer /
https://www.geeksforgeeks.org/nlp/multi-head-attention-mechanism/
