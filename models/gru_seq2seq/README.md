# GRU Seq2Seq Model Comparison (Craig Hobel)

2 GRU encoder-decoder models with Bahdanau attention, trained on 440 NL/SQL pairs, tested on 220 NL/SQL pairs. 

## cosql_nmt_gru_seq2seq_model1.ipynb
- By default this script will load...
- train data from ../../Data/Inputs/train.txt (tab delimited file with 440 NLP -> SQL pairs)
- test data from ../../Data/Inputs/test.txt (tab delimited file with 220 NLP -> SQL pairs)
- This script will output a file which contains 
	- 1 - translating sentence: (the naturally spoken NLP (english) question)
	- 2 - Actual translation: The "correct" response as provided in the test data
	- 3 - Predicted translation: The model's predicted SQL command
- Output File Location:  ../../Data/Results/cosql_nmt_gru_seq2seq_model1_results.txt

## cosql_nmt_gru_seq2seq_model2.ipynb
- By default this script will load...
- train data from ../../Data/Inputs/train.txt (tab delimited file with 440 NLP -> SQL pairs)
- test data from ../../Data/Inputs/test.txt (tab delimited file with 220 NLP -> SQL pairs)
- This script will output a file which contains 
	- 1 - translating sentence: (the naturally spoken NLP (english) question)
	- 2 - expected outcome: The "correct" response as provided in the test data
	- 3 - Translated: The model's predicted SQL command
- Output File Location:  ../../Data/Results/cosql_nmt_gru_seq2seq_model2_results.txt

## Results - output from each of the models can be found here:
- ../../Data/Results/cosql_nmt_gru_seq2seq_model1_results.txt
- ../../Data/Results/cosql_nmt_gru_seq2seq_model2_results.txt

## To Run

```bash
python models/gru_seq2seq/cosql_nmt_gru_seq2seq_model1.ipynb
python models/gru_seq2seq/cosql_nmt_gru_seq2seq_model2.ipynb
```

## Next steps
- Apply Performance Metrics to the models
- Run analysis on SQL results
- Discover errors and analyze to try to improve the models
- Fine tune training dimensions