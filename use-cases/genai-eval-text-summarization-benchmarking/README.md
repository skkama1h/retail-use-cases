# Qualitative Assessment of Text Summarizations
Perform a qualitative assessment of a candidate summarization by comparing it to a reference response. Metrics calculated: 

__BLEU:__ Measures the overlap of n-grams between a candidate text and reference text, emphasizing precision and brevity in translation tasks.

__ROUGE-N:__ Evaluates the overlap of n-grams between candidate and reference texts, focusing on recall to assess content similarity in summarization tasks.

__BERTScore:__ Uses contextual embeddings to compare semantic similarity between cadidate and reference texts, capturing meaning beyond exact matches.

## Installation
Get started by setting up your python enviornment.
```
conda create -n qualbench python=3.11
conda activate qualbench
pip install rouge-score nltk bert-score
```

## Run Examples
```
python run.py
```