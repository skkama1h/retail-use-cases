# Qualitative Assessment of Text Summarizations
Perform a qualitative assessment of a candidate summarization by comparing it to a reference response. Metrics calculated: 

__BLEU:__ Measures the overlap of n-grams between a candidate text and reference text, emphasizing precision and brevity in translation tasks.

__ROUGE-N:__ Evaluates the overlap of n-grams between candidate and reference texts, focusing on recall to assess content similarity in summarization tasks.

__BERTScore:__ Uses contextual embeddings to compare semantic similarity between cadidate and reference texts, capturing meaning beyond exact matches.

## Installation
Get started by installing conda and setting up your python enviornment.
```
bash install.sh
```

## Run Examples
The following script will by default compare the file "candidate_summarization.txt" against the file "reference_summarization.txt". 
```
python run.py
```
To compare different text documents, set the following arguments:

    -c, --candidate_response CANDIDATE_RESPONSE
                        Path to a candidate summarization response to test.
    -r, --reference_response REFERENCE_RESPONSE
                        Path to a reference summarization.
