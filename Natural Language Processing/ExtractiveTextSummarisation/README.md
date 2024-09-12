# TextSummarisation

1. textsummarisation.ipynb 
* Extractive Summarisation - selecting a subset of sentences /extracts objects from the entire collection
* Examples showcasing use of summarize function from gensim.summarization, exploring parameters like ratio, word_count. 
* Gensim implementation makes use of text rank algorithm which is based on PageRank algorithm.


2.Text_Summarisation_Example2.ipynb --
* extractive summarisation example
* takes all articles in the dataframe and split into sentences followed by text preprocessing.
* uses glove word embeddings and creates a sentence vector by averaging values over all words.(Vector representation of Sentences - first fetch vectors (each of size 100 elements) for the constituent words in a sentence and then take mean/average of those vectors to arrive at a consolidated vector for the sentence.)
* calculates similarity score between every possible combination 
* extract the top N sentences based on their rankings(obtained from pagerank algo) for summary generation
