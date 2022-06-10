# constrained-clustering

Dependencies: 

1. scikit-learn


Pre-processing steps: TREC 4 5 dataset:

1. Make sure the TREC_4_5 dataset is in the base folder
2. preprocessing.py extracts article data along with document id from the xml tags
3. output is data.csv

Keyphrase relevant pre-processing: 
Use `mode` flag assigned as `kp` for keyphrase specific pre-processing. It doesn't include stemming.

ML pipelines:

Pipeline 1: bag of words approach, feature vector = 1000, 500, 1500

Pipeline 2: word embedding approach with topic modelling i.e features as topics

Pipeline 3: LDA analysis (again a form of topic modelling with weights as feature values for topics)

Constraints:

1. Sentiments derived from news articles. Pairwise clustering of positive articles,

**Keyphrasification**: Please check out the comments in `keyphrase_extraction.py` before using the routine

