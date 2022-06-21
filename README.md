** Introducing constraints into k-means clustering and evaluating with TREC 4 5 data set **

![Poster](./poster.png)





*Usage:* 

*Step 1: Install Dependencies:* 

1. scikit-learn
2. Gensim
3. python 3

*Step2: Download the TREC 4 5 dataset*

*Step 3: Data Pre-processing*

1. Make sure the TREC_4_5 dataset is in the base folder
2. preprocessing.py extracts article data along with document id from the xml tags
3. output is data.csv

Keyphrase relevant pre-processing: 

Use `mode` flag assigned as `kp` for keyphrase specific pre-processing. It doesn't include stopword removal and stemming. Keyphrases extracted from documents are further used as a weighted constraint while performing clustering.

*Step 4: Feature extraction*

Since our data is constant and we are building offline. Feature extraction step happens once for the dataset. We built 2 ml piplines for evaluation which consume the extracted features for clustering and then further for evaluation.

Feature extraction for Pipeline 1: 

Bag of words with TF-IDF weighting, feature vector size = 1000 features

Feature extraction for Pipeline 2:

Topic Modelling with a combination of features extracted from both word2Vec and Latent Discriminent Analysis.
Hyper-parameters: 

Eeach feature extraction pipelines generates a csv file as it's output which is further used for clustering.

*Step 5: Constraint Extraction:*

* Sentiments derived from news articles. Pairwise clustering of positive articles

2. **Keyphrasification**: Please check out the comments in `keyphrase_extraction.py` before using the routine.
