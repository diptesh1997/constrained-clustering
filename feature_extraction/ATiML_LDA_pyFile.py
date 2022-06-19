import pandas as pd
from gensim.models import Phrases
from gensim import corpora
from gensim import models
import numpy as np

# data = pd.read_csv(r'../../Data/data-2.csv')
data = pd.read_csv(r'C:/Users/Shimony/Desktop/SHIMO/MS DE/SemII/ATiML/ATiML_Assignments/Project/ATiML_data.csv')
# data = data.iloc[:20000]
data['tokens'] = data['text'].str.split(' ')
tokens = data['tokens'].tolist()
bigram = Phrases(tokens)
trigram = Phrases(bigram[tokens], min_count=1)
tokens = list(trigram[bigram[tokens]])
dict_LDA = corpora.Dictionary(tokens)
dict_LDA.filter_extremes(no_below=3)
corpus = [dict_LDA.doc2bow(token) for token in tokens]
np.random.seed(123456)
num_topics = 20
lda = models.LdaModel(corpus, num_topics=num_topics,
                      id2word=dict_LDA,
                      passes=4, alpha=[0.01] * num_topics,
                      eta=[0.01] * len(dict_LDA.keys()))

topics = [lda[corpus[i]] for i in range(len(data))]


def topics_document_to_dataframe(topics_doc, num_topics):
    result = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_doc:
        result.loc[0, topic_weight[0]] = topic_weight[1]
    return result


# LDA feature extraction matrix with docs as rows and 20 topics as column
doc_topic_matrix = pd.concat(
    [topics_document_to_dataframe(topics_doc, num_topics=num_topics) for topics_doc in topics]).reset_index(
    drop=True).fillna(0)
doc_topic_matrix['docno'] = data['docno']

# doc_topic_matrix.to_csv("LDA_features.csv", index=False)

print(doc_topic_matrix.head())
