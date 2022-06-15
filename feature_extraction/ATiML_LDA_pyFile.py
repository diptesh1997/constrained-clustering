import pandas as pd
from gensim.models import Phrases
from gensim import corpora
from gensim import models
import numpy as np

data = pd.read_csv(r'../../Data/data-2.csv')
data['tokens_text'] = data['text'].str.split(' ')
tokens = data['tokens_text'].tolist()
bigram_model = Phrases(tokens)
trigram_model = Phrases(bigram_model[tokens], min_count=1)
tokens = list(trigram_model[bigram_model[tokens]])
dictionary_LDA = corpora.Dictionary(tokens)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
np.random.seed(123456)
num_topics = 20
lda_model = models.LdaModel(corpus, num_topics=num_topics,
                            id2word=dictionary_LDA,
                            passes=4, alpha=[0.01] * num_topics,
                            eta=[0.01] * len(dictionary_LDA.keys()))

topics = [lda_model[corpus[i]] for i in range(len(data))]


def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res


# Final LDA feature extraction matrix with documents as rows and 10 topics as column (can change count of topics by num_topics variable value)
document_topic = pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]).reset_index(drop=True).fillna(0)
document_topic['docno']=data['docno']

document_topic.to_csv("LDA_features.csv",index=False)