

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
import time
from sklearn.preprocessing import MinMaxScaler

start = time.time()

def vectorize(list_of_docs, model):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features
dataset= pd.read_csv("../../Data/data-2.csv")

dataset['text_list']= dataset['text'].apply(lambda x: [item for item in str(x).split()])

model = Word2Vec(sentences=dataset['text_list'], vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
#model.train([["hello", "world"]], total_examples=1, epochs=1)
model.init_sims(replace = True)

vectorized_docs = vectorize(dataset['text_list'], model=model)
dataframe = pd.DataFrame(vectorized_docs)

dataframe=pd.DataFrame(vectorized_docs)

dataframe['docno']=dataset['docno']
dataframe=dataframe.set_index('docno')
std_scaler = MinMaxScaler()
 
df_scaled = pd.DataFrame(std_scaler.fit_transform(dataframe.to_numpy()),columns=dataframe.columns,index=dataframe.index)
df_scaled.to_csv('Word2Vec.csv',index=True)


