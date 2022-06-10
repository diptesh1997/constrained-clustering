import numpy as np
import pandas as pd
import os

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import time


def word2vec_features(df):
    start = time.time()
    df['text_list']= df['text'].apply(lambda x: [item for item in str(x).split()])
    model = Word2Vec(sentences=dataset['text_list'], vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
    model = Word2Vec.load("word2vec.model")
    #model.train([["hello", "world"]], total_examples=1, epochs=1)
    model.init_sims(replace = True)
    pass

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
    