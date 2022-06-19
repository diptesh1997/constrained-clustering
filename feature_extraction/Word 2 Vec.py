#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import os
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import time
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[9]:


dataset= pd.read_csv("C:/Users/Diptesh Mukherjee/Downloads/data.csv")
dataset


# In[10]:


get_ipython().run_cell_magic('time', '', "dataset['text_list']= dataset['text'].apply(lambda x: [item for item in str(x).split()])\ndataset\n")


# In[4]:


get_ipython().run_cell_magic('time', '', 'model = Word2Vec(sentences=dataset[\'text_list\'], vector_size=100, window=5, min_count=1, workers=4)\nmodel.save("word2vec.model")\nmodel = Word2Vec.load("word2vec.model")\n#model.train([["hello", "world"]], total_examples=1, epochs=1)\nmodel.init_sims(replace = True)\n')


# In[13]:


get_ipython().run_cell_magic('time', '', 'import itertools\nvector_size=[100,150,200]\nwindow=[4,6]\nmin_count=[3,5]\nparams=[vector_size,window,min_count]\ncount=1\nfor i in itertools.product(*params):\n    print(str(i) +"model "+ str(count))\n    model = Word2Vec(sentences=dataset[\'text_list\'], vector_size=i[0], window=i[1], min_count=i[2], workers=4,epochs=10)\n    model.save(\'word2vec\'+str(count)+\'.model\')\n    count+=1\n')


# In[7]:


get_ipython().run_cell_magic('time', '', '\n\ndef vectorize(list_of_docs, model):\n    \n    features = []\n\n    for tokens in list_of_docs:\n        zero_vector = np.zeros(model.vector_size)\n        vectors = []\n        for token in tokens:\n            if token in model.wv:\n                try:\n                    vectors.append(model.wv[token])\n                except KeyError:\n                    continue\n        if vectors:\n            vectors = np.asarray(vectors)\n            avg_vec = vectors.mean(axis=0)\n            features.append(avg_vec)\n        else:\n            features.append(zero_vector)\n    return features\n    \n')


# In[13]:


get_ipython().run_line_magic('time', '')
import os
models=[]
for file in os.listdir("./"):
    if file.endswith(".model"):
        models.append(file)
print(models)
vectorized_features=[]
for i in models:
    print("model",i)
    model=Word2Vec.load(i)
    model.init_sims(replace = True)
    vectorized_docs = vectorize(dataset['text_list'], model=model)
    vectorized_features.append(vectorized_docs)
    print("VECTORISED",i)

    dataframe=pd.DataFrame(vectorized_docs)

    dataframe['docno']=dataset['docno']
    dataframe=dataframe.set_index('docno')
    std_scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(std_scaler.fit_transform(dataframe.to_numpy()),columns=dataframe.columns,index=dataframe.index)
    df_scaled
    df_scaled.to_csv('Word2Vec'+str(i)+'.csv',index=True)
    print("WRITTEN",i)


# In[14]:


#from sklearn.datasets import load_digits
#from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
 
#Load Data
data = dataset= pd.read_csv("C:/Users/Diptesh Mukherjee/Downloads/data.csv")
pca = PCA(2)
 
#Transform the data
df = pca.fit_transform(data)
 
df.shape


# In[22]:


dfs=[]
for i in vectorized_features:
    dataframe=pd.DataFrame(i)
    dataframe["docno"]=dataset["docno"]
    dataframe=dataframe.set_index('docno')
    std_scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(std_scaler.fit_transform(dataframe.to_numpy()),columns=dataframe.columns,index=dataframe.index)
    dfs.append(df_scaled)


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for i in dfs:
    print(dfs.index(i))
    for num_clusters in range_n_clusters:
         print(dfs.index(i),num_clusters)
         kmeans = KMeans(n_clusters=num_clusters)
         kmeans.fit(i)
         cluster_labels = kmeans.labels_
         silhouette_avg.append(silhouette_score(i, cluster_labels))
    print("sillehoute done")
    plt.plot(range_n_clusters,silhouette_avg,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis For Optimal k')
    plt.show()


# In[20]:


import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[21]:


files=[]
for file in os.listdir("./Word2Vecword2vec"):
    if file.endswith(".csv"):
        files.append(file)
files


# In[23]:


import matplotlib.pyplot as plt
df=pd.read_csv("./Word2Vecword2vec/"+files[0],index_col="docno")
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []

for num_clusters in range_n_clusters:
         kmeans = KMeans(n_clusters=num_clusters)
         kmeans.fit(df)
         cluster_labels = kmeans.labels_
         silhouette_avg.append(silhouette_score(df, cluster_labels))
print("sillehoute done")
plt.plot(range_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()


# In[24]:


for i in files:
    df=pd.read_csv("./Word2Vecword2vec/"+i,index_col="docno")
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
    silhouette_avg = []

    for num_clusters in range_n_clusters:
             kmeans = KMeans(n_clusters=num_clusters)
             kmeans.fit(df)
             cluster_labels = kmeans.labels_
             silhouette_avg.append(silhouette_score(df, cluster_labels))
    print("sillehoute done")
    plt.plot(range_n_clusters,silhouette_avg,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis For Optimal k')
    plt.show()


# In[25]:


print(files)


# In[ ]:


print()


# In[ ]:


1,10,2,4,6

