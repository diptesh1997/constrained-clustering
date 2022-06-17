#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df


# In[2]:


from sklearn.cluster import KMeans
X = df[['sepal length (cm)', 'petal length (cm)']].values
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)


# In[3]:


centroids = model.cluster_centers_
centroids


# In[4]:


model.labels_


# In[5]:


from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    model2 = KMeans(n_clusters=k, random_state=42)
    model2.fit(X)
    score = silhouette_score(X, model2.labels_)
    print("Silhouette Score for k = ", k, "is", score)
    silhouette_scores.append(score)


# In[6]:


import matplotlib.pyplot as plt
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette score')
plt.savefig('silhouette plot.png')


# In[7]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer


# In[27]:


k=3;
model_k=KMeans(n_clusters=k,random_state=42)
visualizer_k=SilhouetteVisualizer(model_k,colors='yellowbrick')
visualizer_k.fit(X)
visualizer_k.show(outpath='Silhouette_diagram_k_'+str(k)+'.png')


# In[35]:


# f, axarr = plt.subplots(3,1)

# for k in range(2, 4):
#     model = KMeans(n_clusters=k, random_state=42)
#     visualizer=SilhouetteVisualizer(model,colors='yellowbrick')
#     visualizer.fit(X)
#     outpath='Silhouette_diagram_for_k_'+str(k)+'.png'
#     img= plt.imread(outpath)
#     axarr[i].imshow(img) 
#     axarr.show()
    
# for i in len(img):
#     axarr[i].imshow(img[i])  
    


# In[36]:


# f, axarr = plt.subplots(1)
# model = KMeans(n_clusters=3, random_state=42)
# visualizer=SilhouetteVisualizer(model,colors='yellowbrick')
# visualizer.fit(X)
# outpath='Silhouette_diagram_for_k_'+str(3)+'.png'
# axarr[0].imshow(outpath)


# In[38]:


from sklearn.metrics import calinski_harabasz_score
calinski_harabasz_scores = []
for k in range(2, 11):
    model3 = KMeans(n_clusters=k, random_state=42)
    model3.fit(X)
    score = calinski_harabasz_score(X, model3.labels_)
    print("Calinski Harabasz Score for k = ", k, "is", score)    
    calinski_harabasz_scores.append(score)


# In[ ]:




