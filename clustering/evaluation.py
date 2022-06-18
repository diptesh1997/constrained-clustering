from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

import matplotlib.pyplot as plt

silhouette_scores = []

for k in range(2, 11):
    score = silhouette_score(X, model2.labels)
    print("Silhouette Score for k = ", k, "is", score)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette score')
plt.savefig('silhouette_plot_1.png')

calinski_harabasz_scores = []

for k in range(2, 11):
    model3 = KMeans(n_clusters=k, random_state=42)
    model3.fit(X)
    score = calinski_harabasz_score(X, model3.labels_)
    print("Calinski Harabasz Score for k = ", k, "is", score)    
    calinski_harabasz_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Silhouette score')
plt.savefig('silhouette_plot_2.png')