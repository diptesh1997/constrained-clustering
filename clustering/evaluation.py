#This major ML Ops line, assumes features have benn extracted
#constraints are applied on run-time while clustering
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
import sys, getopt
import traceback

import pandas as pd
import numpy as np

from clustering import k_means
from neighbours_provider import get_neighbours
n = len(sys.argv)
try:
    ppline=int(sys.argv[1]) #If ppline=1 => Tfidif+sentiment ,If ppline=2 => Word2vec+LDA+sentiment

    neighbours=int(sys.argv[2]) #number of neighbours
    query_doc=sys.argv[3]  #query document id
    # n_clusters=int(sys.argv[4]) #number of clusters
    must_link = float(sys.argv[4])
    cannot_link = float(sys.argv[5])
    if(ppline==1):
            print("Selected pipeline => "+"Tf_IDF+sentiment")
            doc_path="C:/Users/Shimony/Desktop/SHIMO/MS DE/SemII/ATiML/ATiML_Assignments/Project/tf_idf_features+sentiment.csv"
    elif (ppline == 2):
            print("Selected pipeline => " + "Word2vec+LDA+sentiment")
            doc_path="C:/Users/Shimony/Desktop/SHIMO/MS DE/SemII/ATiML/ATiML_Assignments/Project/word2vec_LDA_sentiment.csv"

    else:
        raise Exception("Invalid choice for pipeline")
    print("number of neighbours "+str(neighbours))
    print("Query document " + query_doc)
    # print("Number of Clusters " +str(n_clusters ))
    print("Must link penalty " + str(must_link))
    print("Cannot link penalty  " + str(cannot_link))
    dataframe=pd.read_csv(doc_path,index_col='docno')
    query_point=dataframe[dataframe.index == query_doc].drop("class",axis=1)

    # Get the neighbours
    data=get_neighbours(dataframe.drop("class",axis=1),query_point,neighbours)
    # Merge sentimnets value
    final_data = pd.merge(data, dataframe["class"], on="docno")
    # Drop distance column
    final_data=final_data.drop('dist',axis=1)
    pos_doc_df = final_data[final_data["class"] == 1]

    neg_doc_df = final_data[final_data["class"] == -1]
    neu_doc_df = final_data[final_data["class"] == 0]
    keyphrase_df = pd.read_csv("C:/Users/Shimony/Desktop/SHIMO/MS DE/SemII/ATiML/ATiML_Assignments/Project/keyphrase_docno_added.csv", index_col='docno')

    final_keyphrase = keyphrase_df[keyphrase_df.index.isin(final_data.index)]
    final_keyphrase = final_keyphrase.reindex(final_data.index)

    print(final_keyphrase.index, final_data.index)
    keyphrase_penalty = np.array([10, 4])
    final_data = pd.merge(data, dataframe["class"], on="docno")

    silhouette_scores = []
    for k in range(3, 11):
        result = pd.DataFrame(
            k_means(k, final_data, final_keyphrase, [], pos_doc_df, neg_doc_df, neu_doc_df, must_link, cannot_link,
                    keyphrase_penalty))
        print(result.head())
        if (ppline == 1):
            DF = result.iloc[:, 1:101]
            df_labels = result.iloc[:, 102]
        elif (ppline == 2):
            DF = result.iloc[:, 1:101]          #please update as per the result set
            df_labels = result.iloc[:, 103]     #please update as per the result set

        labels=df_labels.to_numpy()
        X = DF.values
        score = silhouette_score(X, labels)
        print("Silhouette Score for k = ", k, "is", score)
        silhouette_scores.append(score)

    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette score')
    plt.savefig('silhouette_plot.png')

except Exception as e:
    traceback.print_exc(e)