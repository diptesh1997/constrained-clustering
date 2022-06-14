import pandas as pd

def get_sentiment_constraints():
    dataset = pd.read_csv('./sentiment2.csv', index_col=False)
    positive_docs=dataset[dataset['class']==1]['docno']
    negative_docs=dataset[dataset['class']==-1]['docno']
    return([list(positive_docs),list(negative_docs)])
print(get_sentiment_constraints())
