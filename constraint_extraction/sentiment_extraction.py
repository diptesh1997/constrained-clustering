
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

dataset=pd.read_csv("../../Data/data-2.csv")

sid = SentimentIntensityAnalyzer()
def f(row):
    ss = sid.polarity_scores(row["text"])
    if ss['compound']>=0.95:
        val = 1
    elif ss['compound']<=-0.9:
        val = -1
    else:
        val = 0
    return val
dataset['class'] = dataset.apply(f, axis=1)
dataset=dataset[['docno','class']]
dataset.to_csv('./sentiment_data.csv',index=True)
print('sentiment.csv created')