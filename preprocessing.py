import os
import string
import pandas as pd
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


cols = ["docno", "text"]
rows = []
data_source = ['FBIS','FR94','FT','LATIMES']
# data_source = ['FR94']

data_path = "./ATiML_TREC_4_5_Dataset/TREC_4_5/"
PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocessing_pipeline(text):
    processed_text_1 = remove_urls(text)
    processed_text_2 = remove_punctuation(processed_text_1)
    processed_text_3 = remove_stopwords(processed_text_2)
    processed_text_4 = stem_words(processed_text_3)
    return processed_text_4


def process_file(file_path):
    try:
        with open(file_path, encoding="ISO-8859-1") as f:
            doc_string = f.read()
            f.close()
        soup = BeautifulSoup(doc_string, "lxml")
        doc_list = soup.select('DOC')
        print(len(doc_list))
        for doc in doc_list:
            text = preprocessing_pipeline(doc.find("text").text)
            docno = doc.find("docno").text
            rows.append({"docno": docno, "text": text})
    except Exception as e:
            pass

def get_nested_path(base_path):
    for file_dir in os.listdir(base_path):
        file_dir_path = os.path.join(base_path, file_dir)
        if os.path.isfile(file_dir_path):
            process_file(file_dir_path)
        elif os.path.isdir(file_dir_path):
            get_nested_path(file_dir_path)
        else:
            pass


for news_src in data_source:
    path = os.path.join(data_path, news_src)
    print(path)
    get_nested_path(path)

df = pd.DataFrame(rows, columns=cols)
df.to_csv('news_articles.csv', index=False)
