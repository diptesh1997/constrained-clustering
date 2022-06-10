import os, sys, getopt
from statistics import mode
import string
import pandas as pd
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sqlalchemy import false, true

argv = sys.argv[1:]
try:
    opts, argv = getopt.getopt(argv, '', ["mode="])
    for opt,val in opts:
        if opt in ("--mode"):
            mode = val
except getopt.error as err:
	print (str(err))

PUNCT_TO_REMOVE = string.punctuation
out_file_name = 'data.csv'
keyphrase_mode = false
cols = ["docno", "doclen", "text"]
rows = []
data_source = ['LATIMES', 'FT']
data_path = "./ATiML_TREC_4_5_Dataset/TREC_4_5/"

if mode == 'kp':
    keyphrase_mode = true
    PUNCT_TO_REMOVE = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\n'
    out_file_name = 'keyphrase_data.csv'

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text):
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def stem_words(text):
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(word) for word in text.split()])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocessing_pipeline(text):
    processed_text_1 = remove_urls(text)
    processed_text_2 = remove_punctuation(processed_text_1)
    processed_text_3 = remove_stopwords(processed_text_2)
    if keyphrase_mode == true:
        return processed_text_3
    else:
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
                if len(doc.findAll("text"))== 0:
                    continue
                text = preprocessing_pipeline(doc.find("text").text)
                docno = doc.find("docno").text
                doclen = len(text.split())
                rows.append({"docno": docno, "text": text, "doclen": doclen})
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
df.to_csv(out_file_name, index=False)
