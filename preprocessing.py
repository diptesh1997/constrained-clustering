from email.mime import base
from fileinput import filename
from bs4 import BeautifulSoup
import re
import pandas as pd
import os

cols = ["file", "docno", "date", "title", "text"]
rows = []
# data_source = ['FBIS','FR94','FT','LATIMES']
data_source = ['FT']
data_path = "./ATiML_TREC_4_5_Dataset/TREC_4_5/"
file_path = []


def process_file(file_path):
    try:
        with open(file_path, encoding="ISO-8859-1") as f:
            doc_string = f.read()
            f.close()
        soup = BeautifulSoup(doc_string, "lxml")
        doc_list = soup.select('DOC')
        print(len(doc_list))
        for doc in doc_list:
            docno = doc.find("docno").text
            text = doc.find("text").text
            title = doc.find("headline").text
            date = doc.find("date").text
            rows.append({"docno": docno, "date": date,
                        "title": title, "text": text})
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
