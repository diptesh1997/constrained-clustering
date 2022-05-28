#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import re
import pandas as pd
import os
  
cols = ["docno", "date", "title", "text"]
rows = []
for filename in os.listdir(r'C:\Users\Shimony\Desktop\SHIMO\MS DE\SemII\ATiML\ATiML_Assignments\Project\ATiML_TREC_4_5_Dataset\TREC_4_5\FBIS'):
    
        with open(os.path.join(r'C:\Users\Shimony\Desktop\SHIMO\MS DE\SemII\ATiML\ATiML_Assignments\Project\ATiML_TREC_4_5_Dataset\TREC_4_5\FBIS', filename)) as f:
#         with open(filename, 'r') as f:
            doc_string = f.read()
            f.close()

        doc_string_str = doc_string.replace('TEXT', 'htmltag')
        doc_string = doc_string_str.encode()

        soup = BeautifulSoup(doc_string, "lxml")
        doc_list = soup.select('DOC')

        doc_no = []
        doc_content = []
        for doc in doc_list:
#             doc_no.append(doc.find('docno').get_text())
#             doc_raw = doc.find('htmltag')
    
            docno = doc.find("docno").text
            text = doc.find("htmltag").text
            title = doc.find("ti").text
#             date = doc.find("date1").text
  
            rows.append({"docno": docno,
                         "text": text,
                         "title": title,
                         "date": date
                        })
  
        df = pd.DataFrame(rows, columns=cols)
#         print(df.head())
        df.to_csv('output1.csv')

