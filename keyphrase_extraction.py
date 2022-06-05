import pke
import pandas as pd

# NOTE(Abid): Please make sure the preprocessed data given to this routine have not gone through the following processes:
#             1. Stopword removal
#             2. Stemming
#             For removing punctuations please make sure to use '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~\n' as the value of `PUNCT_TO_REMOVE` variable
#
#             DEPENDENCIES:
#             1. import pke
#             2. import pandas as pd
#
#             INPUT:  pandas.DataFrame
#             OUTPUT: pandas.DataFrame

def keyphrasification(df):
    keyphrase_rows = []
    for i in range(df.shape[0]):
        extractor.load_document(input=df.iloc[i,1], language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()

        # TODO(Abid): Only 5 best candidates are selected
        keyphrases = extractor.get_n_best(n=5, stemming=True)
        keyphrase_rows.append(keyphrases[0][0])
    return pd.DataFrame(keyphrase_rows, columns=['keyphrase'])
