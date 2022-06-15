from IPython.display import clear_output, display # NOTE(Abid): Used with Jupyter only, otherwise comment it out
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
#             3. from IPython.display import clear_output, display
#
#             INPUT:  pandas.DataFrame,
#                     int
#             OUTPUT: None

def keyphrasification(df, chunks):
    keyphrase_rows_1 = []
    weight_rows_1 = []
    keyphrase_rows_2 = []
    weight_rows_2 = []
    data = {}
    df_len = len(df)
    
    stride = 0
    total_complete = 0
    while True:
        remain = df.shape[0] - total_complete
        print(f"Remain = {remain}")
        if remain <= 0:
            break
        if (remain) > chunks:
            stride = chunks
        else:
            stride = remain
        for i in range(stride):
            extractor = pke.unsupervised.TopicRank()
            extractor.load_document(input=df.iloc[total_complete + i,2], language='en')
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = extractor.get_n_best(n=2, stemming=False)
            len_keyphrase = len(keyphrases)
            key_1_val = '<None>'
            key_1_wei = 0
            key_2_val = '<None>'
            key_2_wei = 0
            if len_keyphrase >= 1:
                key_1_val = keyphrases[0][0]
                key_1_wei = keyphrases[0][1]
            if len_keyphrase >= 2:
                key_2_val = keyphrases[1][0]
                key_2_wei = keyphrases[1][1]

            keyphrase_rows_1.append(key_1_val)
            weight_rows_1.append(key_1_wei)
            keyphrase_rows_2.append(key_2_val)
            weight_rows_2.append(key_2_wei)
            clear_output(wait=True)
            print(f"{i + 1 + total_complete}/{df_len}\t\t {key_1_val}")
        
        data = {'keyphrase_1' : keyphrase_rows_1,
                'weight_1' : weight_rows_1,
                'keyphrase_2': keyphrase_rows_2,
                'weight_2' : weight_rows_2}
        extracted_df = pd.DataFrame(data)
        
        extracted_df.to_csv(f"keyphrase_{total_complete}_{total_complete + stride}.csv", index=False)
        keyphrase_rows_1 = []
        weight_rows_1 = []
        keyphrase_rows_2 = []
        weight_rows_2 = []
        total_complete = total_complete + stride
