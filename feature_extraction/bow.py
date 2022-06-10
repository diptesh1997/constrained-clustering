from sklearn.feature_extraction.text import TfidfVectorizer

def bow_features(df):
    vectorizer = TfidfVectorizer(max_features=100, strip_accents='unicode', token_pattern = '[a-z]+\w*', analyzer='word',  lowercase=True, use_idf=True)
    model = vectorizer.fit_transform(df['text'])
    vectorizer.get_feature_names()