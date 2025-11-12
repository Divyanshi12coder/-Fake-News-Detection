import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def preprocess_dataframe(df, column):
    df[column] = df[column].apply(clean_text)
    return df

def get_vectorizer():
    return TfidfVectorizer(max_features=5000)

def vectorize(corpus, vectorizer):
    return vectorizer.fit_transform(corpus)