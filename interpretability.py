import numpy as np

def get_top_keywords(model, vectorizer, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_pos = np.argsort(coefs)[-n:]
    top_neg = np.argsort(coefs)[:n]
    return {
        "REAL": [(feature_names[i], coefs[i]) for i in top_pos],
        "FAKE": [(feature_names[i], coefs[i]) for i in top_neg]
    }