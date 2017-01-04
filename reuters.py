import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Make sure we have all the data
nltk.download("reuters")
nltk.download("stopwords")

from nltk.corpus import reuters, stopwords


def _get_ids_starting_with(start):
    return list(filter(lambda id: id.startswith(start), reuters.fileids()))


def get_training_ids():
    return _get_ids_starting_with("train")

 
def get_test_ids():
    return _get_ids_starting_with("test")


def clean_documents(documents):
    """
    Remove stopwords and punctuation
    Returns a list of strings
    """
    sw = set(stopwords.words("english"))
    corpus = []
    for d in documents:
        words = (w.lower() for w in reuters.words(d))
        corpus.append(" ".join(w for w in words if w not in sw))
    return corpus


def get_ngram(docs, n):
    corpus = clean_documents(docs)
    ngram = CountVectorizer(analyzer="char", ngram_range=(n, n), min_df=1)
    counts = ngram.fit_transform(corpus)
    return ngram.get_feature_names(), counts.toarray().astype(int)


def get_bow(docs):
    corpus = clean_documents(docs)
    v = CountVectorizer(analyzer="word", min_df=1)
    bow = v.fit_transform(corpus)
    return v.get_feature_names(), bow.toarray()

