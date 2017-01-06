import nltk
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import reuters, stopwords


def download():
    nltk.download("reuters")
    nltk.download("stopwords")


def _fileids_starting_with(ids, start):
    return list(filter(lambda i: i.startswith(start), ids))


def training_ids(fileids):
    return _fileids_starting_with(fileids, "train")

 
def test_ids(fileids):
    return _fileids_starting_with(fileids, "test")


def get_category_ids(category):
    return reuters.fileids(category)


def categories(fileids):
    return reuters.categories(fileids)


def get_documents(category):
    docs = get_category_ids(category)
    return training_ids(docs), test_ids(docs)


def create_corpus(fileids):
    """
    Remove stopwords and punctuation
    Returns a list of strings
    """
    sw = set(stopwords.words("english"))
    corpus = []
    for doc in fileids:
        words = (w.lower() for w in reuters.words(doc))
        corpus.append(" ".join(w for w in words if w not in sw))
    return corpus


def get_ngram(corpus, n):
    ngram = CountVectorizer(analyzer="char", ngram_range=(n, n), min_df=1)
    counts = ngram.fit_transform(corpus)
    return ngram, counts.toarray().astype(int)


def get_bow(corpus):
    v = CountVectorizer(analyzer="word", min_df=1)
    bow = v.fit_transform(corpus)
    return v, bow.toarray()

