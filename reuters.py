import nltk
from nltk.corpus import reuters, stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def download():
    """
    Download reuters data and stopwords if not already present"
    """
    nltk.download("reuters")
    nltk.download("stopwords")


def _fileids_starting_with(ids, start):
    return list(filter(lambda i: i.startswith(start), ids))


def training_ids(fileids):
    return _fileids_starting_with(fileids, "train")

 
def test_ids(fileids):
    return _fileids_starting_with(fileids, "test")


def get_category_ids(category):
    """
    Returns list of fileids in a category
    """
    return reuters.fileids(category)


def categories(fileids):
    """
    Returns all categories for fileids
    """
    return reuters.categories(fileids)


def get_documents(category):
    """
    Returns training and testing fileids for a category
    """
    docs = get_category_ids(category)
    return training_ids(docs), test_ids(docs)


def create_corpus(fileids):
    """
    Creates a corpus from fileids
    Removes stopwords and punctuation
    Returns a list of strings
    """
    sw = set(stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r"[A-Za-z]+")
    corpus = []
    for doc in fileids:
        words = (w.lower() for w in tokenizer.tokenize(reuters.raw(doc)))
        corpus.append(" ".join(w for w in words if w not in sw))
    return corpus


def normalize(counts):
    transformer = TfidfTransformer(smooth_idf=1)
    return transformer.fit_transform(counts).toarray()


def get_ngram(corpus, n):
    ngram = CountVectorizer(analyzer="char", ngram_range=(n, n), min_df=1)
    counts = ngram.fit_transform(corpus)
    return ngram, normalize(counts.toarray())


def get_bow(corpus):
    v = CountVectorizer(analyzer="word", min_df=1)
    bow = v.fit_transform(corpus)
    return v, normalize(bow.toarray())

