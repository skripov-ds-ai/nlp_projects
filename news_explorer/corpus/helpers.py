from tqdm import tqdm
from gensim.models import TfidfModel
from news_explorer.corpus.bow_corpus import BOWCorpus


def filter_low_tfidf(data: BOWCorpus, tfidf: TfidfModel, threshold=0.1):
    """
    This function modify corpus and make its __allow_update value equal to False
    :param data: BOWCorpus to preprocess low tfidf tokens
    :param tfidf: tfidf model of corpus :param data
    :param threshold: tfidf value for removing less valued tokens from corpus dictionary
    :return: None
    """
    low_tfidf_tokens = set()
    for doc in tfidf[data]:
        tokens = [
            id
            for id, freq in doc
            if freq < threshold
        ]
        low_tfidf_tokens.update(tokens)

    data.dictionary.filter_tokens(
        low_tfidf_tokens
    )
    data.allow_update = False
