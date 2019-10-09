import gc
import json
from glob import glob
from pprint import pprint
from smart_open import open
from gensim.corpora import MmCorpus
from gensim.models import TfidfModel
from news_explorer.corpus.bow_corpus import BOWCorpus
from news_explorer.corpus.helpers import filter_low_tfidf
from news_explorer.preprocess_text import helpers
from news_explorer.corpus.preprocess_corpus import CorpusPreprocessor

"""
GET ONLY ENGLISH DOCUMENTS
"""
print(
    "---[" + "GET ONLY ENGLISH DOCUMENTS" + "]---"
)
# files = glob('news/*/*.json')
# helpers.validate_english_corpus('en_news.txt', files)

with open('en_news.txt', 'r') as f:
    en_news = list(
        map(
            lambda x: x.strip(),
            f.readlines()
        )
    )


# pprint(en_news[0])
print("-"*6)

"""
PREPROCESS AND SAVE FILTERED DOCUMENTS
"""
print(
    "---[" + "PREPROCESS AND SAVE FILTERED DOCUMENTS" + "]---"
)
cp = CorpusPreprocessor(en_news)
cp.save()
print("-"*6)


# files = glob('news_corpus/*.json')
# pprint(files)

# with open(files[0], 'r') as f:
#     s = json.load(f)
#     pprint(s)

"""
FILTER AND SAVE CORPUS
"""
print(
    "---[" + "FILTER AND SAVE CORPUS" + "]---"
)
news = glob('news_corpus/*.txt')
corpus = BOWCorpus(news)
tfidf = TfidfModel(corpus)

filter_low_tfidf(corpus, tfidf)

del tfidf
gc.collect()

corpus.dictionary.save('bow_corpus.dict')
MmCorpus.serialize('bow_corpus.mm', corpus)
print("-"*6)

"""

"""

from gensim.corpora import Dictionary
d = Dictionary.load('bow_corpus.dict')
pprint(d.token2id)


# pic.twitter.com/funssqbvdr
