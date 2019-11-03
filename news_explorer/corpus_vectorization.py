import gc
import json
from glob import glob
from pprint import pprint
from smart_open import open
from gensim.corpora import MmCorpus
from gensim.models import TfidfModel

import numpy as np
import matplotlib.pyplot as plt


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
#
# with open('en_news.txt', 'r') as f:
#     en_news = list(
#         map(
#             lambda x: x.strip(),
#             f.readlines()
#         )
#     )


# pprint(en_news[0])
print("-"*6)

"""
PREPROCESS AND SAVE FILTERED DOCUMENTS
"""
print(
    "---[" + "PREPROCESS AND SAVE FILTERED DOCUMENTS" + "]---"
)
# cp = CorpusPreprocessor(en_news)
# cp.save()
print("-"*6)


# files = glob('news_corpus/*.json')
# pprint(files)
#
# with open(files[0], 'r') as f:
#     s = json.load(f)
#     pprint(s)

"""
FILTER AND SAVE CORPUS
"""
print(
    "---[" + "FILTER AND SAVE CORPUS" + "]---"
)
# news = glob('news_corpus/*.txt')
# corpus = BOWCorpus(news)
# tfidf = TfidfModel(corpus)
#
# filter_low_tfidf(corpus, tfidf)
#
# del tfidf
# gc.collect()
#
# corpus.dictionary.save('bow_corpus.dict')
# MmCorpus.serialize('bow_corpus.mm', corpus)
print("-"*6)

"""
GENSIM LDA
"""
print(
    "---[" + "GENSIM LDA" + "]---"
)
# from gensim.corpora import Dictionary
# d = Dictionary.load('bow_corpus.dict')
# pprint(d.token2id)

import gensim
from gensim.models.coherencemodel import CoherenceModel

id2word = gensim.corpora.Dictionary.load(
    'bow_corpus.dict'
)
mm = MmCorpus(
    'bow_corpus.mm'
)

from tqdm import trange

def compute_coherence_values(dictionary, corpus, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    best_model = None
    for num_topics in trange(start, limit, step):
        model = gensim.models.ldamulticore.LdaMulticore(
            corpus=corpus,
            num_topics=num_topics,
            id2word=id2word,
            iterations=100,
            workers=6,
            random_state=3,
        )
        coherencemodel = CoherenceModel(
            model=model,
            corpus=corpus,
            dictionary=dictionary,
            coherence='u_mass',
            # coherence='c_v'
        )
        value = coherencemodel.get_coherence()
        if len(coherence_values) and value > coherence_values[-1]:
            best_model = model
            gc.collect()
        coherence_values.append(value)

    return best_model, coherence_values



# lda = gensim.models.ldamulticore.LdaMulticore(
#     corpus=mm,
#     id2word=id2word,
#     chunksize=1000,
#     num_topics=120,
#     passes=15,
#     eval_every=1,
#     random_state=14,
#     workers=6
# )
#
# pprint(
#     lda.print_topics(
#         num_topics=20,
#         num_words=15,
#     )
# )


limit = 170
start = 150
step = 1

best_model, coherence_values = compute_coherence_values(
    dictionary=id2word,
    corpus=mm,
    start=start,
    limit=limit,
    step=step
)

x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

lda = best_model#model_list[np.argmax(coherence_values)]

from gensim.test.utils import datapath
temp_file = datapath("model")
lda.save(temp_file)

# x = np.linspace(-1, 1)
# plt.plot(x, x**2)
# plt.show()

# pic.twitter.com/funssqbvdr
