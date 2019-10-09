import json
from tqdm import tqdm
from smart_open import open
from gensim import corpora


class BOWCorpus:
    def __init__(self, paths, dictionary=None):
        self.__paths = paths

        if dictionary is None:
            self.dictionary = corpora.Dictionary()
            self.__allow_update = True
        else:
            if type(dictionary) is not corpora.Dictionary:
                raise ValueError(
                    "Parameter d should be bool"
                )
            self.dictionary = dictionary
            self.__allow_update = False

    def dictionary(self):
        return self.dictionary

    @property
    def allow_update(self):
        return self.__allow_update

    @allow_update.setter
    def allow_update(self, update):
        if type(update) is not bool:
            raise ValueError(
                "Parameter update should be bool"
            )
        self.__allow_update = update

    def __iter__(self):
        for path in tqdm(self.__paths):
            with open(path, 'r') as f:
                text = f.read()
                tokens = text.split()

                bow = self.dictionary.doc2bow(
                    tokens,
                    allow_update=self.__allow_update
                )

                yield bow





