import os
import spacy
from cleantext import clean
from string import whitespace, punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, MWETokenizer
from news_explorer.preprocess_text import collocations
from news_explorer.preprocess_text.helpers import get_wordnet_pos, url_cleaner


special_collocations = []
special_collocations.extend(
    collocations.but_negation_clauses
)
special_collocations.extend(
    collocations.but_expressions
)

for i in range(len(special_collocations)):
    special_collocations[i] = special_collocations[i].strip().split()

special_collocations = list(
    filter(
        lambda x: len(x) > 1,
        special_collocations
    )
)

sentiment_lexicon = []
path = os.path.join(
    os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1]),
    'senticnet5.txt'
)
with open(path, 'r') as f:
    tmp = f.readlines()[1:]
    for t in tmp:
        w = t.split()[0].split('_')
        if len(w) > 1:
            sentiment_lexicon.append(w)

special_collocations.extend(
    sentiment_lexicon
)
special_collocations.append(
    ['a', 'lot', 'of']
)

stop_technical_tokens = {
    'rw',
}


class BagOfWordsPreprocessor:
    def __init__(self, ents=None, tag2ent=None, collocations=special_collocations, appos=collocations.appos):
        self.__tokenizer = TweetTokenizer(
            reduce_len=True
        )
        self.__collocations = collocations
        self.__tknzr = MWETokenizer(
            self.__collocations
        )

        self.__lemm = WordNetLemmatizer()
        self.__nlp = spacy.load("en_core_web_sm")
        if ents is None:
            self.__ents = {}
            if tag2ent is not None:
                raise ValueError(
                    "ent2tag and ents should be None or not None both"
                )
            self.__tag2ent = {}
        else:
            if tag2ent is None:
                raise ValueError(
                    "ent2tag and ents should be None or not None both"
                )
            self.__ents = ents
            self.__tag2ent = tag2ent
        self.__appos = appos
        for a in appos:
            self.__appos[a] = '_'.join(self.__appos[a].split())

        self.__punctuation = punctuation + "“”‘’‚"
        self.__stop_symbols = '←↓→↑'

    @property
    def ents(self):
        return self.__ents

    @property
    def tag2ent(self):
        return self.__tag2ent

    @property
    def appos(self):
        return self.__appos

    @property
    def collocations(self):
        return self.__collocations

    @property
    def stop_symbols(self):
        return self.__stop_symbols

    @property
    def punctuation(self):
        return self.__punctuation

    def __lemmatize_pos(self, word, pos):
        return self.__lemm.lemmatize(word, pos)

    def __clean_whitespaces(self, text):
        for w in whitespace:
            text = text.replace(w, ' ')
        return text

    def __clean_stopsymbols(self, text):
        for w in self.__stop_symbols:
            text = text.replace(w, '')
        return text

    def __to_utf8(self, text):
        return text.decode("utf-8-sig")

    def __tokenize(self, text):
        s = " ".join(self.__tokenizer.tokenize(text))
        tokens = self.__tknzr.tokenize(s.split())
        return tokens

    def __ent_to_tag(self, name):
        if name[0] == '<':
            return name
        t = "<" + self.__ents[name][0] + "_" + self.__ents[name][1] + ">"
        return t

    def preprocess(self, text):
        text = self.__clean_whitespaces(text)
        text = self.__clean_stopsymbols(text)

        text = clean(
            text,
            fix_unicode=True,
            no_urls=True,
            no_emails=True,
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            lang='en'
        )

        text = url_cleaner(text)

        tokens = self.__tokenize(text)
        tokens = [
            self.__appos[word] if word in self.__appos else word
            for word in tokens
        ]

        processed_text = ' '.join(tokens)
        tokenized_text = []
        spacy_ents = self.__nlp(processed_text).ents
        ent_idx = []
        for i in range(len(spacy_ents)):
            e = spacy_ents[i]
            if e.text.lower() not in self.__ents:
                self.__ents[e.text.lower()] = (e.label_, str(len(self.__ents)))
                self.__tag2ent[self.__ent_to_tag(e.text.lower())] = e.text.lower()
            if i == 0:
                tokenized_text.append(
                    processed_text[:e.start_char]
                )
            else:
                ee = spacy_ents[i - 1]
                tokenized_text.append(
                    processed_text[ee.end_char:e.start_char]
                )
            tokenized_text.append(
                processed_text[e.start_char:e.end_char]
            )
            ent_idx.append(len(tokenized_text) - 1)
            if i == len(spacy_ents) - 1:
                tokenized_text.append(
                    processed_text[e.end_char:]
                )

        for i in ent_idx:
            tokenized_text[i] = self.__ent_to_tag(tokenized_text[i].lower())

        tokens = " ".join(tokenized_text).split()

        tokens = [
            token if token[0] == '<' else token.lower()
            for token in tokens
        ]

        tokens = [
            word
            for word in tokens
            if
            word not in self.__punctuation
            and
            word not in stop_technical_tokens
        ]

        tags = list(map(get_wordnet_pos, tokens))

        tokens = list(
            map(
                lambda x: self.__lemmatize_pos(x[0], x[1]),
                zip(tokens, tags)
            )
        )
        for i in range(len(tokens)):
            if tokens[i].lower() in self.__ents:
                tokens[i] = self.__ent_to_tag(tokens[i].lower())

        return tokens

