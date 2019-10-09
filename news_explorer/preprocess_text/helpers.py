import re
import json
import nltk
from nltk.corpus import wordnet
from tqdm import tqdm
from smart_open import open
from langdetect import detect

URL_RE = r'\b(?:(?:https?|ftp)://)?\w[\w-]*(?:\.[\w-]+)+\S*(?<![.,])'


def validate_english_corpus(corpus_files_filename, files):
    en_news = []
    for file in tqdm(files):
        with open(file, 'r') as f:
            d = json.load(f)
            if d['language'] == 'english':
                try:
                    if detect(d['text']) == 'en':
                        en_news.append(file)
                except KeyboardInterrupt:
                    break
                except Exception:
                    pass

    with open(corpus_files_filename, 'w') as f:
        f.write(
            "\n".join(
                en_news
            )
        )


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def url_cleaner(text):
    text = re.sub(URL_RE, '<URL>', text, flags=re.IGNORECASE)
    return text
