import os
import json
import shutil
from tqdm import tqdm
from smart_open import open
from news_explorer.preprocess_text.preprocessor import BagOfWordsPreprocessor


class CorpusPreprocessor:
    def __init__(self, paths, preprocessed_folder='news_corpus'):
        self.__paths = paths

        p = os.sep.join(
            os.path.abspath(__file__).split(os.sep)[:-2]
        )
        p = os.path.join(
            p,
            preprocessed_folder
        )
        self.__preprocessed_folder = p

        if os.path.exists(p):
            shutil.rmtree(p)
        os.mkdir(p)

        preprocessed_paths = [
            os.path.join(
                p,
                path.split(os.path.sep)[-1].split('.')[0] + '.txt'
            )
            for path in paths
        ]

        self.__original_to_preprocessed = dict(zip(
            self.__paths,
            preprocessed_paths
        ))

        self.__prep = BagOfWordsPreprocessor()

    def save(self):
        for path in tqdm(self.__paths):
            s = ''
            with open(path, 'r') as f:
                d = json.load(f)
                text = d['text']
                tokens = self.__prep.preprocess(text)
                s = ' '.join(tokens)

            if len(s):
                with open(self.__original_to_preprocessed[path], 'w') as f:
                    f.write(s)

        """
        SAVE NAMED ENTITIES DICTIONARY
        """
        p = os.path.join(
            self.__preprocessed_folder,
            'ents.json'
        )
        with open(p, 'w') as f:
            f.write(json.dumps(self.__prep.ents))

        """
        SAVE DICTIONARY WITH ENTITY-TAG MAP
        """
        p = os.path.join(
            self.__preprocessed_folder,
            'tag2ent.json'
        )
        with open(p, 'w') as f:
            f.write(json.dumps(self.__prep.tag2ent))
