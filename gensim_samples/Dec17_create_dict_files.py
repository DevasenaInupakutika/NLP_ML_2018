from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os

# Create gensim dictionary form a single tet file
dictionary = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open('sample.txt', encoding='utf-8'))

# Token to Id map
print(dictionary.token2id)

#Class named ReadTextFiles for reading multiple text files and create a dictionary from them
class ReadTxtFiles(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='latin'):
                yield simple_preprocess(line)

#Actions using ReadTxtFiles
path_to_text_directory = "/home-new/mmn609/gensim_samples/datasets/lsa_sports_food_docs"

dictionary = corpora.Dictionary(ReadTxtFiles(path_to_text_directory))

# Token to Id map
print(dictionary.token2id)
