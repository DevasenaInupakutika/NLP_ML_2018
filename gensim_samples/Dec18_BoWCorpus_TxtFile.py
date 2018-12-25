import gensim
from gensim.utils import simple_preprocess
from smart_open import smart_open
from gensim import corpora
import nltk
nltk.download('stopwords')  # run once
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


class BoWCorpus(object):
    def __init__(self, path, dictionary):
        self.filepath = path
        self.dictionary = dictionary

    def __iter__(self):
        global mydict  # OPTIONAL, only if updating the source dictionary.
        for line in smart_open(self.filepath, encoding='latin'):
            # tokenize
            tokenized_list = simple_preprocess(line, deacc=True)

            # create bag of words
            bow = self.dictionary.doc2bow(tokenized_list, allow_update=True)

            # update the source dictionary (OPTIONAL)
            mydict.merge_with(self.dictionary)

            # lazy return the BoW
            yield bow


# Create the Dictionary
mydict = corpora.Dictionary()

# Create the Corpus
bow_corpus = BoWCorpus('sample.txt', dictionary=mydict)  # memory friendly

# Print the token_id and count for each line.
for line in bow_corpus:
    print(line)

#Saving the gensim dictionary and corpus to disk and load them later
# Save the Dict and Corpus
mydict.save('mydict.dict')  # save dict to disk
corpora.MmCorpus.serialize('bow_corpus.mm', bow_corpus)  # save corpus to disk 

# Load them back
loaded_dict = corpora.Dictionary.load('mydict.dict')

corpus = corpora.MmCorpus('bow_corpus.mm')
for line in corpus:
    print(line)
