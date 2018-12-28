import gensim
import gensim.downloader as api
from pprint import pprint

# Download the models
#fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
word2vec_model300 = api.load('word2vec-google-news-300')
glove_model300 = api.load('glove-wiki-gigaword-300')

# Get word embeddings
#word2vec_model300.most_similar('support')

#pprint(fasttext_model300.most_similar('support'))
pprint(word2vec_model300.most_similar('support'))
pprint(glove_model300.most_similar('support'))

#Evaluating the 3 word embedding models and check the accuracy of prediction
#Fasttext Accuracy
#pprint(fasttext_model300.evaluate_word_analogies(analogies="questions-words.txt")[0])

#Word2vec accuracy
pprint(word2vec_model300.evaluate_word_analogies(analogies="questions-words.txt")[0])

#GloVe accuracy
pprint(glove_model300.evaluate_word_analogies(analogies="questions-words.txt")[0]
)
