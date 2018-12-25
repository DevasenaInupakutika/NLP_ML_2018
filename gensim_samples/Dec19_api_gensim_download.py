import gensim
from gensim import corpora
import gensim.downloader as api

# Get information about the model or dataset
api.info('glove-wiki-gigaword-50')

# Download
w2v_model = api.load("glove-wiki-gigaword-50")
print(w2v_model.most_similar('blue'))
