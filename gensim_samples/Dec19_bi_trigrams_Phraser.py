import gensim
from gensim import corpora
import gensim.downloader as api

# Get information about the model or dataset
api.info('text8')

#text8 is a dataset consisting of first 100,000,000 bytes of plain text from Wikipedia
#Download the pre-trained text8 model
dataset = api.load("text8")
dataset = [wd for wd in dataset]

dct = corpora.Dictionary(dataset)
corpus = [dct.doc2bow(line) for line in dataset]

# Build the bigram models
bigram = gensim.models.phrases.Phrases(dataset, min_count=3, threshold=10)

# Construct bigram
print(bigram[dataset[0]])

#After generating bigrams, we can pass the output to train a new Phrases Model by applying the bigrammed corpus on the trained bigram model
# Build the trigram models
trigram = gensim.models.phrases.Phrases(bigram[dataset], threshold=10)

# Construct trigram
print(trigram[bigram[dataset[0]]])
