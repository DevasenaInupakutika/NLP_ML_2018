import gensim
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api
from pprint import pprint

# Download dataset
dataset = api.load("text8")
data = [d for d in dataset]

# Split the data into 2 parts. Part 2 will be used later to update the model
data_part1 = data[:1000]
data_part2 = data[1000:]

# Train Word2Vec model. Defaults result vector size = 100
model = Word2Vec(data_part1, min_count = 0, workers=cpu_count())

# Get the word vector for given word
pprint(model['topic'])

pprint(model.most_similar('topic'))

# Save and Load Model
model.save('newmodel')
model = Word2Vec.load('newmodel')

#Update existing Word2Vec Model with the new data
model.build_vocab(data_part2, update=True)
model.train(data_part2, total_examples=model.corpus_count, epochs=model.iter)

pprint(model['topic'])
