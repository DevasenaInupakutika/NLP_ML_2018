import gensim
import gensim.downloader as api

# Download dataset
dataset = api.load("text8")
data = [d for d in dataset]

# Create the tagged document needed for Doc2Vec model
# Input preparation
def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

train_data = list(create_tagged_document(data))

#print(train_data[:1])

# Training the Doc2Vec model
# Init the Doc2Vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# Build the Volabulary
model.build_vocab(train_data)

# Train the Doc2Vec model
model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

# Getting document vector for a sentence
print(model.infer_vector(['australian', 'captain', 'elected', 'to', 'bowl']))

print(model.infer_vector(['hello', 'how', 'are', 'you']))
