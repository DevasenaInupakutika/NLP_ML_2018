import gensim
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api

sent_1 = 'Sachin is a cricket player and a opening batsman'.split()
sent_2 = 'Dhoni is a cricket player too He is a batsman and keeper'.split()
sent_3 = 'Anand is a chess player'.split()

word2vec_model300 = api.load('word2vec-google-news-300')
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
glove_model300 = api.load('glove-wiki-gigaword-300')

# Prepare a dictionary and a corpus.
documents = [sent_1, sent_2, sent_3]
dictionary = corpora.Dictionary(documents)

# Prepare the similarity matrix
similarity_matrix = word2vec_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

# Convert the sentences into bag-of-words vectors.
sent_1 = dictionary.doc2bow(sent_1)
sent_2 = dictionary.doc2bow(sent_2)
sent_3 = dictionary.doc2bow(sent_3)

# Compute soft cosine similarity
print(softcossim(sent_1, sent_2, similarity_matrix))

print(softcossim(sent_1, sent_3, similarity_matrix))

print(softcossim(sent_2, sent_3, similarity_matrix))

# Which word from the given list doesn't go with the others?
print(fasttext_model300.doesnt_match(['india', 'australia', 'pakistan', 'china', 'beetroot']))  

# Compute cosine distance between two words.
print(fasttext_model300.distance('king', 'queen'))


# Compute cosine distances from given word or vector to all words in `other_words`.
print(fasttext_model300.distances('king', ['queen', 'man', 'woman']))


# Compute cosine similarities
print(fasttext_model300.cosine_similarities(fasttext_model300['king'], 
                                            vectors_all=(fasttext_model300['queen'], 
                                                        fasttext_model300['man'], 
                                                        fasttext_model300['woman'],
                                                        fasttext_model300['queen'] + fasttext_model300['man'])))  
# Note: Queen + Man is very similar to King.

# Get the words closer to w1 than w2
print(glove_model300.words_closer_than(w1='king', w2='kingdom'))


# Find the top-N most similar words.
print(fasttext_model300.most_similar(positive='king', negative=None, topn=5, restrict_vocab=None, indexer=None))


# Find the top-N most similar words, using the multiplicative combination objective,
print(glove_model300.most_similar_cosmul(positive='king', negative=None, topn=5))

