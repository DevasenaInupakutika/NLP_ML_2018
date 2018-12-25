#Create corpus

raw_corpus=["Human machine interface for lab abc computer applications",
"A survey of user opinion of computer system response time", "The EPS user interface management system",
"System and human system engineering testing of EPS",
"Relation of user perceived response time to error measurement",
"The generation of random binary unordered trees",
"The intersection graph of paths in trees",
"Graph minors IV Widths of trees and well quasi ordering",
"Graph minors A survey"]


# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))

#Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist] for document in raw_corpus]

#Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token]+=1

#Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

print(processed_corpus)

#Associate each word to unique integer ID
from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

#Vectors Section
#Using the dictionary to turn tokenized documents into <no of words or entries in dictionaries here 12> dimensional vectors
print(dictionary.token2id)

#Vectorizing new phrase other than the ones present in the corpus
new_doc = "Human Computer Interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)

#Converting whole corpus to list of vectors
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print(bow_corpus)

#Transforming the vector spaces using tf-idf model
from gensim import models
#train the model
tfidf = models.TfidfModel(bow_corpus)
#Tranform the "system minors" string
print(tfidf[dictionary.doc2bow("system minors".lower().split())])
