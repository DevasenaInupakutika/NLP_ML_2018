#Building topic models with LSI
#Step 0: Load the necessary packages and import stopwords
import gensim
from gensim.models import LsiModel
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
import re
import logging
from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
stop_words = stopwords.words('english')
stop_words = stop_words + ['com', 'edu', 'subject', 'lines', 'organization', 'would', 'article', 'could', 'also', 'many']

#Step 1: Import the dataset and get the text and real topic of each news article
dataset = api.load("text8")
data = [d for d in dataset]

#Step 2: Prepare the downloaded data and remove stopwords and lemmatize it (using pattern library)
data_processed = []

for i, doc in enumerate(data[:100]):
    doc_out = []
    for wd in doc:
        if wd not in stop_words:  # remove stopwords
            lemmatized_word = lemmatize(wd, allowed_tags=re.compile('(NN|JJ|RB)'))  # lemmatize
            if lemmatized_word:
                doc_out = doc_out + [lemmatized_word[0].split(b'/')[0].decode('utf-8')]
        else:
            continue
    data_processed.append(doc_out)

# Print a small sample    
print(data_processed[0][:5]) 

#data_processed is now a list of list of words. This can now be used to create the Dictionary and the Corpus which will then be used as inputs to the LDA model
# Step 3: Create the Inputs of LDA model: Dictionary and Corpus
dct = corpora.Dictionary(data_processed)
corpus = [dct.doc2bow(line) for line in data_processed]

#Step 4: Now for building the LSI model with say 7 topics here: (number of topics is an arbitrary choice)Train the LSI model
lsi_model = LsiModel(corpus=corpus,
                         id2word=dct,
                         num_topics=7,
                         decay=0.5)

# See the topics
pprint(lsi_model.print_topics(-1))


