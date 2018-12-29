from gensim.summarization import summarize, keywords
from pprint import pprint
from smart_open import smart_open

text = " ".join((line for line in smart_open('sample.txt', encoding='utf-8')))

# Summarize the paragraph
pprint(summarize(text, word_count=20))

# Important keywords from the paragraph
print(keywords(text))
