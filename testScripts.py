import gensim
import numpy as np
from nltk import sent_tokenize, word_tokenize
import nltk
# nltk.download('all')


def usingGensimv1(X, Y):
  X_docs = []
  Y_docs = []

  # tokenize sentences
  tokens = sent_tokenize(X)
  for line in tokens:
    X_docs.append(line)

  # creating index document with sentence X
  # Tokenize words and create dictionary
  gen_docs = [[w.lower() for w in word_tokenize(text)]
              for text in X_docs]

  # mapping words to unique id's
  dictionary = gensim.corpora.Dictionary(gen_docs)

  # Create a bag of words
  corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

  # Term Frequency – Inverse Document Frequency(TF-IDF)
  tf_idf = gensim.models.TfidfModel(corpus)
  for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

  # Creating similarity measure object
  # building the index
  sims = gensim.similarities.Similarity('/home/shubhambombarde/workspace/StringSimilarity', tf_idf[corpus],
    num_features=len(dictionary))

  # creating query document with sentence Y
  tokens = sent_tokenize(Y)
  for line in tokens:
    Y_docs.append(line)

  for line in Y_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = [dictionary.doc2bow(query_doc)]   #update an existing dictionary and create bag of words

  query_doc_tf_idf = tf_idf[query_doc_bow]
  print('Comparing Result:', sims[query_doc_tf_idf])

def usingGensimv2(X, Y):
  # creating index document with sentence X
  X_tokens = [[w.lower() for w in word_tokenize(X)]]

  # mapping words to unique id's
  dictionary = gensim.corpora.Dictionary(X_tokens)

  # Create a bag of words
  corpus = [dictionary.doc2bow(X_token) for X_token in X_tokens]

  # Term Frequency – Inverse Document Frequency(TF-IDF)
  tf_idf = gensim.models.TfidfModel(corpus)
  for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

  # Creating similarity measure object
  # building the index
  sims = gensim.similarities.Similarity('/home/shubhambombarde/workspace/StringSimilarity', tf_idf[corpus],
    num_features=len(dictionary))

  # update an existing dictionary and create bag of words
  query_doc = [w.lower() for w in word_tokenize(Y)]
  query_doc_bow = dictionary.doc2bow(query_doc)

  query_doc_tf_idf = tf_idf[query_doc_bow]
  print('Comparing Result:', sims[query_doc_tf_idf])

  return

def usingGensimv3(X, Y):
  X_docs = []
  Y_docs = []

  # tokenize sentences
  tokens = [X]
  for line in tokens:
    X_docs.append(line)

  # creating index document with sentence X
  # Tokenize words and create dictionary
  gen_docs = [[w.lower() for w in word_tokenize(text)]
              for text in X_docs]

  # mapping words to unique id's
  dictionary = gensim.corpora.Dictionary(gen_docs)

  # Create a bag of words
  corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

  # Term Frequency – Inverse Document Frequency(TF-IDF)
  tf_idf = gensim.models.TfidfModel(corpus)
  for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

  # Creating similarity measure object
  # building the index
  sims = gensim.similarities.Similarity('/home/shubhambombarde/workspace/StringSimilarity', tf_idf[corpus],
    num_features=len(dictionary))

  # creating query document with sentence Y
  tokens = [Y]
  for line in tokens:
    Y_docs.append(line)

  for line in Y_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)   #update an existing dictionary and create bag of words

  query_doc_tf_idf = tf_idf[query_doc_bow]
  print('Comparing Result:', sims[query_doc_tf_idf])

usingGensimv1('Mars is the fourth planet in our solar system It is second-smallest planet in the Solar System after Mercury Saturn is yellow planet.', 'Mars is the fourth planet in our solar system It is second-smallest planet in the Solar System after Mercury Saturn is yellow planet.')
# usingGensimv3("Saturn is yellow planet. It is beautiful.", "Saturn is yellow planet. It is beautiful.")