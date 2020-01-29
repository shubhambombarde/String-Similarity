import gensim
import smart_open
from nltk import sent_tokenize, word_tokenize
import numpy as np
from scipy import spatial

def read_corpus(fname, tokens_only=False):
  with smart_open.open(fname, encoding="iso-8859-1") as f:
    for i, line in enumerate(f):
      tokens = gensim.utils.simple_preprocess(line)
      if tokens_only:
        yield tokens
      else:
        # For training data, add tags
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def createModel():
  gen_docs = list(read_corpus('datasets/hotel_reviews.txt'))

  # build vocabulary and train model
  model = gensim.models.Doc2Vec(vector_size=50, min_count=2, epochs=40)
  model.build_vocab(gen_docs)
  model.save('models/gensim.models.Doc2Vec.bin')
  print('Model created and saved successfully')


def avg_feature_vector(sentence, model, num_features, index2word_set):
  words = word_tokenize(sentence)
  feature_vec = np.zeros((num_features,), dtype='float32')
  n_words = 0
  for word in words:
    if word in index2word_set:
      n_words += 1
      feature_vec = np.add(feature_vec, model[word])
  if (n_words > 0):
    feature_vec = np.divide(feature_vec, n_words)
  return feature_vec

def calculateSimilarity(X, Y):
  model = gensim.models.Doc2Vec.load('models/gensim.models.Doc2Vec.bin')
  score = model.wv.similarity(w1=X, w2=Y)

  # index2word_set = set(model.wv.index2word)
  # s1_afv = avg_feature_vector('Sun is bright and yellow.', model=model, num_features=150, index2word_set=index2word_set)
  # s2_afv = avg_feature_vector('yellow and bright is the sun.', model=model, num_features=150,
  #   index2word_set=index2word_set)
  # sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
  # print(sim)

  return score


createModel()
# score = calculateSimilarity('daylight', 'bright')
# print(score)