import gensim
from nltk import sent_tokenize, word_tokenize
import numpy as np
from scipy import spatial

def createModel():
  readFileObj = open('datasets/hotel_reviews.txt', mode='r', encoding='utf-8', errors='ignore')
  dataset = readFileObj.read()

  # data preprocessing
  sentences = sent_tokenize(dataset)
  documents = []
  for sentence in sentences:
    documents.append(sentence)
  gen_docs = [[w.lower() for w in word_tokenize(text)] for text in documents]


  # build vocabulary and train model
  model = gensim.models.Word2Vec(gen_docs, size=150, window=10, min_count=2, workers=10, iter=10)
  model.save('gensim.models.Word2Vec.bin')
  print('Model created and saved successfully')

def getWordDictionary():
    readFileObj = open('datasets/hotel_reviews.txt', mode='r', encoding='utf-8', errors='ignore')
    dataset = readFileObj.read()

    # data preprocessing
    sentences = sent_tokenize(dataset)
    documents = []
    for sentence in sentences:
        documents.append(sentence)
    gen_docs = [[w.lower() for w in word_tokenize(text)] for text in documents]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    return dictionary

def calculateSimilarity(X, Y):
  model = gensim.models.Word2Vec.load('gensim.models.Word2Vec.bin')
  score = model.wv.similarity(w1=X, w2=Y)
  return score

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = word_tokenize(sentence)
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

# createModel()
model = gensim.models.Word2Vec.load('gensim.models.Word2Vec.bin')
# score = calculateSimilarity('money', 'penny')
# print(score)
index2word_set = set(model.wv.index2word)
s1_afv = avg_feature_vector('Sun is bright and yellow', model=model, num_features=150, index2word_set=index2word_set)
s2_afv = avg_feature_vector('yellow and bright is the sun', model=model, num_features=150, index2word_set=index2word_set)
sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
print(sim)