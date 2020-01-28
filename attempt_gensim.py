import gensim
from nltk import sent_tokenize, word_tokenize

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

def calculateSimilarity(X, Y):
  model = gensim.models.Word2Vec.load('gensim.models.Word2Vec.bin')
  score = model.wv.similarity(w1=X, w2=Y)
  return score

# createModel()
score = calculateSimilarity('money', 'penny')
print(score)