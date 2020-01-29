import gensim
import smart_open
from nltk import sent_tokenize, word_tokenize
from gensim.test.utils import common_texts


def read_corpus(fname, tokens_only=False):
  with smart_open.open(fname, encoding="iso-8859-1") as f:
    for i, line in enumerate(f):
      tokens = gensim.utils.simple_preprocess(line)
      if tokens_only:
        yield tokens
      else:
        # For training data, add tags
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def createWord2VecModel():
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
  model.save('models/gensim.models.Word2Vec.bin')
  print('Model created and saved successfully')

def createDoc2VecModel():
  gen_docs = list(read_corpus('datasets/hotel_reviews.txt'))
  # build vocabulary and train model
  model = gensim.models.Doc2Vec(vector_size=50, min_count=2, epochs=40)
  model.train(sentences=gen_docs, total_examples=len(gen_docs), epochs=40)
  model.build_vocab(gen_docs)
  model.save('models/gensim.models.Doc2Vec.bin')
  print('Model created and saved successfully')

def createFastTextModel():
  model = gensim.models.FastText(size=4, window=3, min_count=1)
  gen_docs = list(read_corpus('datasets/hotel_reviews.txt'))
  texts = common_texts
  model.build_vocab(common_texts)
  model.train(sentences=common_texts, total_examples=len(common_texts), epochs=40)
  model.save('models/gensim.models.FastText.bin')
  print('Model created and saved successfully')

createFastTextModel()
# model = gensim.models.Doc2Vec.load('models/gensim.models.FastText.bin')
# score = model.wv.similarity(w1='computer', w2='computer')
# print(score)