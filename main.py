import gensim
import numpy as np
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
# nltk.download('all')

app = Flask(__name__)


def cosineSimilarity(X, Y):
  # tokenization
  X_list = word_tokenize(X)
  Y_list = word_tokenize(Y)

  # sw contains the list of stopwords
  sw = stopwords.words('english')
  l1 = [];
  l2 = []

  # remove stop words from string
  X_set = {w for w in X_list if not w in sw}
  Y_set = {w for w in Y_list if not w in sw}

  # form a set containing keywords of both strings
  rvector = X_set.union(Y_set)
  for w in rvector:
    if w in X_set:
      l1.append(1)  # create a vector
    else:
      l1.append(0)
    if w in Y_set:
      l2.append(1)
    else:
      l2.append(0)
  c = 0

  # cosine formula
  for i in range(len(rvector)):
    c += l1[i] * l2[i]
  cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
  return cosine

def usingGensim(X, Y):
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
  sims = gensim.similarities.Similarity('workdir/', tf_idf[corpus],
    num_features=len(dictionary))

  # creating query document with sentence Y
  tokens = sent_tokenize(Y)
  for line in tokens:
    Y_docs.append(line)

  for line in Y_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)   #update an existing dictionary and create bag of words

  query_doc_tf_idf = tf_idf[query_doc_bow]
  print('Comparing Result:', sims[query_doc_tf_idf])

@app.route("/", methods=['GET', 'POST'])
def index():
  X = ''
  Y = ''
  result_summary = [
    {
      'methodName': 'Method name',
      'similarityScore': 'Similarity Score'
    }
  ]
  if request.method == 'POST':
    X = request.form['string1']
    Y = request.form['string2']
    print(X)
    print(Y)

    result_summary.append({
      'methodName': 'Cosine Similarity',
      'similarityScore': cosineSimilarity(X, Y)
    })
    print(result_summary)

  return render_template('index.html', result_summary=result_summary, len=len(result_summary), X=X, Y=Y)


if __name__ == "__main__":
  app.run(debug=True)