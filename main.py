import gensim
import numpy as np
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy import spatial
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

def gensimWord2Vec(X, Y):
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

  model = gensim.models.Word2Vec.load('gensim.models.Word2Vec.bin')
  index2word_set = set(model.wv.index2word)
  s1_afv = avg_feature_vector(X, model=model, num_features=150, index2word_set=index2word_set)
  s2_afv = avg_feature_vector(Y, model=model, num_features=150,
    index2word_set=index2word_set)
  similarity = 1 - spatial.distance.cosine(s1_afv, s2_afv)
  return similarity

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
    result_summary.append({
      'methodName': 'Gensim Word2Vec',
      'similarityScore': gensimWord2Vec(X, Y)
    })
    print(result_summary)

  return render_template('index.html', result_summary=result_summary, len=len(result_summary), X=X, Y=Y)


if __name__ == "__main__":
  app.run(debug=True)