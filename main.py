from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('all')

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