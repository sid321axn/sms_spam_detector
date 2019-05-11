

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def predict():
    url = "http://bit.ly/2W1fRmS"
    df = pd.read_csv(url, encoding='latin-1')
    df['label'] = df["v1"].map({'spam': 1, 'ham': 0})
    y = df['label']
    cv = CountVectorizer(stop_words='english')
    X = cv.fit_transform(df["v2"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    if request.method == 'POST':
        message = request.form['comment']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('results.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
