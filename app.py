import sys
import os
import shutil
import time
import traceback
import normalization

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)


model_directory = 'model'
classifier_path = '%s/classifier' % model_directory
vectorizer_path = '%s/vectorizer' % model_directory

# These will be populated at training time
vectorizer = None
classifier = None

@app.route('/', methods=['GET'])
def index():
    return jsonify({"predict": "use '/predict?query=...' for sentiment analysis"})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if classifier:
        try:
            # here we want to get the value of user (i.e. ?query=some-value)
            query = request.args.get('query')
            # query = pd.get_dummies(pd.DataFrame(json_))
            text = normalization.normalize_documents([query])
            prediction = classifier.predict(vectorizer.transform(text))
            return jsonify({'prediction': prediction[0]})
            # print('prediction :' + prediction)

        except Exception as e:
            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'


if __name__ == '__main__':

    try:
        port = 5001
        classifier = joblib.load(classifier_path)
        print('model loaded')
        vectorizer = joblib.load(vectorizer_path)
        print('vectorizer loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        classifier = None

    app.run(host='127.0.0.1', port=port, debug=True)
