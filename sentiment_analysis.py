# get the data - csv
import pandas as pd
import pickle
df = pd.read_csv("movie_reviews.csv", encoding="ISO-8859-1")



# prepare training and test data
import numpy as np
X = np.array(df["review"])
y = np.array(df["sentiment"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

import unicodedata
from bs4 import BeautifulSoup
from normalization import normalize_documents

# # normalize the data
# def normalize_documents(doc_list):
#     normalize_doc_list = []
#     for doc in doc_list:
#         doc1 = unicodedata.normalize('NFKD',doc).encode('ascii', 'ignore')
#         doc2 = BeautifulSoup(doc1, 'html.parser').get_text()
#         doc3 = expand_contractions(doc2, CONTRACTION_MAP)
#         doc4 = lemmatize_text(doc3)
#         doc5 = remove_special_characters(doc4)
#         doc6 = keep_text_characters(doc5)
#         doc7 = remove_stopwords(doc6)
#         normalize_doc_list.append(doc7)
#     return normalize_doc_list

# extract the features
X_train_normalized = normalize_documents(X_train)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=0.0, max_df=1.0, ngram_range=(1, 1))
X_train_features = tfidf_vectorizer.fit_transform(X_train_normalized)


# train classifier
from sklearn.linear_model import SGDClassifier
sgd_classifier = SGDClassifier(loss='hinge', penalty='l2', n_iter=500).fit(X_train_features, y_train)


pickle.dump(sgd_classifier, open('classifier', 'wb'))

# validate test set
y_predicted = sgd_classifier.predict(tfidf_vectorizer.transform(normalize_documents(X_test)))
pickle.dump(tfidf_vectorizer, open('vectorizer', 'wb'))

# stats - confusion matrix, accuracy, precision and recall
from sklearn import metrics
print("{}".format(metrics.confusion_matrix(y_true=y_test, y_pred=y_predicted)))
print("Accuracy {}".format(metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)))
print("Precision(positive) {}".format(metrics.precision_score(y_true=y_test, y_pred=y_predicted, pos_label='positive')))
