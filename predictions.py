# this is the file while will do predictions

# get the data - csv
import pandas as pd


import unicodedata
from bs4 import BeautifulSoup
from normalization import expand_contractions, CONTRACTION_MAP, lemmatize_text
from normalization import remove_special_characters, keep_text_characters, remove_stopwords


# normalize the data
def normalize_documents(doc_list):
    normalize_doc_list = []
    for doc in doc_list:
        doc1 = unicodedata.normalize('NFKD',doc).encode('ascii', 'ignore')
        doc2 = BeautifulSoup(doc1, 'html.parser').get_text()
        doc3 = expand_contractions(doc2, CONTRACTION_MAP)
        doc4 = lemmatize_text(doc3)
        doc5 = remove_special_characters(doc4)
        doc6 = keep_text_characters(doc5)
        doc7 = remove_stopwords(doc6)
        normalize_doc_list.append(doc7)
    return normalize_doc_list


import pickle
classifier = pickle.load(open('classifier', 'rb'))
vectorizer = pickle.load(open('vectorizer', 'rb'))

# validate test set
y_predicted = classifier.predict(vectorizer.transform(normalize_documents(["this movie is too good, what can I do"])))
print(y_predicted)