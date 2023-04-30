import pickle
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#import tensorflow_hub as hub

# Define the Streamlit app
st.title('Stack Overflow Question Tagger')

# Load the Universal Sentence Encoder
#use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

y = [['android', 'c', 'c#', 'c++', 'css', 'html', 'ios', 'java', 'javascript', 'jquery', 'net', 'php', 'python', 'r']]
mlb = MultiLabelBinarizer()
y_ = mlb.fit_transform(y)
vectorizer = TfidfVectorizer()
texte = st.text_area('Enter some text here')

if st.button('Predict the text'):
    soup = BeautifulSoup(texte, 'html.parser')
    tokens = simple_preprocess(texte)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    X = [" ".join(tokens)]
    vectorizer.fit(X)
    X_vec = vectorizer.transform(X)
    #X_emb = use(X)
    # importer le modele
    model = pickle.load(open('../Web_app/svc_v.pkl', 'rb'))
    prediction = model.predict(X_vec)
    tags = mlb.inverse_transform(prediction)
    df_pred = pd.DataFrame(tags)
    st.write(df_pred)

# Afficher le widget de téléchargement de fichier
#fichier = st.file_uploader("Télécharger un fichier")

#if st.button('Predict the file'):
    #st.write()