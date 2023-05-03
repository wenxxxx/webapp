import pickle
import numpy as np
import gensim
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(token)
    return result

# Define the Streamlit app
st.title('Stack Overflow Question Tagger')

y = [['android', 'c', 'c#', 'c++', 'css', 'html', 'ios', 'java', 'javascript', 'jquery', 'net', 'php', 'python', 'r']]
mlb = MultiLabelBinarizer()
y_ = mlb.fit_transform(y)
vec = pickle.load(open('https://github.com/wenxxxx/webapp/blob/methode_Tfdf/vec.pkl', 'rb'))
texte = st.text_area('Enter some text here')
data ={'body':texte}
df =pd.DataFrame(data,index=[0])

if st.button('Predict the text'):
    df['soup'] = df['body'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    df['token'] = df['soup'].apply(lambda x:preprocess(x))
    X = df['token'] .apply(lambda x: " ".join(x))
    X_vec = vec.transform(X)
    # importer le modele
    model = pickle.load(open('https://github.com/wenxxxx/webapp/blob/methode_Tfdf/svc_v.pkl', 'rb'))
    prediction = model.predict(X_vec)
    tags = mlb.inverse_transform(prediction)
    df_pred = pd.DataFrame(tags)
    st.write(df_pred)

# Afficher le widget de téléchargement de fichier
#fichier = st.file_uploader("Télécharger un fichier")

#if st.button('Predict the file'):
    #st.write()
