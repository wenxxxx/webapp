import pickle
import gensim
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

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
vectorizer = TfidfVectorizer()
texte = st.text_area('Enter some text here')

if st.button('Predict the text'):
    soup = texte.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
    tokens = soup.apply(lambda x:preprocess(x))
    X = tokens.apply(lambda x: " ".join(x))
    X_vec = vectorizer.fit_transform(X)
    arr = np.array(X_vec)
    while arr.size < 28075:
        arr = np.append(arr, 0)
    X_vec_new = arr.reshape((1, 28075))
    # importer le modele
    model = pickle.load(open('../Web_app/svc_v.pkl', 'rb'))
    prediction = model.predict(X_vec_new)
    tags = mlb.inverse_transform(prediction)
    df_pred = pd.DataFrame(tags)
    st.write(df_pred)

# Afficher le widget de téléchargement de fichier
#fichier = st.file_uploader("Télécharger un fichier")

#if st.button('Predict the file'):
    #st.write()