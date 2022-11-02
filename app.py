import streamlit as st

import pickle 
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 

ps = PorterStemmer( )


def data_clening(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    x = []
    
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)
    z = []
    
    for i in x:
        z.append( ps.stem(i))
        
    
    return ' '.join(z)




tfid = pickle.load(open('vectoriz.pkl', 'rb'))
model = pickle.load(open('nodel.pkl', 'rb'))


st.title('SMS Spam Detection')
input_sms = st.text_area('Enter SMS')

if st.button('Predict'):
    
    data_cl = data_clening(input_sms)
    
    vac = tfid.transform([data_cl])
     
    result = model.predict(vac)[0]
    
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')



