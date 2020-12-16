#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stopwords_l = stopwords.words('english')
stopwords_l.remove('not') 
from PIL import Image

#app=Flask(__name__)
#Swagger(app)
embedding_dict = pickle.load(open("embedding_dictionary.pkl", "rb"))
pickle_in = open("RF_boost_pos.pkl","rb")
rf_boost_pos=pickle.load(pickle_in)
embedding_dictionary = open("embedding_dict.pkl","rb")
#@app.route('/')
def welcome():
    return "Welcome All"

import re
def text_conversion_deployment(x):
    s = ['don','aren','couldn','didn','doesn','hadn','hasn','haven','isn','mightn','mustn','needn','shouldn','wasn','weren','won','wouldn']
    rev_list = []
    st = re.sub('[^a-zA-Z]',' ',x)
    st = st.lower()
    st = st.strip().split()
    update = ''
    for word in st:
        if word  in s:
            word = word[:-1]+' not'
            update = update+' '+word
        else:
            update = update+' '+word
    rev_list.append(update.strip())

    return corpus_text(x)
#####################################################################
######################################################################
def corpus_text(x):
  lem = WordNetLemmatizer()
  rv = re.sub('[^a-zA-Z]',' ',x)
  rv = rv.lower()
  rv = rv.split()
    
  rv = [lem.lemmatize(word) for word in rv if not word in stopwords_l]
  rv =' '.join(rv)
  tokens = word_tokenize(rv)
  #tokens
  sentence = np.zeros(300)
  for word in tokens:
    try:
      sentence += embedding_dict[word]   
            
    except KeyError:
          continue
  dimens = pd.DataFrame(sentence.reshape(1,300)) #converting into dataframe and returning
  return dimens

########################################################################################

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Additional_Number_of_Scoring,Average_Score,
                                Total_Number_of_Reviews_Reviewer_Has_Given,
                                           Reviewer_Score,Hotel_Country,review_month,Trip,Stayed_nights,yr,review):

    others_df = pd.DataFrame(np.array([Additional_Number_of_Scoring,Average_Score,
                                Total_Number_of_Reviews_Reviewer_Has_Given,
                                           Reviewer_Score,Hotel_Country,review_month,Trip,Stayed_nights,yr]).reshape(1,9))
    dimens = text_conversion_deployment(review)
    
    full_inputs = pd.concat([others_df,dimens],axis=1)
    

    prediction=rf_boost_pos.predict(full_inputs)
    print(prediction)
    return prediction



def main():
    st.title("Hotel Reviews Predictions")
    html_temp = """
    <div style="background-color:lightgray;padding:8px">
    <h2 style="color:blue;text-align:center;">Hotel Reviews Predictions ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Additional_Number_of_Scoring = st.text_input("Additional_Number_of_Scoring","Type Here")
    Average_Score = st.text_input("Average_Score","Type Here")
    Total_Number_of_Reviews_Reviewer_Has_Given = st.text_input("Total_Number_of_Reviews_Reviewer_Has_Given","Type Here")
    Reviewer_Score = st.text_input("Reviewer_Score","Type Here")
    Hotel_Country = st.text_input("Hotel_Country","Type Here")
    review_month = st.text_input("review_month","Type Here")
    Trip = st.text_input("Trip","Type Here")
    Stayed_nights = st.text_input("Stayed_nights","Type Here")
    yr = st.text_input("yr","Type Here")
    review = st.text_input("review","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Additional_Number_of_Scoring,Average_Score,Total_Number_of_Reviews_Reviewer_Has_Given,
                                           Reviewer_Score,Hotel_Country,review_month,Trip,Stayed_nights,yr,review)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Hotel Reviews ML model is predicting the positivity and negativity in the reviews given by customers")
        st.text("--")

if __name__=='__main__':
    main()


# In[ ]:




