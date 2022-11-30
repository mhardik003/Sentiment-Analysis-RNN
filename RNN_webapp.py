# Importing the required libraries
from keras.datasets import imdb #importing the imdb dataset from the keras library
from keras.utils import pad_sequences
import os
import re
from keras.models import load_model
import numpy as np

import streamlit as st
import string



# Setting the webserver
st.set_page_config(page_title='Sentiment Analyser')
st.title("The Psychic")
st.header("Enter your review below ")
review = st.text_input("","It is a good movie")
st.subheader("Result : ")
sentiment=""


# Creating the word2id functions to change the review obtained to a vector
vocabulary_size = 5000 # setting the vocabulary size to 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
max_words=500 # setting the maximum number of words per document (for both training and testing)
word_to_id=imdb.get_word_index()
word_to_id={k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"]=0
word_to_id["<START>"]=1
word_to_id["<UNK>"]=2



# Preprocessing
review=re.sub(r'[^A-Za-z ]','',review)
review=review.lower().split()

review = [w if w in word_to_id else '<UNK>' for w in review ]

review=["<START>"]+review


# Finalizing the Vector
review_ids=[np.array([word_to_id[w] for w in review])]
review_ids=pad_sequences(review_ids,maxlen=max_words)
review_ids=[np.array(review_ids)]
 
# Loading the RNN model made
cache_dir = os.path.join("cache", "sentiment_analysis")
model_file = "rnn_model.h5"  # HDF5 file
model = load_model(os.path.join(cache_dir, model_file))


#Predicting the sentiment of the review
predicted_score=model.predict(review_ids)[0]



# since 0 is negative and 1 is positive
if(predicted_score<=0.4):
    st.write("The review is negative")

else:
    st.write("The review is positive")

