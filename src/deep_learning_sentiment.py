## Import pandas to read in data
import pandas as pd
import numpy as np
from numpy import array

## import nltk
import nltk

# import keras to embed word and neural network
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.layers import Input

# import sklearn model selection
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import preprocessing

# import plotting
from matplotlib import pyplot as plt


""" Data Preparation """
data = pd.read_csv("data.txt", delimiter="\t", header=None)
data.columns = ["rating", "reviews"]
data['rating'].hist()

# drop rate-30 class
data['rating'] = data['rating'].apply(lambda x: np.nan if x == 30 else int(x))  
data = data.dropna()

""" unused code 
# drop all the rates in 10 && 30
# data_10 = data[data['rating'] == 10]
# data_10 = data_10.sample(n = 500)
# data['rating'] = data['rating'].apply(lambda x: np.nan if x in [10,30] else int(x))  
# data = data.dropna()

# append rates 10 dataframe with reduced numbers 
# data = data.append(data_10)
# data['rating'].value_counts()
"""

# convert numeric rates to positive and negative 
data["rating"] = data.loc[:]["rating"].apply(lambda x: 1 if x>30 else 0)
data
y = data['rating']
y = np.array(list(y))

# create empty list for X
X = []
reviews = list(data['reviews'])
# copy all the reviews into X
for review in reviews:
    X.append(review)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7)

""" Text Processing """
# tokenize words
# create embedded layers - convert sentences to number
tokenizer = Tokenizer(num_words=5000) 
tokenizer.fit_on_texts(X_train)

train_embedded = tokenizer.texts_to_sequences(X_train)
test_embedded = tokenizer.texts_to_sequences(X_test)

# the corpus contains 15163 unique words
vocab_size = len(tokenizer.word_index) + 1

## make all the sentences of uniform size - the length of longest sentences
# find out the longest sentences and get the length - length should be uniformed in training and testing dataset - use longest length
longest_sent = max(X_train, key=lambda sent: len(nltk.word_tokenize(sent)))
len_longest = len(nltk.word_tokenize(longest_sent))

# increase the length of sentences by padding
padding_sent = pad_sequences(train_embedded, len_longest, padding="post")
padding_sent_test = pad_sequences(test_embedded, len_longest, padding="post")

## Glove embedding
# create empty list for embedding dictionary
embedding_dict = {}
with open("glove.6B.100d.txt", 'r', encoding='utf8') as f:
    for line in f:
        # read the glove txt line by line and split the text  
        records = line.split()
        # take the first items of each record out 
        word = records[0]
        # convert following items of each record to array
        vector_dimensions = np.asarray(records[1:],dtype='float32')
        # use word as key and vector dimensions as value in the dictionary
        embedding_dict[word] = vector_dimensions
embedding_dict

# create a shape-specified matrix,  15117 * 100
embedding_matrix = np.zeros(shape=(vocab_size, 100))
for key, index in tokenizer.word_index.items():
    # match the items between tokenizer.word_index and embedding dictionary
    embedding_vector = embedding_dict.get(key)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

""" Modeling - Neural Network """
# Input layer
deep_input = Input(shape=(len_longest,))
# add embedding layer 
embedding_layer = Embedding(vocab_size, 100, 
                    weights=[embedding_matrix], 
                    input_length=len_longest, trainable=False)(deep_input)
# Hidden layer - Long short term memory
LSTM_layer = LSTM(128)(embedding_layer)
# Output layer - Densed layer 
densor_layer = Dense(1, activation='sigmoid')(LSTM_layer) # The sigmoid function is used for the two-class, whereas the softmax function is used for the multiclass
model = Model(inputs=deep_input, outputs=densor_layer)
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

# fit the model
neuralnet = model.fit(padding_sent, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2)

# evaluate - return 2 values cost and accuracy
cost, accuracy = model.evaluate(padding_sent_test, y_test, verbose=1)
print('Accuracy: %f' %(accuracy*100))
print('Cost: %f' %(cost))

""" Plotting """
## plotting
plt.plot(neuralnet.history['acc'])
plt.plot(neuralnet.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()

plt.plot(neuralnet.history['loss'])
plt.plot(neuralnet.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
