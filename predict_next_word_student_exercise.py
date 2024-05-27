import pandas as pd
import os
import numpy as np
import pickle
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from nltk.corpus import stopwords
nltk.download('stopwords')
print(stopwords.words('english'))

import matplotlib.pyplot as plt

for uwah in stopwords.words('english'):
    uwah = " " + uwah + " "
# Step 1: Obtain the reviews and split them into separate sentences.  Make sure to remove punctuations and other non-word characters.

# Covert list of sentences into pandas dataframe.
tad = []
wah = 0
dat = open("foods.txt", "r")
for line in dat:
    if "review/text" in line:
        line = line.replace("review/text: ", " ")
        line = line.replace("\n", " ")
        line = line.replace("<br />", " ")
        line = line.replace("."," ")
        line = line.replace(","," ")
        line = line.replace("!"," ")
        line = line.replace(";"," ")
        line = line.replace(":"," ")
        line = line.replace("'","")
        line = line.replace("?"," ")
        line = line.replace("-"," ")
        for haw in stopwords.words('english'):
            line = line.replace(haw, "")
        tad.append(line)
        wah+=1
        if wah > 100:
            break
final_dataset = pd.DataFrame(tad)
print(final_dataset)

# Train vectorizer on the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(final_dataset[0])
# Step 2: Obtain the vector for a specific word; use command tokenizer.word_index['word']
# Get the vector for five different words of your choosing; test some that are in the reviews and some not in the reviews.
# Also print the total number of words that are in the vectorizer; hint: tokenizer.word_index returns a list of the vectors of ALL the words the vectorizer has seen.
#stuff = [tokenizer.word_index['molts'], tokenizer.word_index['weavil'], tokenizer.word_index['route'], tokenizer.word_index['spoooooky'], tokenizer.word_index['allopathic']]
total_words = len(tokenizer.word_index)+1
#print("The stuff: " + str(stuff))
print("Total words: " + str(total_words))
# Step 3: With vectors for each word, construct a list of vectors for every part of each sentence.
input_sequences = []
for line in final_dataset[0]:
    token_list = tokenizer.texts_to_sequences([line])[0]    
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
#print(input_sequences)
# Print the first few sentences and the elements in input_sequences that correspond to them, as well as the total number of elements.
first = []
firstSentence = pd.DataFrame(tad)[0]
first_list = tokenizer.texts_to_sequences([firstSentence[0]])[0]
for aa in range(1,len(first_list)):
    grey = first_list[:aa+1]
    first.append(grey)
print(firstSentence[0])
print(first)

# LATER: Only obtain vectors for entire sentences, not subsets of it.  Then train with these vectors and see how the performance of the model changes.

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Print the first few elements of input_sequences.  What has changed?  Why is this change required?

# create inputs and outputs to the model; the inputs will be the vectors up to but not including the last vector, and the outputs will be the last vector.  Why?  Because we're trying to predict the last word given all the words before it.
inputs = input_sequences[:,:-1]
outputs = input_sequences[:,-1]
# required for the model.
final_outputs = tf.keras.utils.to_categorical(outputs, num_classes=total_words)

# The model construction!
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(learning_rate=.12)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# train the model
model_performance = model.fit(inputs, final_outputs, epochs=1, steps_per_epoch = 2,verbose=1)

# Plot the model performance epochs vs. accuracy.  Add the x-axis and y-axis labels and title.
plt.plot(model_performance.history["accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Here's my Graph!")
plt.show()
with open('modelTrains.pkl','wb') as f:
    pickle.dump(model_performance,f)
# Testing time!  Come up with a random phrase and have the model predict the next word.
some_text = "I would like a "

# Go through the usual steps: Vectorize the words then add padding.
token_list = tokenizer.texts_to_sequences([some_text])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')#

# Use trained model to predict number.
predicted_temp = model.predict(token_list)
predicted = np.argmax(predicted_temp, axis = 1)

# Find word in vectorizer that corresponds to the predicted number.
output_word = ""
for word, index in tokenizer.word_index.items():
   if index == predicted[0]:
       output_word = word
       break

# Task: Print the starting words and the predicted word together to see the entire text.
some_text += output_word
# Challenge 1: Write code that will not just predict the next one word, but several one after the other to create a long string of text.
i = 1
nono = ""
for i in range (0,5):
    some_text += " "
    token_list = tokenizer.texts_to_sequences([some_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_temp = model.predict(token_list)
    predicted = np.argmax(predicted_temp, axis = 1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted[0] and word != nono:
            output_word = word
            break
#        else if word == nono:
            
    some_text += output_word
    nono = output_word
print(some_text)
# Challenge 2: Write code that explores the relationship between how well the model performs from changing values of the Adam optimizer learning rate.  Find the optimal learning rate value from doing this.
spork = np.arange(0.01, 1, 0.05)
spoon = 0
knife = 0
for fork in spork:
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(learning_rate=fork)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model_performance = model.fit(inputs, final_outputs, epochs=1, steps_per_epoch = 2,verbose=1)
    if (model_performance.history["accuracy"][0]) > spoon:
        spoon = model_performance.history["accuracy"][0]
        knife = fork
    print(fork)
    print(model_performance.history["accuracy"][0])
print(knife)
print(spoon)