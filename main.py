import json
import random

import nltk
import numpy
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []  # list of list of tokenized words (patterns)
    docs_y = []  # list of intents mapped with docs_x with index position

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # print(json.dumps(labels, indent=4))
    # docs_x = [['Hi'], ['How', 'are', 'you'], ['Is', 'anyone', 'there', '?']...]
    # docs_y = ['greeting', 'greeting', 'greeting', 'greeting'....]

    # Need to reduce the vocabulary of our chatbox by using stemming

    # words = ['Hi', 'How', 'are', 'you', 'Is', 'anyone', 'there', '?', 'Hello', 'Good', 'day', 'Whats', 'up', 'cya', 'See', 'you', 'later', 'Goodbye', 'I', 'am', 'Leaving', 'Have', 'a', 'Good', 'day', 'how', 'old', 'how', 'old', 'is', 'tim', 'what', 'is', 'your', 'age', 'how', 'old', 'are', 'you', 'age', '?', 'what', 'is', 'your', 'name', 'what', 'should', 'I', 'call', 'you', 'whats', 'your', 'name', '?', 'Id', 'like', 'to', 'buy', 'something', 'whats', 'on', 'the', 'menu', 'what', 'do', 'you', 'reccommend', '?', 'could', 'i', 'get', 'something', 'to', 'eat', 'when', 'are', 'you', 'guys', 'open', 'what', 'are', 'your', 'hours', 'hours', 'of', 'operation']
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    # words = ['a', 'ag', 'am', 'anyon', 'ar', 'buy', 'cal', 'could', 'cya', 'day', 'do', 'eat', 'get', 'good', 'goodby', 'guy', 'hav', 'hello', 'hi', 'hour', 'how', 'i', 'id', 'is', 'lat', 'leav', 'lik', 'menu', 'nam', 'of', 'old', 'on', 'op', 'reccommend', 'see', 'should', 'someth', 'the', 'ther', 'tim', 'to', 'up', 'what', 'when', 'yo', 'you']

    labels = sorted(labels)

    # Bag of Words - https://techwithtim.net/tutorials/ai-chatbot/part-2/
    # it is array of integers - which represents number of occurances of word

    # We need of array of integers bcoz - neural network expects an array of ints

    training = []  # filling using doc_x (patterns) in form of  0 & 1
    output = []  # filling using labels - on which index above (training) word lies

    out_empty = [0 for _ in range(len(labels))]

    list_of_words_in_patterns = docs_x
    list_of_labels_mapped_with_patters = docs_y

    for x, word_list in enumerate(list_of_words_in_patterns):
        bag = []

        current_word_list = [stemmer.stem(w.lower()) for w in word_list]

        for w in words:
            if w in current_word_list:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(list_of_labels_mapped_with_patters[x])] = 1

        training.append(bag)
        output.append(output_row)

    # PART 3 - DEVELOPING A MODEL
    # We will use Neural network with 2 hidden layers
    # The goal of our network will be to look at a bag of words and give a class(label) that they belong too

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Training & Saving the Model

# n_epoch = amount of times that the model will see the same information while training.
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I didn't get that")


chat()
