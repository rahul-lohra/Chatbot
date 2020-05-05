import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

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

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

list_of_words_in_patterns = docs_x

for x, word_list in enumerate(list_of_words_in_patterns):
    bag = []

    current_word_list = [stemmer.stem(w.lower()) for w in word_list]

    for w in words:
        if w in current_word_list:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

print("h")