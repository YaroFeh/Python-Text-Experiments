from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plop


# let's explore bag-of-words!
# it's yet another way to vectorize a set of sentences based on the frequency of words that appear over and over again.

# we'll see how bag-of-words is used to do sentiment classification!
# we'll focus on just two kinds of sentiment, "delicious" and "not very good".

# first, gotta prepare your training data.
sentences = ["This food is tasty.", "The chicken was pretty good.", "Yesterday's toad meat was disgusting", "Yes I imagine the sofa would be very palatable", "Of course the doorknob would taste horrible", "You'd think uncooked dough wouldn't be awful but it is"]
y = ["delicious", "delicious", "not very good", "delicious", "not very good", "not very good"]

# we'll later download a dataset that gives us a whole bunch of food reviews.

# Write code that will omit periods, commas, and make everything lowercase if it is not already.
for f in range (len(sentences)):
    sentences[f] = (sentences [f]).replace(".","")
    sentences[f] = (sentences [f]).replace(",","")
    sentences[f] = (sentences [f]).replace("!","")
    sentences[f] = (sentences [f]).replace(";","")
    sentences[f] = (sentences [f]).replace(":","")
    sentences[f] = (sentences [f]).replace("'","")
    sentences[f] = (sentences [f]).replace("?","")
    sentences[f] = (sentences [f]).replace("-","")
    sentences[f] = (sentences [f]).lower()
print(sentences)
# split the sentences into training and test datasets
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# print each one to understand what you get!
print(sentences_train)
print(sentences_test)
print(y_train)
print(y_test)
# which one of these bag-of-words models should we use?
vectorizer = CountVectorizer(lowercase=False)
#vectorizer = CountVectorizer()

# train the bag-of-words model with the training dataset.
vectorizer.fit(sentences_train)

# let's see what all the different words the bag-of-words model got.
print(vectorizer.vocabulary_)
print(vectorizer.transform(["wouldnt uncooked toad meat be sofa food"]))
# we will use a simple logistic regression model to make the sentiment classification model.
classifier = LogisticRegression()

# we have to vectorize the training and test sentences to train the sentiment classification model.
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

# train the model
classifier.fit(X_train, y_train)
# let's see how it does on the test data.
score = classifier.score(X_test, y_test)

# print the score.
print(score)
# let's see what the classifier outputs for an example sentence:
p = vectorizer.transform(["I like hot dogs."])
print(classifier.predict(p))

# print what the classifier outputs for each sentence in train one at a time.  The results should look like 
# sentence 1 : score 1
# sentence 2 : score 2
# and on.
for i in range (len(sentences_train)):
    q = vectorizer.transform([sentences_train [i]])
    print(sentences_train[i] + " : " + str(classifier.predict(q)))
print(classifier.score(X_train, y_train))
b = []
v = []
k = {1:"Vomit-inducing", 2:"Just bad", 3:"Mediocre", 4:"Nice:)", 5:"Godly"}
t = open("foods.txt", "r")
m = t.readline
for line in t:
    if "review/score" in line:
        line = line.replace("review/score: ", "")
        line = line.replace("\n", "")
        w = float(line)
        v.append(k[w])
    if "review/text" in line:
        line = line.replace("review/text: ", "")
        line = line.replace("\n", "")
        line = line.replace("<br />", " ")
        line = (line).replace(".","")
        line = (line).replace(",","")
        line = (line).replace("!","")
        line = (line).replace(";","")
        line = (line).replace(":","")
        line = (line).replace("'","")
        line = (line).replace("?","")
        line = (line).replace("-","")
        line = (line).lower()
        b.append(line)

toad = LogisticRegression()
tray, tesd, q_tray, q_tesd = train_test_split(b, v, test_size = 0.25, random_state = 1000)

vizer = CountVectorizer(lowercase = False)
vizer.fit(tray)
xTray = vizer.transform(tray)
xTesd = vizer.transform(tesd)
toad.fit(xTray, q_tray)
scare = toad.score(xTesd, q_tesd)
#for thing in range (len(tray)):
 #   o = vizer.transform([tray [thing]])
  #  print(tray[thing] + " : " + str(toad.predict(o)))
print(toad.score(xTray, q_tray))
print(scare)
fest = []
thatch = []
raining = []
# LATER: do you see where we did test_size = 0.25?  what do you think this little piece of code does?  What happens as this value increases?  Write code that explores the relationship between this value and the accuracy of the classification model.
for whatever in range (25, 100, 10):
    tray, tesd, q_tray, q_tesd = train_test_split(b, v, test_size = (whatever/100), random_state = 1000)
    vizer = CountVectorizer(lowercase = False)
    vizer.fit(tray)
    xTray = vizer.transform(tray)
    xTesd = vizer.transform(tesd)
    toad.fit(xTray, q_tray)
    scare = toad.score(xTesd, q_tesd)
    print(str (whatever/100))
    print("train score: " + str(toad.score(xTray, q_tray)))
    print("test score: " + str(scare))
    fest.append(scare)
    raining.append(toad.score(xTray, q_tray))
    thatch.append(whatever/100)
plop.plot(thatch, fest)
plop.plot(thatch, raining)
plop.xlabel("Batch Size")
plop.ylabel("Scores")
plop.show()