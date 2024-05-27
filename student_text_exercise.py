import nltk
#nltk.download('gutenberg')
nltk.download('punkt')
from nltk.corpus import gutenberg
import spacy

# let's look at what the gutenberg data has
print(gutenberg.fileids())

# pick one of these and let's look at how many words it has
h = gutenberg.words('edgeworth-parents.txt')
print(len(h))

# let's look at the actual words
sentences = gutenberg.sents('edgeworth-parents.txt')
print(sentences)

# if you want to just look at the 10th sentence, what do you have to do?
print(sentences [10])


# let's take a look at the vectors (i.e. the numbers) that represent the words

# nlp is set to the already-trained machine-learning model that can already convert words, phrases, and sentences into vectors.
nlp = spacy.load('en_core_web_lg')
processed_words = nlp('This is an example!')
for token in processed_words:
	print(token)
	
	v = token.vector
	print(v [0:10])
	
	# #print just the first few values in the vector
	
# with the libraries we are using, we have the capability of finding the similarities of two words with their vectors.
#To-do: get processed words for two different example sentences and find their similarity.

g = nlp('I like rice noodles')
r = nlp('I like udon noodles')
n = nlp('The strainer is really cool')
w = nlp('Spatulas are nice and flat')
c = g.similarity(r)
l = n.similarity(w)
print(c)
print(l)
# play around with punctuation, synonyms, and antonyms
# why does similarity look like this?  how does similarity have a numerical value?  what do you think the range of values is?
#p = nlp('i like this thing')
#s = nlp('i do not like this thing')
#q = p.similarity(s)
#print(q)
# To-do: Combine all these and let's determine what two books are most similar to each other?
#t = str('')
#a = str('')
#for s in range (len(gutenberg.words('edgeworth-parents.txt'))):
#    t += (h [s])
#    t += ' '
#for u in range (len(gutenberg.words('milton-paradise.txt'))):
#    a += (z [u])
#    a += ' '
#g = nlp(t)
#i = nlp(a)
#tad = g.similarity(i)
#print(tad)
#get every combination and print the most similar
answer = 0
chickendinner = ''
o = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
v = str('')
for z in range (len(gutenberg.fileids())):
    for i in range(z + 1, len(gutenberg.fileids())):
        x = gutenberg.words(o[z])
        b = gutenberg.words(o[i])
        for k in range (50):
            v += (x[k])
            v += ' '
            c = nlp(v)
            
            j = str('')    
            j += (b[k])
            j += ' '
            r = nlp(j)
        
        y = str(o[i])
        p = str(o[z])
        print(y + " and " + p + ":")
        h = c.similarity(r)
        if (h - answer >= 0):
            answer = h
            chickendinner = y + " and " + p + ": " + str(answer)
        print(h)
        v = ''
        j = ''
print("Winner winner chicken dinner: " + chickendinner)

#bag of words