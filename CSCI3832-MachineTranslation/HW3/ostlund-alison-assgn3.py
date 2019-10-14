#Alison Ostlund
#CSCI3832 HW3
#Text Classification
#Collaborated with Matthew Donovan

from collections import Counter
import string
import math



#read in neg text file, read in line by line ignoring the ID at beginning of each sentence
neg = open("hotelNegT-train.txt").readlines()
negdict = []
for line in neg:
    line = line.translate(str.maketrans('', '', string.punctuation)) #strip puncuation
    line = line.lower() #lowercase
    negword = line.split("\t")[1] #split by taps and removes ID
    negdict.extend(negword.split())
negfreqs = Counter(negdict)

#read in pos text file, read in line by line ignoring the ID at beginning of each sentence
pos = open("hotelPosT-train.txt").readlines()
posdict = []
for line in pos:
    line = line.translate(str.maketrans('', '', string.punctuation))
    line = line.lower()
    posword = line.split("\t")[1]
    posdict.extend(posword.split())
posfreqs = Counter(posdict)


#size of both pos and neg unigrams
negN = sum(negfreqs.values())
posN = sum(posfreqs.values())

#size of the vocab
vocab = len(posfreqs) + len(negfreqs)


#create probs associated with neg uni using Niave Bayes
negUni = {}
for word in negfreqs:
    negUni[word]= (negfreqs[word] + 1)/ (negN + vocab)


#create probs associated with pos uni using Niave Bayes
posUni = {}
for word in posfreqs:
    posUni[word] = (posfreqs[word] + 1)/ (posN + vocab)


logNeg = {}
for word in negfreqs:
    logNeg[word]= math.log((negfreqs[word] + 1)/ (negN + vocab))

logPos = {}
for word in posfreqs:
    logPos[word] = math.log((posfreqs[word] + 1)/ (posN + vocab))


def classification(text):
    neg = 1
    pos = 1
    text = text.split()
    for i in range(1, len(text)):
        if (text[i] in negUni):
            neg *= negUni[text[i]]
        else:
            neg *= 1 / (negN + vocab) 
        if (text[i] in posUni):
            pos *= posUni[text[i]]
        else:
            pos *= 1/ (posN + vocab)
#deal with underflow if prob of neg or pos very small
    if (pos <= 0 and neg <= 0):
        pos = 0
        neg = 0
        print("underflow: {}, {}". format(pos,neg))
        for i in range(1, len(text)):
            if (text[i] in logNeg):
                neg += logNeg[text[i]]
            else:
                neg += 1/ (negN + vocab) 
            if (text[i] in logPos):
                pos += logPos[text[i]]
            else:
                pos += 1/ (posN + vocab)
        pos = abs(pos)
        neg = abs(neg)
    print(pos, neg)
    if(neg > pos):
        return(" NEG")
    elif(pos > neg):
        return(" POS")
    else:
        return("Unclassified") 


#write the Classifications of text set to a file
def classFile(filename):
    f = open(filename)
    lines = f.readlines()
    r = open("ostlund-alison-assgn3-out.txt", 'w')
    text = []
    for line in lines:
        iD = line.split("\t")[0]
        line = line.split("\t")[1]
        line = line.lower()
        line = line.translate(str.maketrans('', '', string.punctuation))
        text = classification(line)
        r.write(str(iD) + str(text) + "\n")
    f.close()
    r.close()
    
classFile("HW3-testset.txt")

