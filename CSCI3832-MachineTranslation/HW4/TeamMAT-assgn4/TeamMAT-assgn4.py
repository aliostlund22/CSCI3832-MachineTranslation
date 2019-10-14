
# coding: utf-8

# CSCI3832 - HW4 NER <br>
# Matthew Donovan, Alison Ostlund, and Thadeus Labuszewski <br>
# Team MAT

# In[37]:


#Add clumn to test set to properly use load_conll
def add_dummy_column(file):
    newfile = open("testset.txt", "w")
    with open(file) as openfileobject:
        for line in openfileobject:
            if line == "\n":
                newfile.write("\n")
            else:
                newfile.write(line[:-1]+"\tD\n")
    newfile.close()

add_dummy_column("F18-assgn4-test.txt")


# In[38]:


from __future__ import print_function
import fileinput
import numpy as np
import sys
import math
from seqlearn.datasets import load_conll
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score

#Recognition features: 
#lowercase, uppercase, number, dash, previous word(i-1)/word(i-2) and next word(i+1)/word(i+2)
def features(sentence, i):
    line = sentence[i]
    word = line.split('\t')
    yield "word:{}" + str(word[1]).lower()

    
    if word[1].isupper():
        yield "CAP"
    if word[1].islower():
        yield "LOWER"
    if word[1].isnumeric():
        yield "NUM"
    if word[1] == '-':
        yield "DASH"  

    if i > 1:
        yield "word-1:{}" + str(sentence[i - 1].split("\t")[1]).lower()
        if word[-1].isupper():
            yield "PREV CAP"
        if word[-1] == '-':
            yield "DASH" 

        if i > 1:
            yield "word-2:{}" + str(sentence[i - 2].split("\t")[1]).lower()
            yield "word-2:{}" + str(sentence[i - 2].split("\t")[1]).upper()

    if i + 1 < len(sentence):
        yield "word+1:{}" + str(sentence[i + 1].split("\t")[1]).lower()
        if word[+1].isupper():
            yield " NEXT CAP"
        if word[+1] == '-':
            yield "DASH" 

        if i + 2 < len(sentence):
            yield "word+2:{}" + str(sentence[i + 2].split("\t")[1]).lower()
            yield "word+2:{}" + str(sentence[i + 2].split("\t")[1]).upper()


def describe(X, lengths):
#Function that gives us a rough idea of what our data looks like, number of sequences(senetneces) and tokens(words)
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))

    
def load_data():
    #Use this to load in our data so that we can pass it in to some machine learning algorithm
    #We return a training data set and a test data set
    print("Loading training data...", end=" ")
    train = load_conll(fileinput.input("gene-trainF18.txt"), features)
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)
    
    print("Loading test data...", end=" ")
    test = load_conll(fileinput.input("testset.txt"), features)
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)

    return train, test


if __name__ == "__main__":
    print(__doc__)
    #load our training and test data, seqlearn has a function load_conll that makes it easy for us
    train, test = load_data()
    X_train, y_train, lengths_train = train
    X_test, y_test, lengths_test = test

    #train a model
    #This implements the averaged structured perceptron algorithm of Collins and Daumé, 
    #with the addition of an adaptive learning rate.
    clf = StructuredPerceptron(verbose=True, max_iter=15)
    print("Training %s" % clf)
    clf.fit(X_train, y_train, lengths_train)
    
    #extract predicted IOB labels for our X_test
    y_pred = clf.predict(X_test, lengths_test)
   
    #write labels to our file so we can compare with golden standard
    f2 = open('results.txt', 'w')
    i = 0
    with open('testset.txt') as openfileobject:
        for line in openfileobject:
            if (line == '\n'):
                f2.write("\n")
            else:
                line=line.replace(line[len(line)-2:-1], str(y_pred[i]))
                f2.write(line)
                i = i+1       
    openfileobject.close()
    f2.close()
    
    print("Done...IOB predicted tagging written to 'results.txt' in your directory.")
    
    #print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
    #print("CoNLL F1: %.3f" % (100 * bio_f_score(y_test, y_pred)))

