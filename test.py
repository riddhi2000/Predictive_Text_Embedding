#!/usr/bin/python

import numpy as np
import string
import os
import sys
import re
import json
import random
from sklearn import svm
from sklearn.cross_validation import train_test_split

class test_pte(object):

	def __init__(self):

		self.train_documents = []
		self.train_labels = []
		self.test_documents = []
		self.test_actual_labels = []
		self.test_predicted_labels = []
		self.all_words = json.load(open('word_mapping.json'))
		self.word_embedding = np.load('lookupW.npy')
		self.D = 40

	def load_data(self):
		train_set = []
		test_set = []
		document_no = 1
		files = ['./data/small/train-pos.txt','./data/small/train-neg.txt']
		class_labels = ['pos','neg']
		index = 0
		for file_name in files:
		    fp = open(file_name)
		    lines = fp.readlines()
		    for line in lines:
		    	line = line.strip()
		        words = line.split(" ")
		        document = (words,class_labels[index],document_no)
		        train_set.append(document)
		        document_no += 1
		    index += 1

		files = ['./data/small/test-pos.txt','./data/small/test-neg.txt']
		class_labels = ['pos','neg']
		index = 0
		for file_name in files:
		    fp = open(file_name)
		    lines = fp.readlines()
		    for line in lines:
		    	line = line.strip()
		        words = line.split(" ")
		        document = (words,class_labels[index],document_no)
		        test_set.append(document)
		        document_no += 1
		    index += 1
		return train_set, test_set

	def test(self):

		train_set,test_set = self.load_data()
		for index in range(len(train_set)):
			document_sum = np.zeros(self.D)
			for word_index in range(len(train_set[index][0])):
				word = train_set[index][0][word_index]
				i = self.all_words[word]
				embedding = self.word_embedding[i-1]
				document_sum = np.add(document_sum, embedding)
			document_average = np.divide(document_sum, len(train_set[index][0]))
			self.train_documents.append(document_average)
			self.train_labels.append(train_set[index][1])
		for index in range(len(test_set)):
			document_sum = np.zeros(self.D)
			for word_index in range(len(test_set[index][0])):
				word = test_set[index][0][word_index]
				i = self.all_words[word]
				embedding = self.word_embedding[i-1]
				document_sum = np.add(document_sum, embedding)
			document_average = np.divide(document_sum, len(test_set[index][0]))
			self.test_documents.append(document_average)
			self.test_actual_labels.append(train_set[index][1])
		clf = svm.SVC()
		clf.fit(self.train_documents, self.train_labels)
		self.test_predicted_labels = clf.predict(self.test_documents)
		correct = 0
		for i in range(len(self.test_predicted_labels)):
			if self.test_predicted_labels[i] == self.test_actual_labels[i]:
				correct = correct + 1
		accuracy = (correct)/float(len(self.test_predicted_labels)) * 100.0
		print 'Accuracy : ',accuracy


if __name__ == "__main__":
    pte = test_pte()
    pte.test()
