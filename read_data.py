#!/usr/bin/python

import numpy as np
import scipy
import json
import os 
import glob

path="data/small/"

class READ():
	def __init__(self,window_size):
		self.w={}  	# word and their vertes id
		self.wcount={}
		self.listoffreq=[]
		self.d=[]	# document and their vertex id
		self.l={}	# class label and their vertex id
		self.w2w={}	#word to word mapping with edge weight
		self.w2d={}	#word to document mapping with edge weight
		self.w2l={}	#word to class mapping with edge weight
		self.ndocs=0	#no of document
		self.nlabel=0	#no of labels
		self.window_size=window_size;
		self.nedge=0
		self.nvertex=0

	def generate_graphs(self):
		files = []
		labels = []
		for filename in glob.glob(os.path.join(path, '*.txt')):
			files.append(filename)
			if 'pos' in filename:
				labels.append('pos')
			elif 'neg' in filename:
				labels.append('neg')
        
		document=1
		nodes=1
		label=1
		for x in labels:
			if x not in self.l:
				self.l[x]=label
				label+=1
		self.nlabel = label

		for filename in files:
			fd=open(filename)
			for x in fd:
				x = x.strip()
				x=x.split()
				if x[-1][-1:]=='\n':
					x[-1]=x[-1][:-1]
				self.d.append((x,labels[files.index(filename)],document))
				document+=1
				for y in x:
					if y not in self.w:
						self.w[y]=nodes
						self.wcount[nodes]=1
						nodes+=1
					else:
						self.wcount[self.w[y]]+=1
		self.ndocs = document
		self.nvertex = nodes-1

		i=1
		# print self.nvertex
		while i <=self.nvertex:
			self.listoffreq.append(self.wcount[i])
			i= i+1
		
		for x in self.d:
			index=0
			len1=len(x[0])
			label=self.l[x[1]]
			for y in x[0]:
				word=self.w[y] #word is vertex id of "word and document" graph
				left=index-self.window_size/2
				right=index+self.window_size/2
				if left<0:
					left=0
				if right>len1:
					right=len1-1

				# word to word dictionary
				for z in xrange(left,index):
					a=self.w[x[0][z]]
					# print a,word
					if a==word:
						continue
					if word not in self.w2w:
						self.w2w[word]={}
					b=self.w2w[word]
					if a not in b:
						b[a]=1
					else:
						b[a]=b[a]+1
				
				for z in xrange(index+1,right):
					# print x[0][z],y,z
					a=self.w[x[0][z]]
					if a==word:
						continue
					if word not in self.w2w:
						self.w2w[word]={}
					b=self.w2w[word]
					if a not in b:
						b[a]=1
					else:
						b[a]=b[a]+1

				# document to word  dictionary
				if x[2] not in self.w2d:
					self.w2d[x[2]]={}
				if word not in self.w2d[x[2]]:
					self.w2d[x[2]][word]=0
				self.w2d[x[2]][word]=self.w2d[x[2]][word]+1				
				
				# word to label dictionary
				if label not in self.w2l:
					self.w2l[label]={}
				if word not in self.w2l[label]:
					self.w2l[label][word]=0
				self.w2l[label][word] = self.w2l[label][word]+1
				index+=1

		json.dump(self.w, open('word_mapping.json', 'wb'))
  #       json.dump(self.l, open('label_mapping.json', 'wb'))
  #       json.dump(self.d, open('document_mapping.json', 'wb'))
  #       print 'w2l', len(self.w2l.keys())
  #       print 'w2d', len(self.w2d.keys())
  #       print 'w2w', len(self.w2w.keys())

	def gen_edgeprob(self):
		p = []	#probability of edge b/w w1-w2
		v1 = []	#w1
		v2 = [] #w2
		pd = [] #probability of edge b/w w-d
		v3 = [] #d
		v4 = []	#w
		pl = [] #probability of edge b/w w-l
		v5 = [] #l
		v6 = [] #w
		for k in self.w2w.keys():
			for kj in self.w2w[k].keys():
				p.append(self.w2w[k][kj])#all co-occurences added
				v1.append(k)
				v2.append(kj)
				self.nedge += 1
		p = np.asarray(p, dtype=np.float64)
		nP=np.asarray(self.listoffreq, dtype=np.float64)
		nP = np.power(nP, (3.0/4.0))
		p = p / float(sum(p))
		nP = nP / float(sum(nP))

		for doc in self.w2d.keys():
			for word in self.w2d[doc].keys():
				pd.append(self.w2d[doc][word])#frequency of that word in the doc
				v3.append(doc)
				v4.append(word)
		pd = np.asarray(pd, dtype=np.float64)
		pd = pd / float(sum(pd))

		for label in self.w2l.keys():
			for word in self.w2l[label].keys():
				pl.append(self.w2l[label][word])#frequency of that word in that label
				v5.append(label)
				v6.append(word)
		pl = np.asarray(pl, dtype=np.float64)
		pl = pl / float(sum(pl))
		
		return p, nP, v1, v2, pd, v3, v4, pl, v5, v6
				
	def test(self):
		fd1=open("w.txt","w")
		fd2=open("d.txt","w")
		fd3=open("l.txt","w")
		fd4=open("w2w.txt","w")
		fd5=open("w2d.txt","w")
		fd6=open("w2l.txt","w")
		for key in self.w:
			fd1.write("%s     %s\n" % (key, self.w[key]))
		for key in self.d:
			for x in key[0]:
				fd2.write(x+" ")
			fd2.write(str(key[2])+"\n")
		for key in self.l:
			fd3.write("%s     %s\n" % (key, self.l[key]))
		for key in self.w2w:
			fd4.write(str(key)+"\n")
			for x in self.w2w[key]:
				fd4.write("            "+ str(x)+"           "+ str(self.w2w[key][x]) + "\n")
		for key in self.w2d:
			fd5.write(str(key)+"\n")
			for x in self.w2d[key]:
				fd5.write("            "+ str(x)+"           "+ str(self.w2d[key][x]) + "\n")
		
		for key in self.w2l:
			fd6.write(str(key)+"\n")
			for x in self.w2l[key]:
				fd6.write("            "+ str(x)+"           "+ str(self.w2l[key][x]) + "\n")
