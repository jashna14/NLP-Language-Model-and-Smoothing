import sys
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

def unigrams_get(f):
	unigrams = {}
	for line in f:
		if line.strip():
			line = re.sub('[^a-zA-Z ]+', "",line)
			line = re.sub(' +', ' ',line)
			line = line.lower()
			words = nltk.word_tokenize(line)
			for x in words:
				if x not in unigrams:
					unigrams[x] = 1
				else:
					unigrams[x] += 1

	return unigrams					

def trigrams_get(f):
	trigrams = {}
	for line in f:
		if line.strip():
			line = re.sub('[^a-zA-Z ]+', "",line)
			line = re.sub(' +', ' ',line)
			line = line.lower()
			words = nltk.word_tokenize(line)
			length = len(words)
			length = length - 2 
			for x in range(length):
				if words[x] not in trigrams:
					trigrams[words[x]] = {}
				if words[x+1] not in trigrams[words[x]]:
					trigrams[words[x]][words[x+1]] = {}
				if words[x+2] not in trigrams[words[x]][words[x+1]]:
					trigrams[words[x]][words[x+1]][words[x+2]] = 1
				else:
					trigrams[words[x]][words[x+1]][words[x+2]] += 1

	return trigrams


def bigrams_get(f):
	bigrams = {}
	for line in f:
		if line.strip():
			line = re.sub('[^a-zA-Z ]+', "",line)
			line = re.sub(' +', ' ',line)
			line = line.lower()
			words = nltk.word_tokenize(line)
			length = len(words)
			length = length - 1 
			for x in range(length):
				if words[x] not in bigrams:
					bigrams[words[x]] = {}
				if words[x+1] not in bigrams[words[x]]:
					bigrams[words[x]][words[x+1]] = 1
				else:
					bigrams[words[x]][words[x+1]] += 1

	return bigrams

def bigram_k(unigrams , bigrams , sentence):
	d = 5
	prob = 0

	if sentence.strip():
		prob = 1
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		length = length - 1
		for x in range(length):
			porb1 = max(bigrams[words[x]][words[x+1]] - d , 0)/unigrams[words[x]]
			lam = (float(d)/float(sum(bigrams[words[x]].values())))*len(bigrams[words[x]])
			cnt = 0
			for i in bigrams:
				if words[x+1] in bigrams[i]:
					cnt += 1
			cnt1 = 0
			for i in bigrams:
				cnt1 += len(bigrams[i])		
			pcont = float(cnt)/float(cnt1)
			porb1 += lam*pcont
			prob *= porb1

	return prob	 	




model_type = sys.argv[1]
smoothing_type = sys.argv[2]
corpus = sys.argv[3]
f = open(corpus, "r")
f1 =  open(corpus, "r")
f2 =  open(corpus, "r")

unigrams = unigrams_get(f)
bigrams = bigrams_get(f1)
trigrams = trigrams_get(f2)

print(unigrams)
print()
print()
print(bigrams)
cnt1 =0
for i in bigrams:
	cnt1 += len(bigrams[i])
print(cnt1)	
print(float(5)/float(70))

sen = input("Input Sentence :")

if model_type == '2' and smoothing_type == 'k':
	prob = bigram_k(unigrams , bigrams , sen)
	print(prob)