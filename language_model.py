import sys
import re
import nltk
import math
from nltk.tokenize import sent_tokenize, word_tokenize

def unigrams_get(f):
	unigrams = {}
	for line in f:
		if line.strip():
			line = re.sub('-+', ' ',line)
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
			line = re.sub('-+', ' ',line)
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
			line = re.sub('-+', ' ',line)
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

def unigram_k(unigrams , sentence):
	d = 0.5
	prob = 0

	if sentence.strip():
		# prob = 1
		sentence = re.sub('-+', ' ',sentence)
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		for x in range(length):
			cnt1 = 0
			cnt2 = 0
			if words[x] in unigrams:
				cnt1 = unigrams[words[x]]
			else:
				cnt1 =0	
			for i in unigrams:
				cnt2 += unigrams[i]
			prob1 = float(max(cnt1 - d ,0))/float(cnt2)
			cnt1 = 0
			for i in unigrams:
				cnt1 += 1

			cnt2 = 0
			for i in unigrams:
				cnt2 += unigrams[i] 	
			lam = (float(d)/float(cnt2))*cnt1
			V = cnt1
			prob1 = prob1 + lam/V
			prob += math.log(prob1)

	return prob

def bigram_k(unigrams , bigrams , sentence):
	d = 0.75
	prob = 0

	if sentence.strip():
		# prob = 1
		sentence = re.sub('-+', ' ',sentence)
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		length = length - 1

		for x in range(length):
			if words[x] in unigrams:
				if words[x+1] in bigrams[words[x]]:
					prob1 = float(max(bigrams[words[x]][words[x+1]] - d , 0))/float(unigrams[words[x]])
				else:
					prob1 = 0	
				lam = (float(d)/float(unigrams[words[x]]))*len(bigrams[words[x]])
				cnt1 = 0
				for i in bigrams:
					if words[x+1] in bigrams[i]:
						cnt1 += 1

				cnt2 = 0
				for i in bigrams:
					cnt2 += len(bigrams[i])		
				pkn = float(max(cnt1-d,0))/float(cnt2)

				cnt1 = 0
				for i in unigrams:
					cnt1 += 1

				cnt2 = 0
				for i in unigrams:
					cnt2 += unigrams[i] 	
				lam1 = (float(d)/float(cnt2))*cnt1
				V = cnt1

				pkn = pkn + float(lam1)/float(V)
				prob1 = prob1 + lam*pkn
			
			else:
				lam = (float(d)/float(sum(unigrams.values())))*len(unigrams)
				cnt = 0
				for i in bigrams:
					cnt += len(bigrams[i])
				prob1 = lam/cnt 	

			prob += math.log(prob1)

	return prob	

def trigram_k(unigrams , bigrams , trigrams , sentence):
	d = 7
	prob = 0

	if sentence.strip():
		# prob = 1
		sentence = re.sub('-+', ' ',sentence)
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		length = length - 2

		for x in range(length):
			if words[x] in unigrams and words[x+1] in unigrams and words[x+1] in bigrams[words[x]]:
				if words[x+2] in trigrams[words[x]][words[x+1]]:
					prob1 = max(trigrams[words[x]][words[x+1]][words[x+2]] - d , 0)/bigrams[words[x]][words[x+1]]
				else:
					prob1 = 0	
				lam1 = (float(d)/float(bigrams[words[x]][words[x+1]]))*len(trigrams[words[x]][words[x+1]])
				
				cnt1 = 0
				cnt2 = 0

				for i in trigrams:
					if words[x+1] in trigrams[i]:
						if words[x+2] in trigrams[i][words[x+1]]:
							cnt1 += 1

				for i in trigrams:
					if words[x+1] in trigrams[i]:
						cnt2 += len(trigrams[i][words[x+1]])

				pkn = float(max(cnt1 - d , 0))/float(cnt2)
				lam2 = (float(d)/float(unigrams[words[x+1]]))*len(bigrams[words[x+1]])
				cnt1 = 0
				for i in bigrams:
					if words[x+2] in bigrams[i]:
						cnt1 += 1

				cnt2 = 0
				for i in bigrams:
					cnt2 += len(bigrams[i])		
				pkn1 = float(max(cnt1-d,0))/float(cnt2)

				cnt1 = 0
				for i in unigrams:
					cnt1 += 1

				cnt2 = 0
				for i in unigrams:
					cnt2 += unigrams[i] 	
				lam3 = (float(d)/float(cnt2))*cnt1
				V = cnt1

				pkn1 = pkn1 + float(lam3)/float(V)
				pkn = pkn + lam2*pkn1
				prob1 = prob1 + lam1*pkn

			else:
				lam = (float(d)/float(sum(unigrams.values())))*len(unigrams)
				cnt = 0
				for i in trigrams:
					for j in trigrams[i]:
						cnt += len(trigrams[i][j])
				prob1 = lam/cnt 	

			prob += math.log(prob1)
			

	return prob	 	 	



def unigram_w(unigrams,sentence):
	if sentence.strip():
		prob = 0
		sentence = re.sub('-+', ' ',sentence)
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		for x in range(length):
			if words[x] in unigrams:
				prob1 = unigrams[words[x]]/(sum(unigrams.values())+ len(unigrams))
			else:
				lam = (float(0.5)/float(sum(unigrams.values())))*len(unigrams)
				prob1 = lam/len(unigrams)

			prob = prob + math.log(prob1)

	return prob				


def bigram_w(unigrams , bigrams , sentence):
	prob = 0

	if sentence.strip():
		# prob = 1
		sentence = re.sub('-+', ' ',sentence)
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		length = length - 1

		for x in range(length):
			if words[x] in unigrams and words[x+1] in unigrams:
				cnt1 = 0
				cnt2 = 0
				cnt1 = len(bigrams[words[x]])
				cnt2 = unigrams[words[x]]
				lam = 1 - (cnt1/(cnt1+cnt2))
				prob1 = 0
				if words[x+1] in bigrams[words[x]]:
					prob1 = bigrams[words[x]][words[x+1]]/(unigrams[words[x]] + len(bigrams[words[x]]))
				else:
					cnt1 = len(unigrams) - len(bigrams[words[x]])
					prob1 = len(bigrams[words[x]])/(cnt1*(unigrams[words[x]] + len(bigrams[words[x]])))		
				prob1 = prob1*lam + (1 - lam)*(unigrams[words[x+1]]/(sum(unigrams.values()) + len(unigrams)))
			else:
				lam = (float(0.75)/float(sum(unigrams.values())))*len(unigrams)
				cnt = 0
				for i in bigrams:
					cnt += len(bigrams[i])
				prob1 = lam/cnt

			prob = prob + math.log(prob1)

	return prob		


def trigram_w(unigrams , bigrams , trigrams, sentence):
	prob = 0

	if sentence.strip():
		# prob = 1
		sentence = re.sub('-+', ' ',sentence)
		sentence = re.sub('[^a-zA-Z ]+', "",sentence)
		sentence = re.sub(' +', ' ',sentence)
		sentence = sentence.lower()
		words = nltk.word_tokenize(sentence)
		length = len(words)
		length = length - 2

		for x in range(length):
			if words[x] in unigrams and words[x+1] in unigrams and words[x+1] in bigrams[words[x]]:	
				cnt1 = 0
				cnt2 = 0
				cnt1 = len(trigrams[words[x]][words[x+1]])
				cnt2 = bigrams[words[x]][words[x+1]]
				lam = 1 - (cnt1/(cnt1+cnt2))
				prob1 = 0
				if words[x+2] in trigrams[words[x]][words[x+1]]: 
					prob1 = trigrams[words[x]][words[x+1]][words[x+2]]/(bigrams[words[x]][words[x+1]] + len(trigrams[words[x]][words[x+1]]))
				else:
					cnt1 = len(unigrams) - len(trigrams[words[x]][words[x+1]])
					prob1 = len(trigrams[words[x]][words[x+1]])/(cnt1*(bigrams[words[x]][words[x+1]] + len(trigrams[words[x]][words[x+1]])))
				
				new_sen = words[x+1] + " " + words[x+2]
				prob1 = prob1*lam + (1 - lam)*math.exp(bigram_w(unigrams , bigrams , new_sen))

			else:
				lam = (float(7)/float(sum(unigrams.values())))*len(unigrams)
				cnt = 0
				for i in trigrams:
					for j in trigrams[i]:
						cnt += len(trigrams[i][j])
				prob1 = lam/cnt

			prob = prob + math.log(prob1)

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


while 1:
	sen = input("Input Sentence :")

	if model_type == '2' and smoothing_type == 'k':
		prob = bigram_k(unigrams , bigrams , sen)
		print(math.exp(prob))

	if model_type == '3' and smoothing_type == 'k':
		prob = trigram_k(unigrams , bigrams , trigrams , sen)
		print(math.exp(prob))

	if model_type == '1' and smoothing_type == 'k':
		prob = unigram_k(unigrams , sen)
		print(math.exp(prob))

	if model_type == '2' and smoothing_type == 'w':
		prob = bigram_w(unigrams , bigrams , sen)
		print(math.exp(prob))

	if model_type == '3' and smoothing_type == 'w':
		prob = trigram_w(unigrams , bigrams , trigrams , sen)
		print(math.exp(prob))

	if model_type == '1' and smoothing_type == 'w':
		prob = unigram_w(unigrams , sen)
		print(math.exp(prob))

	# prob = unigram_k(unigrams , sen)
	# print(math.exp(prob))	

	# prob = bigram_k(unigrams , bigrams , sen)
	# print(math.exp(prob))	

	# prob = trigram_k(unigrams ,bigrams ,trigrams ,sen)
	# print(math.exp(prob))
	
	# prob = unigram_w(unigrams , sen)
	# print(math.exp(prob))	

	# prob = bigram_w(unigrams , bigrams , sen)
	# print(math.exp(prob))	

	# prob = trigram_w(unigrams , bigrams , trigrams ,sen)
	# print(math.exp(prob))
		