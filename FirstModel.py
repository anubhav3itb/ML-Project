import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
import nltk
import string
import numpy
import math
from sklearn import linear_model
	
training_essay = []
training_score = []
test_essay = []
test_score = []

global_word_list = []

all_vector_list = []
	
#Reading CSV and dividing data into test set and training set
def readCSV(filename, essay_set, min_max):

	f = open(filename, 'rb')
	reader = csv.reader(f)
	for row in reader:
		if(int(row[1]) == essay_set) and (int(row[2])) >= min_max:
			training_essay.append(row[4])
			training_score.append(int(row[2]))
		elif(int(row[1]) == essay_set) and (int(row[2])) <= min_max:
			test_essay.append(row[4])
			test_score.append(int(row[2]))
			
	f.close()

#Stemming and Lemmatization using Snowball stemmer
def wordStemmingSnowball(word):
	stemmer = SnowballStemmer("english")
	stem = str(stemmer.stem(word))
	return stem

#Stemming and Lemmatization using Porter stemmer
def wordStemmingPorter(word):
	stemmer = PorterStemmer()
	stem = str(stemmer.stem(word))
	return stem

#Removing Punctuations and Stop words
def cleanText(training_essay):
	stop_words = set(stopwords.words("english"))
	
	clean_training_essays = []
	for each in training_essay:
		word_set = word_tokenize(each)
		word_set = filter(lambda x: x not in string.punctuation, word_set)
		cleaned_text = filter(lambda x: x not in stop_words, word_set)
		clean_training_essays.append(" ".join(str(x) for x in cleaned_text))
	return clean_training_essays



#Keeping only nouns, adjectives and verbs	
def keepImportant(sentence):
	stemmed_list = []
	text=nltk.word_tokenize(sentence)
	for each in text:
		stemmed_word = wordStemmingSnowball(each)
		stemmed_list.append(stemmed_word)
	
	important_words = []
	pos_list = nltk.pos_tag(stemmed_list)
	for every in pos_list:
		if (every[1] == 'NN' or every[1] == 'JJ' or every[1] == 'VB' or every[1] == 'VBP' or every[1] == 'VBD'):
			important_words.append(every[0])
	
	return important_words 
	

#Generate global list of important words
def generateGlobalList(clean_training_essays):
	for each in clean_training_essays:
		imp_words = keepImportant(each)
		for every in imp_words:
			if every not in global_word_list:
				global_word_list.append(every)
	

#generate cost function vector of X's
def generateVector(sentence):
	temp_vector = []
	
	#inserting 1 in the beginning of each vector as X0
	#temp_vector.append(1)
	
	
	imp_words = keepImportant(sentence)
	
	for each in global_word_list:
		if each in imp_words:
			temp_vector.append(1)
		else:
			temp_vector.append(0)
	return temp_vector
	

#Generate list of vectors
def generateAllVectors(clean_training_essays):
	for each in clean_training_essays:
		temp_vector = generateVector(each)
		all_vector_list.append(temp_vector)
			

#Calculating theta matrix using Normal Equation => theta(matrix) = (Xt*X)-1*Xt*y and applied Regularization 
def calculateThetas(lambdas):

	
	X = numpy.matrix(all_vector_list)
	print "Dimension of X ",X.shape 
	Xt = X.T
	print "Dimension of X transpose ",Xt.shape
	XtX = Xt*X
	print "Dimension of X transpose * X ",XtX.shape
	print XtX
	
	
	regular_array = numpy.identity(XtX.shape[0])
	regular_array[0][0] = 0
	
	regular_array = lambdas*regular_array
	print "Dimension of regular",regular_array.shape
	#print regular_array
	
	
	XtXminusRegular = XtX - regular_array
	inv_XtXminusRegular = numpy.linalg.inv(XtXminusRegular) 
	tsm = numpy.matrix(training_score).T
	Xty = Xt*tsm
	thetas = inv_XtXminusRegular*Xty
	return thetas

def useScikit(sentence):
	X = numpy.array(all_vector_list)
	#print "Dimension of X ",X.shape
	#print training_score
	tsm = numpy.array(training_score)
	#print "Dimension of X transpose tsm ",tsm.shape
	clf = linear_model.SGDClassifier()
	clf.fit(X, tsm)
	test_vector = generateVector(sentence)
	test_matrix = numpy.array(test_vector)
	return (clf.predict(test_matrix))
	
	
def calculateScoreScikit(sentence):
	test_vector = generateVector(sentence)
	test_matrix = numpy.matrix(test_vector).T
	print(clf.predict(test_matrix))


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

#Cost Function
def computeCost(X,y,theta_vector):
	X_matrix = numpy.matrix(X)
	#print "Dimension of X ",X_matrix.shape 
	y_matrix = numpy.matrix(y).T
	#print "Dimension of Y ",y_matrix.shape 
	theta_vector_matrix = numpy.matrix(theta_vector).T
	#print "Dimension of theta_vector ",theta_vector_matrix.T.shape 

	sigma = X_matrix*theta_vector_matrix - y_matrix
	sigma_square = numpy.square(sigma)
	sigma_square_sum = sum((sigma_square.T).tolist()[0])
	return sigma_square_sum/(2*len(y))
	
def gradientDescent(X, y, theta_vector, alpha, num_iter):
	J_History = zerolistmaker(num_iter)	
	
	#theta_vector_matrix = numpy.matrix(theta_vector).T
	for i in range(0, num_iter):
		theta_vector_matrix = numpy.matrix(theta_vector).T
		print "current iteration = ", i
		#print "current iteration = ", i, " with cost ",computeCost(X, y, theta_vector)
		#theta "print: ",theta_vector
		X_matrix = numpy.matrix(all_vector_list)
		y_matrix = numpy.matrix(training_score).T
		sigma = X_matrix*theta_vector_matrix - y_matrix
		for k in range(0, len(theta_vector)):
		
			
			sigma_new = sigma.T.tolist()[0]
		
			for j in range(0, len(y)):

				sigma_new[j] = sigma_new[j] * X[j][k]

			sigma_sum = sum(sigma_new)
		
			theta_vector[k] = theta_vector[k]-((alpha/len(y))*sigma_sum)
			#print "theta_vector[",k,'] ',theta_vector[k], ' sigma sum: ', sigma_sum
				

def calculateScore(sentence, theta_vectors):
	test_vector = generateVector(sentence)
	test_matrix = numpy.matrix(test_vector).T
	
	theta_matrix = numpy.matrix(theta_vectors)
	
	sigma = theta_matrix*test_matrix
	value = sum(sigma.T.tolist()[0])
	return math.floor(value)

def calculateAccuracy(essay_list, theta_vectors):
	total_sum = 0
	for i in range(len(essay_list)):
		value = calculateScore(essay_list[i], theta_vectors)
		if value < 0.0:
			value = 0.0
		print "current example ", i, " of", len(essay_list), " with original score ", test_score[i], " and predicted score ", value
		if(value == test_score[i]):
			total_sum = total_sum + 1
	
	print "total",total_sum
	print "Accuracy is: ",(total_sum/(len(essay_list)*1.0))*100.0

readCSV('training.csv', 4, 1	)
ab = cleanText(training_essay)
generateGlobalList(ab)
generateAllVectors(ab)
theta_vector = zerolistmaker(len(global_word_list)+1) 
clean_test_essays = cleanText(test_essay)


print "predicted score ", useScikit(clean_test_essays[54])
print "final score ",test_score[54]
#newThetas = calculateThetas(0.001)
#newThetas[0] = 0.7
#print newThetas
#calculateScore(clean_test_essays[0], newThetas.T.tolist())
#print computeCost(all_vector_list, training_score, theta_vector)
#gradientDescent(all_vector_list, training_score, theta_vector, 0.0001, 200)

#print clean_test_essays[0]

#calculateAccuracy(clean_test_essays, newThetas.T.tolist())
