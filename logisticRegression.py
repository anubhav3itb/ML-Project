from pyexcel_ods import get_data
import json
import ast
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
import nltk
import string
import numpy
import math
from sklearn import linear_model
from scipy import optimize

global_essay_list = []
global_marks_list = []
number_classes = 0

global_word_list = []
all_vector_list = []

#Removing @ words
def removeAt(text):
	a = text.split()
	b = []
	for each in a:
		if "@" not in each and "'" not in each and "/" not in each:
			b.append(each.lower())
	return " ".join(b)
	
#Removing Punctuations and Stop words
def cleanText(text):
	stop_words = set(stopwords.words("english"))
	word_set = word_tokenize(text)
	word_set = filter(lambda x: x not in string.punctuation, word_set)
	cleaned_text = filter(lambda x: x not in stop_words, word_set)
	clean_text = " ".join(str(x) for x in cleaned_text)
	return clean_text

#Created global marks and essay list
def readODS(filename):
	data = get_data(filename)
	a = json.dumps(data)
	a = ast.literal_eval(str(a))
	lists = a['Sheet1']
	
	for i in range(1, len(lists)):
		clean_essay = cleanText(removeAt(lists[i][2]))
		global_essay_list.append(clean_essay)
		global_marks_list.append(lists[i][6])
	
	global number_classes	
	number_classes = max(global_marks_list)

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

#Keeping only nouns, adjectives and verbs	
def keepImportant(sentence):
	stemmed_list = []
	text=nltk.word_tokenize(sentence)
	for each in text:
		stemmed_word = wordStemmingPorter(each)
		stemmed_list.append(stemmed_word)
	
	important_words = []
	pos_list = nltk.pos_tag(stemmed_list)
	for every in pos_list:
		if (every[1] == 'NN' or every[1] == 'JJ' or every[1] == 'VB' or every[1] == 'VBP' or every[1] == 'VBD'):
			important_words.append(every[0])
	
	return important_words 
	

#Generate global list of important words
def generateGlobalList(global_essay_list):
	for each in global_essay_list:
		imp_words = keepImportant(each)
		for every in imp_words:
			if every not in global_word_list:
				global_word_list.append(every)
	
	
#generate cost function vector of X's
def generateVector(sentence):
	temp_vector = []
	
	#inserting 1 in the beginning of each vector as X0
	temp_vector.append(1)
	
	
	imp_words = keepImportant(sentence)
	
	for each in global_word_list:
		if each in imp_words:
			temp_vector.append(1)
		else:
			temp_vector.append(0)
	return temp_vector


#Generate list of vectors
def generateAllVectors(global_essay_list):
	for each in global_essay_list:
		temp_vector = generateVector(each)
		all_vector_list.append(temp_vector)
		

def zerolistmaker(n):
	listofzeros = [0] * n
	return listofzeros

#Calculate sigmoid of a function. Takes an numpy.matrix as an input and return an array
def sigmoid(x):
	k = numpy.exp(numpy.multiply(x, -1))
	k = k + 1
	k = numpy.divide(1, k)
	return k
	

def lrCostFunction(theta, X, y, lambdas):
	m = len(y)
	#print X
	grad = zerolistmaker(len(theta))
	
	theta_matrix = numpy.matrix(theta)
	X_matrix = numpy.matrix(X)
	y_matrix = numpy.matrix(y)
	#print "theta shape ", theta_matrix.shape
	#print "X shape ",X_matrix.shape
	#print "y shape ", y_matrix.shape
	
	
	mult_matrix = sigmoid(X_matrix*theta_matrix.T)
	
	J = (y_matrix*numpy.log(mult_matrix)) + ((1-y_matrix) * numpy.log(1-mult_matrix))
	J = numpy.array(J)[0].tolist()[0]
	J = (J * -1)/m

	p = 0
	for i in range(1,len(theta)):
		p = p + math.pow(theta[i], 2)
	
	
	J = J + (lambdas/(2*m))*p
	return J

def gradientCalculation(theta, X, y, lambdas, num_iter,alpha):
	#print "mult ",mult_matrix.shape
	#print X_matrix[:, 0].shape
	m = len(y)
	#print X
	grad = zerolistmaker(len(theta))
	
	theta_matrix = numpy.matrix(theta)
	X_matrix = numpy.matrix(X)
	y_matrix = numpy.matrix(y)
	
	for j in range(0, num_iter):
		print "iteration ", j
		mult_matrix = sigmoid(X_matrix*theta_matrix.T)
	
		grad[0] = (numpy.array(((mult_matrix - y_matrix.T).T*X_matrix[:, 0]))[0].tolist()[0])/m
	
		for i in range(1, len(theta)):
			grad[i] = ((numpy.array(((mult_matrix - y_matrix.T).T*X_matrix[:, i]))[0].tolist()[0])/m) + ((lambdas/m)*theta[i])
		
		grad_matrix = numpy.matrix(grad)*alpha
		theta_matrix = theta_matrix - grad_matrix
		print lrCostFunction(theta_matrix, X, y, lambdas)
	
	return numpy.array(theta_matrix)[0].tolist()
		
		
	
	
	

def oneVsAll(X, y, num_labels, lambdas, initial_theta):
	y_matrix = numpy.matrix(y)
	final_matrix = []
	for i in range(0, num_labels):
		print "current class is, ", i
		a = gradientCalculation(initial_theta, X, y_matrix == i, lambdas, 200, 0.01)
		final_matrix.append(a)
	return final_matrix
	
def predictValue(final_theta, X):
	final_theta_matrix = numpy.matrix(final_theta)
	X_matrix = numpy.matrix(X)
	
	mult = sigmoid(X_matrix * final_theta_matrix.T)
	mult_list = numpy.array(mult)[0].tolist()
	#print mult_list
	return mult_list.index(max(mult_list))
	

def checkAccuracy(final_theta, essay_list, marks_list):
	count = 0
	for i in range(0, len(essay_list)):
		pred = predictValue(final_theta, generateVector(essay_list[i]))
		print "predicted value is: ", pred, " and actual value is ", marks_list[i]
		if pred == marks_list[i]:
			count = count + 1
	print "accurate: ",count
	print "total: ", len(marks_list)
	print "point to point accuracy is: ", (count/len(marks_list)*1.0)*100, "%"
	


def main():
	readODS("set1_ml.ods")
	print "Reading Complete"
	
	global_train_essay = global_essay_list[:1550]
	global_train_marks = global_marks_list[:1550]
	
	global_test_essay = global_essay_list[1550:]
	global_test_marks = global_marks_list[1550:]
	
	generateGlobalList(global_essay_list)
	print "Generated Global List with size, ", len(global_word_list)
	generateAllVectors(global_train_essay)
	print "Generated all Vectors"
	theta_vector = zerolistmaker(len(global_word_list)+1)
	
	
	
	global number_classes
	#print number_classes
	#theta = [0.25, 0.5, -0.5]
	
	#X = [[1.0000000, 2.2873553, 0.8908079],[1.0000000, 2.4717267,-0.6861101], [1.0000000, 0.3836040,-1.6322217], [1.0000000,-2.0572025,-1.0776761],[1.0000000,-2.6066264, 0.4676799],[1.0000000,-0.7595301, 1.5830532],[1.0000000, 1.7858747, 1.2429747],[1.0000000, 2.6893545,-0.2398890],[1.0000000, 1.1202542,-1.5021998],[1.0000000,-1.4788027,-1.3833951],[1.0000000,-2.7182552, 0.0072967],[1.0000000,-1.4585564, 1.3912800],[1.0000000, 1.1421324, 1.4961268],[1.0000000, 2.6927500, 0.2254416],[1.0000000, 1.7676656,-1.2525136],[1.0000000,-0.7826024,-1.5789136],[1.0000000,-2.6133493,-0.4536676],[1.0000000,-2.0413950, 1.0886782],[1.0000000, 0.4074085, 1.6300983],[1.0000000, 2.4816425, 0.6728136]]
	
	#y = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
	
	#lrCostFunction(theta, X, y, 0.1)
	#lrCostFunction(theta_vector, all_vector_list, global_marks_list, 0.1)
	final_theta = oneVsAll(all_vector_list, global_train_marks, number_classes, 0.1, theta_vector)
	print "Generated Final Theta"
	#print final_theta
	#print generateVector(global_test_essay[1])
	checkAccuracy(final_theta, global_test_essay, global_test_marks)
	
	
	

if __name__ == "__main__":
	main()

