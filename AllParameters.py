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
from sklearn import linear_model, datasets
from nltk.corpus import wordnet as wn
from scipy import optimize
import enchant

global_essay_list = []
global_marks_list = []
global_number_words = []
global_number_sent = []
global_spelling_list = []
global_NN = []
global_JJ = []
global_VBPD = []
global_synonym_list = []

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


def removeSpellingMistakes(sent):
	d = enchant.Dict("en_US")
	sp = sent.split()
	count = 0
	for each in sp:
		if d.check(each) == False:
			count = count + 1
	return count


def getPOSCount(sent, types):
	current = []
	stemmed_list = nltk.word_tokenize(sent)
	pos_list = nltk.pos_tag(stemmed_list)

	for each in types:
		for every in pos_list:
			if (every[1] == each):
				current.append(every[0])
	return len(current)

#Created global marks and essay list
def readODS(filename):
	data = get_data(filename)
	a = json.dumps(data)
	a = ast.literal_eval(str(a))
	lists = a['Sheet1']

	for i in range(1, len(lists)):
		#print i
		num_words = len(lists[i][2].split())
		num_sent = len(lists[i][2].split('.'))
		global_number_words.append(num_words)
		global_number_sent.append(num_sent)


		clean_essay = cleanText(removeAt(lists[i][2]))
		num_spell = removeSpellingMistakes(clean_essay)
		global_spelling_list.append(num_spell)

		global_essay_list.append(clean_essay)
		global_marks_list.append(lists[i][6])

		global_NN.append(getPOSCount(lists[i][2], ['NN']))
		global_JJ.append(getPOSCount(lists[i][2], ['JJ']))
		global_VBPD.append(getPOSCount(lists[i][2], ['VB','VBP','VBD']))


	number_classes = max(global_marks_list)
	return len(lists), number_classes


def NormalizeList(lists):
	mx = max(lists)*1.0
	mn = min(lists)*1.0
	ag = reduce(lambda x, y: x + y, lists) / len(lists)
	temp = []
	for i in range(len(lists)):
		temp.append((lists[i]-ag)/(mx-mn))
	return temp


def generateX(sent_list, word_list, spell_list, NN_list, JJ_list, VBPD_list, synonym_list):
	X = []
	for i in range(len(sent_list)):
		temp = []
		temp.append(1)
		temp.append(sent_list[i])
		temp.append(word_list[i])
		temp.append(spell_list[i])
		temp.append(NN_list[i])
		temp.append(JJ_list[i])
		temp.append(VBPD_list[i])
		temp.append(synonym_list[i])

		X.append(temp)
	return X
	
def getSynonymList(word):
	lists = wn.synsets(word)
	outs = []
	for each in lists:
		outs.append(str(each.name()).split(".")[0])
	outs.append(word)
	return list(set(outs))
	

def getAllSynonyms(string):
	lists = string.split()
	outs = []
	for each in lists:
		outs = outs + getSynonymList(each)
		
	return outs

def getSynonymCount(best_list, string):
	lists = string.split()
	count = 0
	
	for each in lists:
		if each in best_list:
			count = count + 1
	return count

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
		#print "iteration ", j
		mult_matrix = sigmoid(X_matrix*theta_matrix.T)

		grad[0] = (numpy.array(((mult_matrix - y_matrix.T).T*X_matrix[:, 0]))[0].tolist()[0])/m

		for i in range(1, len(theta)):
			grad[i] = ((numpy.array(((mult_matrix - y_matrix.T).T*X_matrix[:, i]))[0].tolist()[0])/m) + ((lambdas/m)*theta[i])

		grad_matrix = numpy.matrix(grad)*alpha
		theta_matrix = theta_matrix - grad_matrix
		#print lrCostFunction(theta_matrix, X, y, lambdas)

	return numpy.array(theta_matrix)[0].tolist()

def oneVsAll(X, y, num_labels, lambdas, initial_theta):
	y_matrix = numpy.matrix(y)
	final_matrix = []
	for i in range(0, num_labels+1):
		#print "current class is, ", i
		a = gradientCalculation(initial_theta, X, y_matrix == i, lambdas, 1500, 0.01)
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
		pred = predictValue(final_theta, essay_list[i])
		print "predicted value is: ", pred, " and actual value is ", marks_list[i]
		if pred == marks_list[i]:
			count = count + 1
	print "accurate: ",count
	print "total: ", len(marks_list)
	print "point to point accuracy is: ", (count/len(marks_list)*1.0)*100, "%"

def scikitLearn(X,Y, X_test, Y_test, final_theta):
	logreg = linear_model.LogisticRegression(C=1e5)
	X_matrix = numpy.matrix(X)
	y_matrix = numpy.matrix(Y)
	logreg.fit(X, Y)
	count_scikit = 0
	count_lr = 0
	for i in range(0,len(X_test)):
		Z = logreg.predict(numpy.matrix(X_test[i]))
		predict_scikit = Z[0]

		if(predict_scikit == Y_test[i]):
			count_scikit = count_scikit + 1

		predict_lr = predictValue(final_theta, X_test[i])
		if(predict_lr == Y_test[i]):
			count_lr = count_lr + 1
		#print "Predict lr = ", predict_lr, " predict_scikit = ",predict_scikit, ". Actual = ", Y_test[i]

	print "Total training examples: ", len(X)
	print "Total test examples: ", len(X_test)
	print "total correct scikit: ", count_scikit
	print "total correct lr: ", count_lr


def main():
	for i in range(1,7):
		print "###########################################"
		global global_spelling_list
		global global_number_words
		global global_number_sent
		global global_VBPD
		global global_JJ
		global global_NN
		global global_essay_list
		global global_marks_list
		global global_synonym_list
	
		global_spelling_list = []
		global_number_words = []
		global_number_sent = []
		global_VBPD = []
		global_JJ = []
		global_NN = []
		global_essay_list = []
		global_marks_list = []
		global_synonym_list = []
	
	
	
	
		print "Current ods is ", i
		print "Reading started"
		#len_lists,number_classes = readODS("test.ods")
		len_lists,number_classes = readODS("set"+str(i)+".ods")
		input_param = (len_lists*85)/100
	
		print "reading Complete"
	
		##############################Populating global synonym List####################################
		m = max(global_marks_list)
		#print m
		max_list = [i for i, j in enumerate(global_marks_list) if j == m]
		best_essay = ""
		for each in max_list:
			best_essay = best_essay + global_essay_list[each]
	
		list_all_synonyms = getAllSynonyms(best_essay)
		#print list_all_synonyms
		for each in global_essay_list:
			 count = getSynonymCount(list_all_synonyms, each)
			 #print count
			 global_synonym_list.append(count)
	


		Normalized_NN = NormalizeList(global_NN)
		Normalized_JJ = NormalizeList(global_JJ)
		Normalized_VBPD = NormalizeList(global_VBPD)
		Normalized_sent = NormalizeList(global_number_sent)
		Normalized_words = NormalizeList(global_number_words)
		Normalized_spelling = NormalizeList(global_spelling_list)
		Normalized_synonym = NormalizeList(global_synonym_list)
	
		#print "Normalization Complete"
	

		X =  generateX(Normalized_sent, Normalized_words, Normalized_spelling, Normalized_NN, Normalized_JJ, Normalized_VBPD, Normalized_synonym)
		#print "size of X:",len(X)
		X_train = X[:input_param]
		X_test = X[input_param:]

		y_train = global_marks_list[:input_param]
		y_test = global_marks_list[input_param:]
	
		#print "train size: ",len(X_train)
		#print "test_ size: ",len(X_test)

		initial_theta = zerolistmaker(len(X[0]))
		final_theta = oneVsAll(X_train, y_train, number_classes, 0.01, initial_theta)
		#print final_theta
		#checkAccuracy(final_theta, X_test, y_test)
		#print "Model Running ..."
		scikitLearn(X_train, y_train, X_test,y_test ,final_theta)
		print "Current Model Complete"

if __name__ == "__main__":
	main()
