from nltk.corpus import wordnet as wn

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
	
string = "dog cat"
best_list = getAllSynonyms(string)
print getSynonymCount(best_list, "chase caterpillar")
