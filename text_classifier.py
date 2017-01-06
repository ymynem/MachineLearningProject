import numpy as np
from sklearn import svm
from sklearn.svm import libsvm
from sklearn.datasets import fetch_20newsgroups

from ssk_by_Mona import normalize
from reuters import *


# build kernel matrix of string list s and string list t
def buildGramMat(sList, tList, l, n):
	lenS = len(sList)
	lenT = len(tList)

	gramMat = np.zeros((lenS, lenT), dtype=np.float64)
	print("Hi")
	for i in range(lenS):
		for j in range(lenT):
			if i == j:
				gramMat[i][j] = 1
				print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
			elif gramMat[j][i] != 0:
				gramMat[i][j] = gramMat[j][i]
				print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
			else:
				gramMat[i, j] = normalize(sList[i], tList[j], l, n)  # here to calculate the ssk value
				print ('normalize done!')
				print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))

	return gramMat


def train(data, label, l, n):
	svc = svm.SVC(kernel='precomputed')

	# build kernel matrix of training data
	gramMat = buildGramMat(data, data, l, n)
	# train support vector classification
	svc.fit(gramMat, label)

	return svc


# predict new dataset
def predict(svc, dataTest, dataTrain, l, n):
	gramMat = buildGramMat(dataTest, dataTrain, l, n)
	return svc.predict(gramMat)


def textClassify(cat1, cat2):
	#trainSize = 2
	#testSicze = 2

	n = 2
	l = 0.5

	'''
	news = fetch_20newsgroups(subset='train')
	trainX = news.data[:trainSize]
	trainY = news.target[:trainSize]
	'''

	# train the SVC
	a_train, a_test = get_documents(cat1)
	b_train, b_test = get_documents(cat2)

	print("Number of documents:", len(a_train), len(b_train))

	trainX = create_corpus(a_train + b_train)
	trainY = [cat1]*len(a_train) + [cat2]*len(b_train)

	print trainX[0]

	svc = train(trainX, trainY, l, n)

	# classify
	testX = create_corpus(a_test + b_test)
	testY = [cat1]*len(a_test) + [cat2]*len(b_train)

	return predict(svc, testX, trainX, l, n)     

# textClassify()
