import numpy as np
from sklearn import svm
from sklearn.svm import libsvm
from sklearn.datasets import fetch_20newsgroups

from ssk_by_Mona import normalize


# build kernel matrix of string list s and string list t
def buildGramMat(sList, tList, l, n):
	lenS = len(sList)
	lenT = len(tList)

	gramMat = np.zeros((lenS, lenT), dtype=np.float64)

	for i in range(lenS):
		for j in range(lenT):
			gramMat[i, j] = normalize(sList[i], tList[j], l, n)  # here to calculate the ssk value
			print 'normalize done!'

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


def textClassify():
	trainSize = 2
	testSicze = 2

	n = 2
	l = 0.5

	news = fetch_20newsgroups(subset='train')
	trainX = news.data[:trainSize]
	trainY = news.target[:trainSize]

	svc = train(trainX, trainY, l, n)
	return predict(svc, news.data[trainSize:trainSize+testSize], trainX, l, n)     

