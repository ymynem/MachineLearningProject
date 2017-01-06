import numpy as np
from sklearn import svm

from ssk_by_Mona import normalize
from reuters import *


# build kernel matrix of string list s and string list t
def buildGramMat(sList, tList, l, n):
	lenS = len(sList)
	lenT = len(tList)

	gramMat = np.zeros((lenS, lenT), dtype=np.float64)
	gramMat[:] = -1

	# optimize calculation of kernel gram matrix when sList equals to tList
	if sList is tList:
		print("Hi")
		for i in range(lenS):
			for j in range(lenT):
				if i == j:
					gramMat[i][j] = 1
					print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
				elif gramMat[i][j] == -1:
					gramMat[i][j] = gramMat[j][i] = normalize(sList[i], tList[j], l, n)  # here to calculate the ssk value
					print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
	else:
		for i in range(lenS):
			for j in range(lenT):
				gramMat[i][j] = normalize(sList[i], tList[j], l, n)  # here to calculate the ssk value
				print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))

	return gramMat


def train(data, label, l, n):
	# build kernel matrix of training data
	gramMat = buildGramMat(data, data, l, n)

	# train support vector classification
	svc = svm.SVC(kernel='precomputed')
	svc.fit(gramMat, label)
	
	return svc


# predict new dataset
def predict(svc, dataTest, dataTrain, l, n):
	gramMat = buildGramMat(dataTest, dataTrain, l, n)
	return svc.predict(gramMat)


def textClassify(cat1, cat2):
	n = 2
	l = 0.5

	# train the SVC using reuters datasets
	a_train, a_test = get_documents(cat1)
	b_train, b_test = get_documents(cat2)

	print("Number of documents:", len(a_train), len(b_train))

	trainX = create_corpus(a_train + b_train)
	trainY = [cat1]*len(a_train) + [cat2]*len(b_train)

	#print trainX[0]   # print the fisrt training text

	svc = train(trainX, trainY, l, n)

	# classify test data
	testX = create_corpus(a_test + b_test)
	testY = [cat1]*len(a_test) + [cat2]*len(b_train)

	return predict(svc, testX, trainX, l, n)     

# textClassify()
