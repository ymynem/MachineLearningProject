import numpy as np
from ssk_by_Mona import normalize
from sklearn import svm
from reuters import *
from subset_creator import *
import cProfile
import time


def Sorting(corpus):
    """
    n = length of string/word needed for subset
    x = how many most frequently occuring n length words
    inputfile = dataset used for algorithm
    
    Returns a list of x most frequently occuring n length caracters in a file
    """
    # f=open(inputfile,'r')
    # S=corpus.lower()
    mapping = [('.', ''), (',', ''), ('?', ''), ('!', ''), ('%', ''), ('\n', ''), ('\t', '')]
    for corp in corpus:
        S = corp.lower()
        for i, j in mapping:
            S = S.replace(i, j)
    return S


# build kernel matrix of string list s and string list t with ssk
def buildGramMat(sList, tList, l, n):
    # slist = train
    # tlist = test
    lenS = len(sList)
    lenT = len(tList)
    gramMat = np.zeros((lenS, lenT), dtype=np.float64)
    gramMat[:] = -1  # fyller med -1or

    # optimize calculation of kernel gram matrix when sList equals to tList
    # in our case, this is to save calculation for kernel gram matrix of training data
    if sList is tList:  # sList is the same object as tList
        for i in range(lenS):
            for j in range(lenT):
                if i == j:
                    gramMat[i][j] = 1
                    #print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
                elif gramMat[i][j] == -1:
                    gramMat[i][j] = gramMat[j][i] = normalize(sList[i], tList[j], l,
                                                              n)  # here to calculate the ssk value
                    #print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
                    # without the two list equal to one another, we have to calculate every element for gram matrix
                # in our case, this is for kernel gram matrix of training data and test data
    else:
        for i in range(lenS):
            for j in range(lenT):
                gramMat[i][j] = normalize(sList[i], tList[j], l, n)  # here to calculate the ssk value
                # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))

    return gramMat


# train svc with kernel gram matrix of training data
def train(data, label, l, n):
    # build kernel matrix of training data
    gramMat = buildGramMat(data, data, l, n)

    # train support vector classification (svc) with above gram matrix built
    svc = svm.SVC(kernel='precomputed')
    """"A column-vector y was passed when a 1d array was"
                          " expected. Please change the shape of y to "
                          "(n_samples, ), for example using ravel().","""
    svc.fit(gramMat, label)  # np.ravel(label)
    return svc


# predict new dataset
def predict(svc, dataTest, dataTrain, l, n):
    # build kernel gram matrix of training data and test data
    gramMat = buildGramMat(dataTest, dataTrain, l,
                           n)  # G[i][j] = K(a[i],b[j]) a,b är list of documents, our inputs to the ssk kernel
    return svc.predict(gramMat)


# retrieve and clean reuters data, wrap up method of train and predict
def textClassify(cat1, cat2):
    n = 2
    l = 0.5
    word_frqu = 3

    # train the SVC using reuters datasets
    a_train, a_test = get_documents(cat1)
    b_train, b_test = get_documents(cat2)
    testX = create_corpus(a_test + b_test)
    trainX = create_corpus(a_train + b_train)

    trainX_most_cm, testX_most_cm = most_common(n, word_frqu, trainX, testX)

    trainY = [cat1] * len(a_train) + [cat2] * len(b_train)  # lista av orden acq och en lista av "corn"
    testY = [cat1] * len(a_test) + [cat2] * len(b_train)  # skapar testdata
    svc = train(trainX_most_cm, trainY, l, n)

    return predict(svc, testX_most_cm, trainX_most_cm, l, n)


if __name__ == "__main__":
    # reuters.download()

    textClassify('acq', 'corn')
    print(textClassify('acq', 'corn'))
    print("DONE")
    Profile.run('textClassify()')
