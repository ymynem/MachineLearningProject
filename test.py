import numpy as np
from ssk_by_Mona import normalize
from sklearn import svm
from reuters import *
from subset_creator import *
import cProfile
import random


# build kernel matrix of string list s and string list t with ssk
def buildGramMat(sList, tList, l, n):
    print("buildGramMat()")
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
                    # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
                elif gramMat[i][j] == -1:
                    gramMat[i][j] = gramMat[j][i] = normalize(sList[i], tList[j], l,
                                                              n)  # here to calculate the ssk value
                    # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
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
    print("train()")
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
    print("predict()")
    # build kernel gram matrix of training data and test data
    gramMat = buildGramMat(dataTest, dataTrain, l,
                           n)  # G[i][j] = K(a[i],b[j]) a,b är list of documents, our inputs to the ssk kernel
    return svc.predict(gramMat)


def partition(dataset, fraction):
    breakPoint = int(len(dataset) * fraction)
    random.shuffle(dataset)
    return dataset[:breakPoint], dataset[breakPoint:]


# retrieve and clean reuters data, wrap up method of train and predict
def textClassify(cat1, cat2):
    n = 3
    l = 0.5
    word_frqu = 3
    len_of_word = 6
    fraction = 1/30

    # train the SVC using reuters datasets
    a_train, a_test = get_documents(cat1)
    b_train, b_test = get_documents(cat2)
    a_train_first_part, second = partition(a_train, 1/40)
    #print("a_train", len(a_train_first_part))
    a_test_first_part, sec_p = partition(a_test, 1/30)
    #print("a_test", len(a_test_first_part))
    b_train_first_part, se_p = partition(b_train, 1/6)
    #print("b_train", len(b_train_first_part))
    b_test_first_part, s_p = partition(b_test, 1/3)
    #print("b_test", len(b_test_first_part))

    testX = create_corpus(a_test_first_part + b_test_first_part)
    trainX = create_corpus(a_train_first_part + b_train_first_part)

    trainX_most_cm, testX_most_cm = most_common(len_of_word, word_frqu, trainX, testX)

    trainY = [cat1] * len(a_train_first_part) + [cat2] * len(b_train_first_part)  # lista av orden acq och en lista av "corn"
    testY = [cat1] * len(a_test_first_part) + [cat2] * len(b_test_first_part)  # skapar testdata
    svc = train(trainX_most_cm, trainY, l, n)
    #print("testY ", testY)
    return predict(svc, testX_most_cm, trainX_most_cm, l, n)



if __name__ == "__main__":
    # reuters.download()

    #textClassify('acq', 'corn')
    print(textClassify('acq', 'corn'))
    print("DONE")
    #cProfile.run('textClassify()')
