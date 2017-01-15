#import numpy as np
#from sklearn import svm, metrics
#from reuters import *
from subset_creator import *
import cProfile
import time
import random
from ssk import ssk, kh


# build kernel matrix of string list s and string list t with ssk
def buildGramMat(sList, tList, l, n):
    # slist = train
    # tlist = test
    lenS = len(sList)
    lenT = len(tList)
    # optimize calculation of kernel gram matrix when sList equals to tList
    # in our case, this is to save calculation for kernel gram matrix of training data
    if sList is tList:  # sList is the same object as tList
        k = 1
        #gramMat = np.eye(lenS, lenT, dtype=np.float64)
        gramMat = []
        for i in range(lenS):
            gramMat.append([1]*lenT)

        sames = []
        for i in range(lenS):
            sames.append(kh(sList[i], sList[i], n, l))
        samet = sames

        for i in range(lenS):
            print("Starting line", i)
            for j in range(k, lenT):
                gramMat[i][j] = gramMat[j][i] = ssk(sList[i], tList[j], n, l, ss=sames[i], tt=samet[j])
            k += 1  # here to calculate the ssk value
            # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
            # without the two list equal to one another, we have to calculate every element for gram matrix
            # in our case, this is for kernel gram matrix of training data and test data
    else:
#        gramMat = np.zeros((lenS, lenT), dtype=np.float64)
        gramMat = []
        for i in range(lenS):
            gramMat.append([1]*lenT)

        sames = []
        for i in range(lenS):
            sames.append(kh(sList[i], sList[i], n, l))
        samet = []
        for i in range(lenT):
            samet.append(kh(tList[i], tList[i], n, l))

        for i in range(lenS):
            print("Starting line", i)
            for j in range(lenT):
                gramMat[i][j] = ssk(sList[i], tList[j], n, l, ss=sames[i], tt=samet[j])  # here to calculate the ssk value

                # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))

    return gramMat


# train svc with kernel gram matrix of training data
def train(data, label, l, n):
    # build kernel matrix of training data
    gramMat = buildGramMat(data, data, l, n)

    # train support vector classification (svc) with above gram matrix built
    svc = svm.SVC(kernel='precomputed')
    svc.fit(gramMat, label)  # np.ravel(label)
    return svc


# predict new dataset
def predict(svc, dataTest, dataTrain, l, n):
    # build kernel gram matrix of training data and test data
    gramMat = buildGramMat(dataTest, dataTrain, l,
                           n)  # G[i][j] = K(a[i],b[j]) a,b är list of documents, our inputs to the ssk kernel
    return svc.predict(gramMat)


def partition(dataset, fraction):
    breakPoint = int(len(dataset) * fraction)
    random.shuffle(dataset)
    return dataset[:breakPoint], dataset[breakPoint:]


# retrieve and clean reuters data, wrap up method of train and predict
def textClassify(cat1, cat2, n, l):
    word_frqu = 3
    len_of_word = 5

    # train the SVC using reuters datasets
    a_train, a_test = get_documents(cat1)
    b_train, b_test = get_documents(cat2)
    a_train_first_part, second = partition(a_train, 1 / 40)
    # print("a_train", len(a_train_first_part))
    a_test_first_part, sec_p = partition(a_test, 1 / 30)
    # print("a_test", len(a_test_first_part))
    b_train_first_part, se_p = partition(b_train, 1 / 6)
    # print("b_train", len(b_train_first_part))
    b_test_first_part, s_p = partition(b_test, 1 / 3)
    # print("b_test", len(b_test_first_part))

    testX = create_corpus(a_test_first_part + b_test_first_part)
    trainX = create_corpus(a_train_first_part + b_train_first_part)

    trainX_most_cm, testX_most_cm = most_common(len_of_word, word_frqu, trainX, testX)

    trainY = [cat1] * len(a_train_first_part) + [cat2] * len(
        b_train_first_part)  # lista av orden acq och en lista av "corn"
    testY = [cat1] * len(a_test_first_part) + [cat2] * len(b_test_first_part)  # skapar testdata
    svc = train(trainX_most_cm, trainY, l, n)
    pr = predict(svc, testX_most_cm, trainX_most_cm, l, n)

    total_correct = len([1 for p in zip(pr, testY) if p[0] == p[1]])
    print("Results: {}/{} - {:.2f}%".format(total_correct, len(pr), 100 * total_correct / len(pr)))

    precision, recall, f1_score = calculate_table_values(testY, pr)
    write_to_file(precision, recall, f1_score, " ")
    return pr


def calculate_table_values(testY, pr):
    precision = metrics.precision_score(testY, pr, average='macro')
    recall = metrics.recall_score(testY, pr, average='macro')
    f1_score = metrics.f1_score(testY, pr, average='macro')
    return precision, recall, f1_score


def write_to_file(f1_score, precision, recall, optional):
    f = open('resultFile.txt', 'a+')
    f.write(str(f1_score) + " " + str(precision) + " " + str(recall) + str(
        optional) + "\n")  # python will convert \n to os.linesep
    f.close()


if __name__ == "__main__":
    # reuters.download()

    # textClassify()
    iter = 0
    n = 2
    l = 0.5
    cat1 = 'acq'
    cat2 = 'corn'

    write_to_file("Subseqlength = " + str(n), " Lambda = " + str(l), " Category1 = " + cat1, " Category2 = " + cat2)

    while (iter < 2):
        textClassify(cat1, cat2, n, l)
        iter = iter + 1
        # cProfile.run('textClassify("acq", "corn")')
