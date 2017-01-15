import numpy as np
from ssk import ssk, kh


# build kernel matrix of string list s and string list t with ssk
def build_gram_matrix(sList, tList, l, n, K=ssk):
    lenS = len(sList)
    lenT = len(tList)
    # optimize calculation of kernel gram matrix when sList equals to tList
    # in our case, this is to save calculation for kernel gram matrix of training data
    if sList is tList:  # sList is the same object as tList
        gramMat = np.eye(lenS, lenT, dtype=np.float64)

        for i in range(lenS):
            for j in range(i+1, lenT):
                gramMat[i][j] = gramMat[j][i] = K(sList[i], tList[j], n, l)
            # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
            # without the two list equal to one another, we have to calculate every element for gram matrix
            # in our case, this is for kernel gram matrix of training data and test data
    else:
        gramMat = np.zeros((lenS, lenT), dtype=np.float64)
        for i in range(lenS):
            print("Starting line", i)
            for j in range(lenT):
                gramMat[i][j] = K(sList[i], tList[j], n, l)  # here to calculate the ssk value

                # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))

    return gramMat


