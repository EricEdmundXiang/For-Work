import numpy as np

def getDataSet():
    dataSet = np.array([
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
        ])
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], dtype=dataSet.dtype)
    N = len(dataSet)
    return dataSet, labels, N

def calcPriPro(labels):
    priPro = {}
    for label in labels:
        if label not in priPro.keys():
            priPro[label] = 0
        priPro[label] += 1
    return priPro

def calcConProb(dataSet, labels, priPro):
    # 统计各类下训练数据集特征向量各维取值集合
    catDataSet = np.concatenate((dataSet, labels.reshape(-1,1)), axis=1)
    conProb = {}
    for sample in catDataSet:
        for i in range(len(dataSet[0])):
            key = sample[i] + sample[-1]
            if key not in conProb.keys():
                conProb[key] = 0
            conProb[key] += 1
    # 计算条件概率
    for key, num in conProb.items():
        conProb[key] = num / priPro[key[1]]
    return conProb

def infer(priPro, conProb, dataSet, N, x):
    inferCls = None
    MAP = 0
    for cls in priPro.keys():
        tmpMAP = priPro[cls] / N
        for i in range(len(dataSet[0])):
            key = x[i] + cls
            tmpMAP *= conProb[key]
        if tmpMAP > MAP:
            inferCls = cls
            MAP = tmpMAP
    return inferCls

dataSet, labels, N = getDataSet()
priPro = calcPriPro(labels)
conProb = calcConProb(dataSet, labels, priPro)
print(infer(priPro, conProb, dataSet, N, ['2', 'S']))