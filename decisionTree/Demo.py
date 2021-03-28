# -*- coding: utf-8 -*-
from math import log

"""
函数说明：创建测试数据集
Parameters: 无
Returns:
    dataSet: 数据集
    labels: 分类属性
    
"""
def createDataSet():
    # 数据集
    dataSet=[[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return dataSet, labels

"""
函数说明：计算给定数据集的经验熵
Parameters:
    dataSet: 数据集
Returns:
    shannonEnt: 经验熵
"""
def calcShannonEnt(dataSet):
    labelCounts = {} # 各类出现次数的字典
    for featVec in dataSet:
        if featVec[-1] not in labelCounts.keys():
            labelCounts[featVec[-1]] = 0
        labelCounts[featVec[-1]] += 1
    
    shannonEnt = 0.
    numEntries = len(dataSet)
    for key in labelCounts.keys():
        prob = float(labelCounts[key] / numEntries)
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt

"""
函数说明：计算给定数据集的经验信息增益
Parameters:
    dataSet: 数据集
Return:
    bestFeature: 最大信息增益特征的索引值
    
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 数据集的样本数
    baseInfoGain = calcShannonEnt(dataSet) # 信息增益初始值
    bestInfoGain = 0.
    bestFeature = -1
    for i in range(numFeatures): # 对于每一个特征
        featList = [example[i] for example in dataSet] # 当前特征下所有分类
        uniqueVals = set(featList)                  # 去除重复元素
        conGain = 0.                                # 条件增益
        for feat in uniqueVals: # 针对每一个An
            subDataSet = splitDataSet(dataSet, i, feat)
            prob = len(subDataSet) / float(len(dataSet))
            conGain += prob * calcShannonEnt(subDataSet)
        infoGain = baseInfoGain - conGain
        # 打印当前特征信息
        print('Feature {}, info gain {:.2f}'.format(i, infoGain))
        # 比较当前特征的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

"""
函数说明：按给定特征划分数据集
Parameters:
    dataSet: 数据集
    i: 特征索引
    feat: 划分类别
Returns:
    subDataSet: 数据子集
    
"""
def splitDataSet(dataSet, i, feat):
    subDataSet = []
    for featVec in dataSet:
        if featVec[i] == feat:
            subDataSet.append(featVec)
    return subDataSet

# main函数
if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(calcShannonEnt(dataSet))
    bestFeature = chooseBestFeatureToSplit(dataSet)
    print('The best feature index: {}'.format(bestFeature))