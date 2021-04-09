"""
《统计学习》第二版
KNN-kd树
"""
import math

"""
函数说明：获取训练集
Parameters:
    None
Returns:
    dataSet: 训练数据集
"""
def getData():
    dataSet = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    return dataSet

"""
函数说明：沿切分坐标轴，按中位数把数据集分成三部分
Parameters:
    segAxis: 切分坐标轴
    dataSet: 数据集
Returns:
    midVal: 中位数元素
    leftSet: 小于中位数元素集合
    rightSet: 大于中位数元素集合
"""
def sepaDataSet(segAxis, dataSet):
        # 沿坐标轴segAxis排序
        dataSet = sorted(dataSet, key=lambda x : x[segAxis])
        # 中位数索引
        midIndex = len(dataSet) // 2
        # 分情况返回
        if len(dataSet) == 1:
            return dataSet[0], None, None
        elif len(dataSet) == 2:
            return dataSet[1], [dataSet[0]], None
        else:
            return dataSet[midIndex], dataSet[:midIndex], dataSet[midIndex+1:]

"""
类说明：kd树的节点类
Parameters:
    val: 节点值
    depth: 节点深度（根的深度是0）
"""
class Node():
    def __init__(self, val, depth):
        self.val = val
        self.depth = depth
        self.lchild = None
        self.rchild = None

"""
类说明：kd树
Parameters:
    None
"""
class KdTree():
    def __init__(self):
        self.root = None
    
    def createKdTree(self, dataSet):
        def recursion(depth, dataSet):
            if not dataSet:
                return None
            segAxis = depth % (len(dataSet[0]))
            midVal, leftSet, rightSet = sepaDataSet(segAxis, dataSet)
            node = Node(midVal, depth)
            node.lchild = recursion(depth+1, leftSet)
            node.rchild = recursion(depth+1, rightSet)
            return node
        self.root = recursion(0, dataSet)
        
    def nearestSearch(self, target):
        """最邻近搜索算法"""
        def recursion(node, target, depth, neaDist):
            if not node: # 递归搜索到达末端
                neaNode = Node([0]*len(target), depth) # 最近实例
                neaDist = float('inf')                 # 最近距离
                return neaNode, neaDist
            segAxis = depth % (len(target)) # 切分坐标轴
            if target[segAxis] <= node.val[segAxis]:
                oppNode = node.rchild
                neaNode, neaDist = recursion(node.lchild, target, depth+1, neaDist)
            else:
                oppNode = node.lchild
                neaNode, neaDist = recursion(node.rchild, target, depth+1, neaDist)
            # 如果当前实例更近，则更新最近实例和最近距离
            tmpDist = math.sqrt(sum((p1-p2)**2 for p1,p2 in zip(node.val, target)))
            if tmpDist < neaDist:
                neaDist = tmpDist
                neaNode = node
            # 判断当前实例的另一个子区域是否和圆心为target，半径为neaDist的超球体相交
            # 如果相交，要对另一个子节点最近邻搜索，否则返回上一个节点
            if abs(target[segAxis]-node.val[segAxis]) > neaDist:
                return neaNode, neaDist
            # 搜索另一个子区域
            tmpNode, tmpDist = recursion(oppNode, target, depth+1, neaDist)
            if tmpDist < neaDist:
                neaDist = tmpDist
                neaNode = tmpNode
            return neaNode, neaDist
        return recursion(self.root, target, 0, float('inf'))
                
    def preOrderTraverse(self):
        """先序遍历kdTree"""
        def recursion(node):
            if not node:
                return []
            return [node.val] + recursion(node.lchild) + recursion(node.rchild)
        res = recursion(self.root)
        return res

dataSet = getData()
k = KdTree()
k.createKdTree(dataSet)
neaNode, neaDist = k.nearestSearch([6, 8])