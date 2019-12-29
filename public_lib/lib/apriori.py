#!/home/dmer/.pyenv/versions/env3/bin/python
# -*- coding: utf-8 -*-

'''
---------------------------------------------------------------------------
File Name: apriori.py
Description: . 

Created by: 'Gamer Think'
Changed by: zhangzhiyong
Changed Activity: First coding on 2018/4/27
---------------------------------------------------------------------------
'''

from pydoc import apropos
import copy

#=========================     准备函数 ==========================================
#加载数据集
def loadDataSet():
    # dataSet = [['豆奶','莴苣'],
    #     ['莴苣','尿布','葡萄酒','甜菜'],
    #     ['豆奶','尿布','葡萄酒','橙汁'],
    #     ['莴苣','豆奶','尿布','葡萄酒'],
    #     ['莴苣','豆奶','尿布','橙汁']]
    
    dataSet = [['BILLINGEVENT_CLICK', 'a2668', 'b693'],
               ['BILLINGEVENT_CLICK', 'a150', 'b671'],
               ['BILLINGEVENT_IMPRESSION', 'a2668', 'b693'],
               ['BILLINGEVENT_CLICK', 'a2668', 'b671'],
               ['BILLINGEVENT_IMPRESSION', 'a2668', 'b671'],
               ['BILLINGEVENT_CLICK', 'a120', 'b671']]
    return dataSet

def createC1(dataSet):
    C1 = []   #C1为大小为1的项的集合
    for transaction in dataSet:  #遍历数据集中的每一条交易
        for item in transaction: #遍历每一条交易中的每个商品
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #map函数表示遍历C1中的每一个元素执行forzenset，frozenset表示“冰冻”的集合，即不可改变
    return map(frozenset,C1)

#Ck表示数据集，D表示候选集合的列表，minSupport表示最小支持度
#该函数用于从C1生成L1，L1表示满足最低支持度的元素集合
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        # print('tid: ', tid)
        for can in Ck:
            #issubset：表示如果集合can中的每一元素都在tid中则返回true  
            if can.issubset(tid):
                # print('can: ', can)
                #统计各个集合scan出现的次数，存入ssCnt字典中，字典的key是集合，value是统计出现的次数
                #if not ssCnt.has_key(can): # python3 can not support
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        #计算每个项集的支持度，如果满足条件则把该项集加入到retList列表中
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
            supportData[key] = support   #构建支持的项集的字典
    return retList, supportData

#======================  Apriori算法  =================================
#Create Ck,CaprioriGen ()的输人参数为频繁项集列表Lk与项集元素个数k，输出为Ck
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            #前k-2项相同时合并两个集合
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
            
    return retList

def apriori(dataSet, minSupport=0.5):
    C1 = list(createC1(dataSet))  #创建C1
    D = list(map(set, dataSet)) #python3
    #C1: [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    #D: [set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]
    # print(D)
    L1,supportData = scanD(D, C1, minSupport)
    L = [L1]
    #若两个项集的长度为k - 1,则必须前k-2项相同才可连接，即求并集，所以[:k-2]的实际作用为取列表的前k-1个元素
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk,supK = scanD(D,Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k +=1
    print('L: ', L)
    print('supportData: ', supportData)
    return L,supportData

#========================   关联规则生成函数     ========================
#调用下边两个函数
#L：表示频繁项集列表，supportData：包含那些频繁项集支持数据的字典，minConf：表示最小可信度阀值
def generateRules(L, supportData,minConf = 0.7):
    bigRuleList = [] #存放可信度，后面可以根据可信度排名
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i>1):
                #如果项集的元素数目超过2，则使用下面的函数对他进行下一步的合并，合并函数如下
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                #如果项集中只有两个元素，则使用下面的函数计算可信度
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)    
            
    return bigRuleList

#第一次修改，出现丢失的那几个关联规则
def generateRules2(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                # 三个及以上元素的集合
                H1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 两个元素的集合
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

#第二次修改，简化函数，和第一步修改结果相同
def generateRules3(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            rulesFromConseq2(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
 
def rulesFromConseq2(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > m): # 判断长度改为 > m，这时即可以求H的可信度
        Hmpl = calcConf(freqSet, H, supportData, brl, minConf)
        if (len(Hmpl) > 1): # 判断求完可信度后是否还有可信度大于阈值的项用来生成下一层H
            Hmpl = aprioriGen(Hmpl, m + 1)
            rulesFromConseq2(freqSet, Hmpl, supportData, brl, minConf) # 递归计算，不变

#第三次修改       消除rulesFromConseq2()函数中的递归项，去掉了多余的Hmpl变量，运行结果和上面相同
def rulesFromConseq3(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    while (len(freqSet) > m): # 判断长度 > m，这时即可求H的可信度
        H = calcConf(freqSet, H, supportData, brl, minConf)
        if (len(H) > 1): # 判断求完可信度后是否还有可信度大于阈值的项用来生成下一层H
            H = aprioriGen(H, m + 1)
            m += 1
        else: # 不能继续生成下一层候选关联规则，提前退出循环
            break

#计算规则的可信度，并找到满足最小可信度的规则存放在prunedH中，作为返回值返回
def calcConf(freqSet,H,supportData, br1, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet - conseq]
      
        if conf>= minConf:
            print(freqSet-conseq,"-->",conseq ,"conf:",conf)
            br1.append((freqSet-conseq,conseq,conf))  #填充可信度列表
            prunedH.append(conseq)    #保存满足最小置信度的规则
    return prunedH

#从最初的项集中产生更多的关联规则，H为当前的候选规则集，产生下一层的候选规则集
#freqSet：频繁项集 H：可以出现在规则右部的元素列表  supportData：保存项集的支持度，brl保存生成的关联规则，minConf：最小可信度阀值
def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) >(m +1)):
        Hmp1 = aprioriGen( H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)          
        if (len(Hmp1) >1):
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)
        
        
if __name__=="__main__":
    dataSet = loadDataSet()
    print(dataSet)
    L,suppData = apriori(dataSet, minSupport=0.1)
    i = 0
    for one in L:
        print("项数为 %s 的频繁项集：" % (i + 1), one,"\n")
        i +=1
        
    # print("generateRules：\nminConf=0.7时：")
    # rules = generateRules(L,suppData, minConf=0.1)
    # print("\nminConf=0.5时：")
    # rules = generateRules(L,suppData, minConf=0.1)
    
    # print("generateRules2：\nminConf=0.7时：")
    # rules = generateRules2(L,suppData, minConf=0.7)
    # print("minConf=0.5时：")
    # rules = generateRules2(L,suppData, minConf=0.5)
    
    
    print("generateRules3：\nminConf=0.7时：")
    rules = generateRules3(L,suppData, minConf=0.1)
    print("minConf=0.5时：")
    rules = generateRules3(L,suppData, minConf=0.1)