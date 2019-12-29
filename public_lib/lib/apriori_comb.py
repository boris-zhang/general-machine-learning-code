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
import datetime
import numpy as np
import itertools

#=========================  准备函数 ===================================
#加载数据集
def loadDataSet():
    # dataSet = [['BILLINGEVENT_CLICK', '2668', '693', 'AA1', 'B1'],
    #            ['BILLINGEVENT_CLICK', '150', '671', 'AA2', 'B2'],
    #            ['BILLINGEVENT_IMPRESSION', '2668', '693', 'AA3', 'B3'],
    #            ['BILLINGEVENT_CLICK', '2668', '671', 'AA4', 'B4'],
    #            ['BILLINGEVENT_IMPRESSION', '2668', '671', 'AA5', 'B5'],
    #            ['BILLINGEVENT_CLICK', '120', '671', 'AA6', 'B6']]
    dataSet = [['BILLINGEVENT_CLICK', 'a2668', 'b693'],
               ['BILLINGEVENT_CLICK', 'a150', 'b671'],
               ['BILLINGEVENT_IMPRESSION', 'a2668', 'b693'],
               ['BILLINGEVENT_CLICK', 'a2668', 'b671'],
               ['BILLINGEVENT_IMPRESSION', 'a2668', 'b671'],
               ['BILLINGEVENT_CLICK', 'a120', 'b671']]
    return dataSet

def createC1(D):
    C1 = []   #C1为大小为1的项的集合
    for tid in D:  #遍历数据集中的每一条交易
        while None in tid:
            tid.remove(None)
        for item in tid: #遍历每一条交易中的每个商品
            if not [item] in C1:
                C1.append([item])
    #map函数表示遍历C1中的每一个元素执行forzenset，frozenset表示“冰冻”的集合，即不可改变
    return map(frozenset,C1)


# D：交易数据
# C1：独立项目集，该函数生成满足最低支持度的元素集合Lk
# TN: 汇总报表中的交易次数
def scanD1(D, C1, minSupport, TN):
    ssCnt = {}
    i = 0
    for tid in D:
        while None in tid:
            tid.remove(None)
        for can in C1:
            #issubset：表示如果集合can中的每一元素都在tid中则返回true  
            if can.issubset(set(tid)):
                #统计各个集合scan出现的次数，存入ssCnt字典中，字典的key是集合，value是统计出现的次数
                if not can in ssCnt:
                    ssCnt[can] = TN[i]
                else:
                    ssCnt[can] +=TN[i]
        i +=1

    sumTrans = np.sum(TN)    # 本次查询的总交易数
    retList = []
    supportData = {}
    for key in ssCnt:
        # 计算每个项集的支持度，如果满足条件则把该项集加入到retList列表中
        support = ssCnt[key] / sumTrans
        # print('key, ssCnt, sumTrans: %s, %d, %d' % (key, ssCnt[key], sumTrans))
        if support >= minSupport:
           retList.append(key)
           supportData[key] = support   #构建支持的项集的字典
    return retList, supportData

def scanDk(D, k, minSupport, TN):
    ssCnt = {}
    tni = 0
    for tid in D:
        while None in tid:
            tid.remove(None)

        # 最后一列为ctr_ratio_bin，考虑只关注该字段的蕴含关系，
        # 如果不为ctr_h, ctr_hh, ctr_ll, ctr_l则跳过
        epos = len(tid)
        # if tid[epos-1] != 'ctr_hh': # 起始为0
        #     tni +=1
        #     continue

        # 对tid中的元素进行两两组合，统计各个集合scan出现的次数，
        # 存入ssCnt字典中，字典的key是集合，value是统计出现的次数
        gap = k - 1     # 项集中两项的间距，如2项项集为1，3项集为2
        for i in range(0, epos - gap):
            for j in range(i + gap, epos):
                itpair = set([tid[i], tid[j]])
                subpair = list(itertools.combinations(tid[i+1: j], k-2))    # [tuple1, tuple2]
                # print('itpair: ', itpair)
                # print('subpair: ', subpair)
                for itmid in subpair:
                    if itmid:   # 非空
                        itend = copy.deepcopy(itpair) | set(list(itmid))    
                        item = frozenset(itend)    # set的key必须是不可变数值
                    else:
                        item = frozenset(itpair)
                    # print('item: ', item)
                    if not item in ssCnt:  # 判断set中是否存在某元素的效率远高于list
                        # print('TN[tni]: %d, %d' % (tni, TN[tni]))
                        ssCnt[item] = TN[tni]
                    else:
                        # print('TN[tni]: %d, %d' % (tni, TN[tni]))
                        ssCnt[item] +=TN[tni]
        tni +=1
        
    sumTrans = np.sum(TN)    # 本次查询的总交易数
    retList = []
    supportData = {}    # 注意是字典而非集合
    for key in ssCnt:
        #计算每个项集的支持度，如果满足条件则把该项集加入到retList列表中
        support = ssCnt[key] / sumTrans
        # print('key, ssCnt, sumTrans: %s, %d, %d' % (key, ssCnt[key], sumTrans))
        if support >= minSupport:
            retList.append(key)           
            supportData[key] = support   #构建支持的项集的字典
            
    return retList, supportData


# ======================  Apriori算法  =================================
def apriori(D, TN, k, minSupport=0.5):
    # 先找到单元素频繁项集
    C1 = list(createC1(D))   # 前n列为组合字段
    if k == 1:
        L, supportData = scanD1(D, C1, minSupport, TN)
    elif k > 1:
        # 两两组合找到频繁k项集
        L, supportData = scanDk(D, k, minSupport, TN)
    else:
        print("error k: ", k)
        return -1
    # print('L: ', L)
    # print('\nsupportData: ', supportData)
    return L, supportData

#========================   关联规则生成函数     ========================
#   L：只包含一条频繁项集的列表，最后一项为优化目标项
#   supportData：包含那些频繁项集支持数据的字典
#   minConf：表示最小可信度阈值
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    freqSet = frozenset(L)
    H = [frozenset([L[-1]])]
    rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf)
    return bigRuleList

# Lk：频繁项集列表
# k：项集元素个数
# Ck：输出
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    # print('Lk: ', Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            #前k-2项相同时合并两个集合
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])            
    return retList

# 从最初的项集中产生更多的关联规则，H为当前的候选规则集，产生下一层的候选规则集
# 频繁项集 H：可以出现在规则右部的元素列表  
# supportData：保存项集的支持度，brl保存生成的关联规则，minConf：最小可信度阀值 
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    while (len(freqSet) > m): # 判断长度 > m，这时即可求H的可信度
        H = calcConf(freqSet, H, supportData, brl, minConf)
        if (len(H) > 1): # 判断求完可信度后是否还有可信度大于阈值的项用来生成下一层H
            H = aprioriGen(H, m + 1)
            m += 1
        else: # 不能继续生成下一层候选关联规则，提前退出循环
            break

#计算规则的可信度，并找到满足最小可信度的规则存放在prunedH中，作为返回值返回
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    # print('supportData: ', supportData)
    for conseq in H:
        # print('freqSet', freqSet)
        # print('conseq', conseq)
        if supportData[freqSet - conseq] > 0:
            conf = supportData[freqSet] / supportData[freqSet - conseq]
            if conf >= minConf:
                # print(freqSet-conseq,"-->",conseq ,"conf:",conf)
                brl.append((freqSet-conseq, conseq, conf))  #填充置信度列表
                prunedH.append(conseq)    #保存满足最小置信度的规则
    return prunedH

        
if __name__=="__main__":
    dataSet = loadDataSet()
    # print(dataSet)
    L, suppData = apriori(dataSet, minSupport=0.1)
    i = 0
    for one in L:
        print("\n项数为 %s 的频繁项集：" % (i + 1), one)
        i += 1
        
    print("\nGenerateRules：")
    print("minConf=0.7时：")
    rules = generateRules(L, suppData, minConf=0.1)
    print("\nminConf=0.5时：")
    rules = generateRules(L, suppData, minConf=0.1)
