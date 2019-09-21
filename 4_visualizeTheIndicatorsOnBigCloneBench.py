# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:22:28 2019

@author: Administrator
"""
import pickle
import random
#coding:utf-8
import matplotlib
from matplotlib.pyplot import MultipleLocator
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve
import sys
def read_from_pkl(pickle_name):
    with open(pickle_name, 'rb') as pickle_f:
        python_content = pickle.load(pickle_f)
    return python_content
def save_to_pkl(python_content, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(python_content, pickle_f)
def cal_precision(TP, FP):
    if((TP+FP)==0): #比较特殊的一点。当全部判断为负的时候，或者很少判断为正的时候，此时FP几乎为0，因此precision一直为1。
        return 1
    else:
        return TP / (TP+FP)


def cal_recall(TP, FN):
    return TP / (TP+FN)

def cal_F1(P, R):
    return 2 * P * R / (P+R)
def cal_FPR(FP, TN):
    return FP / (FP+TN)
def experimentID_1():
#    metrices_traditionalRAE_Root_pkl='./result/metrices_traditionalRAE_root.xiaojiepkl'
#    metrices_traditionalRAE_Root_pkl='./result/20190622traditional_RAE_metrices_weightedRAE_root.xiaojiepkl'
    metrices_traditionalRAE_Root_pkl='./result/unweighted_BigCloneBench_traditionalRAE_metrics_root.xiaojiepkl'
    result_traditionalRAE_Root=read_from_pkl(metrices_traditionalRAE_Root_pkl)
    
#    metrices_weightedRAE_Weighted_pkl='./result/weighted_BigCloneBench_traditionalRAE_metrics_root_TF-IDF.xiaojiepkl'
#    result_weightedRAE_Weighted=read_from_pkl(metrices_weightedRAE_Weighted_pkl)    
#    result_weightedRAE_root=result_weightedRAE_Weighted
#    metrices_traditionalRAE_mean_pkl='./result/metrices_traditionalRAE_Mean.xiaojiepkl'
#    result_traditionalRAE_mean=read_from_pkl(metrices_traditionalRAE_mean_pkl)
    
    
#    metrices_weightedRAE_root_pkl='./result/metrices_weightedRAE_root.xiaojiepkl'
#    metrices_weightedRAE_root_pkl='./result/childparentweight/apywz_metrices_weightedRAE_root.xiaojiepkl' ###weighted模型上random参数
#    metrices_weightedRAE_root_pkl='./result/childparentweight/weighted_epoch_29_jwytx_metrices_weightedRAE_root.xiaojiepkl' ###weighted模型上random参数   
#    metrices_weightedRAE_root_pkl='./result/childparentweight/weighted_epoch_29_jwytx_metrices_weightedRAE_root.xiaojiepkl' ###weighted模型上random参数   
#    weighted_epoch_29_jwytx
#    result_weightedRAE_root=read_from_pkl(metrices_weightedRAE_root_pkl)
#######################保存的结果是：在weighted网络上，随机初始化网络参数，然后不训练，编码后用根节点进行度量######################
#    ###虽然命名用的仍然是weightedRAE，但是我们是随机化网络参数，然后不训练模型，然后用权重模型计算，发现要比tradtional模型（同样不训练，直接用root）要好！！！！
#    
#    metrices_weightedRAE_root_pkl_2='./result/childparentweight/ctvgn_metrices_weightedRAE_root.xiaojiepkl'
#    result_weightedRAE_root=read_from_pkl(metrices_weightedRAE_root_pkl)
######################保存的结果是：在weighted网络上，随机初始化网络参数，然后不训练，编码后用根节点进行度量######################
######################保存的结果是：在traditional网络上，随机初始化网络参数，然后不训练，编码后用根节点进行度量######################
#    metrices_random_root_pkl='./result/random_BigCloneBench_traditionalRAE_metrics_root.xiaojiepkl'
##    metrices_random_root_pkl='./result/metrices_random_root.xiaojiepkl'
#    metrices_random_mean_pkl='./result/metrices_random_mean.xiaojiepkl'
##    metrices_random_root_pkl='./result/childparentweight/iwtzf_metrices_weightedRAE_root.xiaojiepkl'
##    metrices_random_mean_pkl='./result/childparentweight/cweav_metrices_weightedRAE_mean.xiaojiepkl'
#    result_random_root=read_from_pkl(metrices_random_root_pkl)
#    result_random_mean=read_from_pkl(metrices_random_mean_pkl)
######################保存的结果是：在traditional网络上，随机初始化网络参数，然后不训练，编码后用根节点进行度量######################
#######################保存的结果是：在weighted网络上，训练后，编码后用根节点进行度量######################    
#    #训练30轮
#    metrices_weightedRAE_root_pkl='./result/childparentweight/tlped_metrices_weightedRAE_root.xiaojiepkl'
#    
#    #训练30轮
#    #训练20轮
##    metrices_weightedRAE_root_pkl_3='./result/childparentweight/rczoj_metrices_weightedRAE_root.xiaojiepkl'
#    ##训练20轮
#    result_weightedRAE_root=read_from_pkl(metrices_weightedRAE_root_pkl)
#######################保存的结果是：在weighted网络上，训练后，编码后用根节点进行度量######################        


######################保存的结果是：在weighted网络上，含TF-IDF，训练后，编码后用根节点进行度量######################    
#    metrices_weightedRAE_root_pkl='./result/childparentweight/wRAETF_IDF_uiyvq_metrices_weightedRAE_root.xiaojiepkl'
#    result_weightedRAE_root=read_from_pkl(metrices_weightedRAE_root_pkl)
######################保存的结果是：在weighted网络上，训练后，编码后用根节点进行度量######################        
     
#    metrices_weightedRAE_root_pkl='./result/childparentweight/metrices_weightedRAE_root.xiaojiepkl'
#    metrices_weightedRAE_meanAndRootMost_pkl='./result/metrices_weightedRAE_meanAndRootMost.xiaojiepkl'
#    metrices_weightedRAE_meanAndRootMost_pkl='./result/metrices_weightedRAE_meanAndRootMost2.xiaojiepkl'
    
#    result_traditionalRAE_Root=read_from_pkl(metrices_traditionalRAE_Root_pkl)
#    result_traditionalRAE_Root=read_from_pkl(metrices_traditionalRAE_mean_pkl)
#    result_weightedRAE_Weighted=read_from_pkl(metrices_weightedRAE_Weighted_pkl)
#    result_traditionalRAE_mean=read_from_pkl(metrices_traditionalRAE_mean_pkl)
#    
#    
    
#    result_weightedRAE_meanAndRootMost=read_from_pkl(metrices_weightedRAE_meanAndRootMost_pkl)
#    result_weightedRAE_root=result_weightedRAE_meanAndRootMost
        
#    metrices_weightedRAE_trainedWeighted_weightedAndCengci='./result/metrices_weightedRAE_trainedWeighted_weightedAndCengci.xiaojiepkl'
#    result_weightedRAE_weightedAndCengci=read_from_pkl(metrices_weightedRAE_trainedWeighted_weightedAndCengci)
#    result_weightedRAE_root=result_weightedRAE_weightedAndCengci

#####程序向量树模型
        
    metrices_PVT_root_pkl='./result/diffweight/ZHIXIN_ylabx_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
    
#    result_weightedRAE_root=read_from_pkl(metrices_ZHIXIN_root_pkl)
    
        ##########

    result_PVT_root=read_from_pkl(metrices_PVT_root_pkl)


###traditionalRAE_Root模型  
    precision_traditionalRAE_Root=[]
    recall_traditionalRAE_Root=[]
    yuzhi__traditionalRAE_Root=[]
    FPR_traditionalRAE_Root=[]
    recall_T1_traditionalRAE_Root=[]
    recall_T2_traditionalRAE_Root=[]
    recall_VST3_traditionalRAE_Root=[]
    recall_ST3_traditionalRAE_Root=[]
    recall_MT3_traditionalRAE_Root=[]
    recall_WT3_T4_traditionalRAE_Root=[]
    for obj in result_traditionalRAE_Root.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_traditionalRAE_Root.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_traditionalRAE_Root.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_traditionalRAE_Root.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_traditionalRAE_Root.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_traditionalRAE_Root.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_traditionalRAE_Root.append(recall_WT3_T4)
        
        precision_traditionalRAE_Root.append(p)
        recall_traditionalRAE_Root.append(r)
        yuzhi__traditionalRAE_Root.append(key)
        FPR_traditionalRAE_Root.append(FPR)


###result_PVT_root模型 
    precision_PVT_root=[]
    recall_PVT_root=[]
    yuzhi_PVT_root=[]
    FPR_PVT_root=[]
    recall_T1_PVT_root=[]
    recall_T2_PVT_root=[]
    recall_VST3_PVT_root=[]
    recall_ST3_PVT_root=[]
    recall_MT3_PVT_root=[]
    recall_WT3_T4_PVT_root=[]
    for obj in result_PVT_root.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_PVT_root.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_PVT_root.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_PVT_root.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_PVT_root.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_PVT_root.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_PVT_root.append(recall_WT3_T4)
        
        precision_PVT_root.append(p)
        recall_PVT_root.append(r)
        yuzhi_PVT_root.append(key)
        FPR_PVT_root.append(FPR)
      
    ###绘制
    
    
    
    plt.figure(1) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    
    fig = plt.figure(num=1, figsize=(15, 8),dpi=80) 
    plt.plot(recall_traditionalRAE_Root,precision_traditionalRAE_Root,'c*-',label='traditional_RAE')
    
#    plt.plot(recall_traditionalRAE_mean,precision_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(recall_PVT_root,precision_PVT_root,'m.-',label='PVT_root')
#    plt.plot(recall_weightedRAE_Weighted,precision_weightedRAE_Weighted,'b.-',label='weighted_RAE2')
    
    plt.grid()
    plt.legend()
    plt.show()


    
    
    
    
    
    
    
#    plt.figure(2) # 创建图表1
#    plt.title('Recall/threshold Curve')# give plot a title
#    plt.xlabel('threshold')# make axis labels
#    plt.ylabel('Recall')
#    
#    fig = plt.figure(num=2, figsize=(15, 8),dpi=80) 
#    plt.plot(yuzhi__traditionalRAE_Root,recall_traditionalRAE_Root,'c*-',label='traditional_RAE')
##    plt.plot(yuzhi__traditionalRAE_mean,recall_traditionalRAE_mean,label='traditional_RAE_mean')
#    plt.plot(yuzhi_weightedRAE_root,recall_weightedRAE_root,'m.-',label='weighted_RAE')
##    plt.plot(yuzhi_weightedRAE_Weighted,recall_weightedRAE_Weighted,'b.-',label='weighted_RAE_2')
#    plt.legend()
#    plt.show()
    
    #绘制AUC曲线
    
    plt.figure(3) # 创建图表3
    plt.title('TPR/FPR curve')# give plot a title
    plt.xlabel('FPR')#   make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=3, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_traditionalRAE_Root,recall_traditionalRAE_Root,'c*-',label='traditional_RAE')
    
#    plt.plot(FPR_traditionalRAE_mean,recall_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_weightedRAE_Weighted,'m.-',label='weighted_RAE')
    plt.legend()
    
            ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
    
    plt.figure(4) # 创建图表3
    plt.title('TPR_T1/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    fig = plt.figure(num=4, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_traditionalRAE_Root,recall_T1_traditionalRAE_Root,'c*-',label='traditional_RAE')
#    plt.plot(FPR_traditionalRAE_mean,recall_T1_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_T1_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_root,recall_T1_weightedRAE_root,'m.-',label='weighted_RAE_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_T1_weightedRAE_Weighted,'m.-',label='weighted_RAE_weighted')
    plt.legend()
        
    ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    ###设置坐标轴
    plt.show()
    
    plt.figure(5) # 创建图表3
    plt.title('TPR_T2/FPR curve ')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=5, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_traditionalRAE_Root,recall_T2_traditionalRAE_Root,'c*-',label='traditional_RAE')
#    plt.plot(FPR_traditionalRAE_mean,recall_T2_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_T2_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_T2_weightedRAE_Weighted,'b.-',label='weighted_RAE_weighted')
    plt.legend()
    
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()
        
 
    
    plt.figure(6) # 创建图表3
    plt.title('TPR_VST3/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=6, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_traditionalRAE_Root,recall_VST3_traditionalRAE_Root,'c*-',label='traditional_RAE')
#    plt.plot(FPR_traditionalRAE_mean,recall_VST3_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_VST3_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_VST3_weightedRAE_Weighted,'b.-',label='weighted_RAE_weighted')
    plt.legend()
    
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
   

    
    plt.figure(7) # 创建图表3
    plt.title('TPR_ST3/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    plt.ylim((0,1.1))
    
    fig = plt.figure(num=7, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_traditionalRAE_Root,recall_ST3_traditionalRAE_Root,'c*-',label='traditional_RAE')
#    plt.plot(FPR_traditionalRAE_mean,recall_ST3_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_ST3_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_ST3_weightedRAE_Weighted,'b.-',label='weighted_RAE_weighted')
    plt.legend()
    
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()    
    
    
    plt.figure(8) # 创建图表3
    plt.title('TPR_MT3/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=8, figsize=(8, 8),dpi=200) 
    plt.plot(FPR_traditionalRAE_Root,recall_MT3_traditionalRAE_Root,'c*-',label='traditional_RAE')
#    plt.plot(FPR_traditionalRAE_mean,recall_MT3_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_MT3_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_MT3_weightedRAE_Weighted,'b.-',label='weighted_RAE_weighted')
    plt.legend()
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度

    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()    
    
    plt.figure(9) # 创建图表3
    plt.title('TPR_T4/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=9, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_traditionalRAE_Root,recall_WT3_T4_traditionalRAE_Root,'c*-',label='traditional_RAE')
#    plt.plot(FPR_traditionalRAE_mean,recall_WT3_T4_traditionalRAE_mean,label='traditional_RAE_mean')
    plt.plot(FPR_PVT_root,recall_WT3_T4_PVT_root,'m.-',label='PVT_root')
#    plt.plot(FPR_weightedRAE_Weighted,recall_WT3_T4_weightedRAE_Weighted,'b.-',label='weighted_RAE_weighted')
    plt.legend()
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()   
    
    
    
    
            ###计算AUC值 下面的代码是针对梯形曲线才能计算面积。而我们的曲线不是梯形的。只能用积分的方式去计算。
#    def calAUC(FPR,Recall):
#        auc = 0.			
#        prev_x = 0
#        index_xy=0
#        for x in FPR:
#            y=Recall[index_xy]
#            if (x != prev_x):
#                auc+=(x-prev_x)*y
#                prev_x=x
#            index_xy+=1
#            pass
##        print('auc:{}'.format(auc))
#        return auc
#    
#    auc_T1=calAUC(FPR_PVT_root,recall_T1_PVT_root)
#    auc_T2=calAUC(FPR_PVT_root,recall_T2_PVT_root)
#    auc_VST3=calAUC(FPR_PVT_root,recall_VST3_PVT_root)
#    auc_ST3=calAUC(FPR_PVT_root,recall_ST3_PVT_root)
#    auc_MT3=calAUC(FPR_PVT_root,recall_MT3_PVT_root)
#    auc_WT3_T4=calAUC(FPR_PVT_root,recall_WT3_T4_PVT_root)
#    print (auc_T1)
#    print (auc_T2)
#    print (auc_VST3)
#    print (auc_ST3)
#    print (auc_MT3)
#    print (auc_WT3_T4)
    auc_T1=np.trapz(recall_T1_PVT_root,FPR_PVT_root)
    auc_T2=np.trapz(recall_T2_PVT_root,FPR_PVT_root)
    auc_VST3=np.trapz(recall_VST3_PVT_root,FPR_PVT_root)
    auc_ST3=np.trapz(recall_ST3_PVT_root,FPR_PVT_root)
    auc_MT3=np.trapz(recall_MT3_PVT_root,FPR_PVT_root)
    auc_WT3_T4=np.trapz(recall_WT3_T4_PVT_root,FPR_PVT_root)
    print (auc_T1)
    print (auc_T2)
    print (auc_VST3)
    print (auc_ST3)
    print (auc_MT3)
    print (auc_WT3_T4)
    
    
    
    fig = plt.figure(10,figsize=(15, 8))
    plt.title('TPR/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    
    
    label_1='Type-1 (AUC=1.0)'
    label_2='Type-2 (AUC=0.9989)'
    label_3='Very Strongly Type-3 (AUC=0.9987)'
    label_4='Strongly Type-3 (AUC=0.9860)'
    label_5='Moderately Type-3 (AUC=0.9628)'
    label_6='Type-4 (AUC=0.7755)'
    

    plt.plot(FPR_PVT_root,recall_T1_PVT_root,'m.-',label=label_1)
    plt.plot(FPR_PVT_root,recall_T2_PVT_root,'b--',label=label_2)
    plt.plot(FPR_PVT_root,recall_VST3_PVT_root,'g-',label=label_3)
    plt.plot(FPR_PVT_root,recall_ST3_PVT_root,'rv:',label=label_4)
    plt.plot(FPR_PVT_root,recall_MT3_PVT_root,'c^-',label=label_5)
    plt.plot(FPR_PVT_root,recall_WT3_T4_PVT_root,'y>-',label=label_6)
    plt.legend()
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
    ax.yaxis.grid(True, which='major') #y坐标轴的网格使用次刻度
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
#    plt.axis('scaled')    
#    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()   
    

    
    
    
    
    pass
if __name__ == "__main__":    
    experimentID_1()    