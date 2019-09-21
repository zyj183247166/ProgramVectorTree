import pickle
import random
#coding:utf-8
import matplotlib
import matplotlib
from matplotlib.pyplot import MultipleLocator
#matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_curve
import sys
import time

def read_from_pkl(pickle_name):
    with open(pickle_name, 'rb') as pickle_f:
        python_content = pickle.load(pickle_f)
    return python_content
def save_to_pkl(python_content, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(python_content, pickle_f)
def read_from_pkl_writtenBypy2(pickle_name):
    #https://blog.csdn.net/qq_33373858/article/details/83862381 python2与python3保存的pickle文件不兼容问题
    with open(pickle_name, 'rb') as pickle_f:
        python_content = pickle.load(pickle_f,encoding='latin1')
    return python_content
####将id对应向量之间的距离保存下来，可以反复用，避免总是一直再算
def save_distance_file(bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path,id_pair_map_distance_pkl):
    all_data_dict = read_from_pkl(bigCloneBenchIdPair_PKL_Path)
    id_map_vector=read_from_pkl(idMapVector_PKL_Path)
    id_pair_map_distance_dict={}
    index=1
    for obj in all_data_dict.items():
#       ####测试用
        key,value=obj #key是个二元组，保存的是bigCloneBench的函数ID之间的对应关系。value是标签。
        ###########使用我们的模型去进行检测
#        detectionResult=cloneDetector(key)
        #首先是获取该id对应的向量
        function_id1_inBCB=str(key[0])
        function_id2_inBCB=str(key[1])
        vector1=id_map_vector[function_id1_inBCB]
        vector2=id_map_vector[function_id2_inBCB]
        distance=np.linalg.norm(vector1-vector2)
        ####将下面的过程单独抽取出来。即将距离值写入一个pkl文件。
        id_pair_map_distance_dict[key]=distance
        index=index+1
        if(index%10000==0):
            print ('index:%d'%index)
        pass
    save_to_pkl(id_pair_map_distance_dict,id_pair_map_distance_pkl)
    pass
def get_basic_data(distance_PKL,yuzhi):
    #读取用于评测的所有id_pair
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数
    all_data_dict = read_from_pkl(bigCloneBenchIdPair_PKL_Path)
    #读取用于评测的所有id_pair对应的clone_pair
    clone_type_pkl = './SplitDataSet/data/all_clone_id_pair_cloneType_XIAOJIE.pkl'
    all_id_pair_cloneType_dict=read_from_pkl(clone_type_pkl)
    #读取用于评测的所有id_pair对应的距离，根据模型指定
    id_pair_map_distance_dict=read_from_pkl(distance_PKL)
    ####不能用下面的程序去统计数目。因为我们对BigCloneBench中错误标记的，全部去除了。而all_clone_id_pair_cloneType_XIAOJIE文件中没有做如此处理。
#    num_T1=0
#    num_T2c=0
#    num_T2b=0
#    num_VST3=0
#    num_ST3=0
#    num_MT3=0
#    num_WT3_T4=0
#    num_otherType=0
#    ##
#    #遍历all_id_pair_cloneType_dict字典
#    for obj in all_id_pair_cloneType_dict.items():
#        _,clone_type=obj
#        if clone_type=='1':
#            num_T1+=1
#        elif clone_type=='2c':
#            num_T2c+=1
#        elif clone_type=='2b':
#            num_T2b+=1
#        elif clone_type=='VST3':
#            num_VST3+=1
#        elif clone_type=='ST3':
#            num_ST3+=1
#        elif clone_type=='MT3':
#            num_MT3+=1
#        elif clone_type=='WT3_T4':
#            num_WT3_T4+=1
#        else:
#            num_otherType+=1
#    if num_otherType==0:
#        print('all_clone_id_pair_cloneType文件有错误')
    #遍历all_id_pair_cloneType_dict字典
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    #统计各个clone类型的查全率，这里只有克隆关系，即来源于clones表。因为非克隆表中没有细分各种类型。
    TP_T1=0
    TP_T2c=0
    TP_T2b=0
    TP_VST3=0
    TP_ST3=0
    TP_MT3=0
    TP_WT3_T4=0
    TP_otherType=0

    FN_T1=0
    FN_T2c=0
    FN_T2b=0
    FN_VST3=0
    FN_ST3=0
    FN_MT3=0
    FN_WT3_T4=0
    FN_otherType=0
    #统计各个clone类型的查全率，这里只有克隆关系，即来源于clones表。因为非克隆表中没有细分各种类型。
    index=1
    for obj in all_data_dict.items():
#        print((index))
        ####测试用,执行时注释掉。
#        if(index==1000):
#            break
        ####测试用
        key,value=obj #key是个二元组，保存的是bigCloneBench的函数ID之间的对应关系。value是标签。
        ###########使用我们的模型去进行检测
#        detectionResult=cloneDetector(key)
        #首先是获取该id对应的向量
        distance=id_pair_map_distance_dict[key]
        ####将下面的过程单独抽取出来。即将距离值写入一个pkl文件。
        if (distance<=yuzhi):
            detectionResult=1
        else:
            detectionResult=0
        #计算欧几里得距离
        ###########使用我们的模型去进行检测
        
        if value==1: #如果是clone，然后我们要考虑各个clone类型的。
            id1=str(key[0]) #我们两个pkl文件中的id虽然是一个意思，但是一个是int，一个是string
            id2=str(key[1])
            keyyy=(id1,id2)
            clone_type=all_id_pair_cloneType_dict[keyyy]
            if detectionResult==1:
                TP+=1
                if clone_type=='1':
                    TP_T1+=1
                elif clone_type=='2c':
                    TP_T2c+=1
                elif clone_type=='2b':
                    TP_T2b+=1
                elif clone_type=='VST3':
                    TP_VST3+=1
                elif clone_type=='ST3':
                    TP_ST3+=1
                elif clone_type=='MT3':
                    TP_MT3+=1
                elif clone_type=='WT3_T4':
                    TP_WT3_T4+=1
                else:
                    TP_otherType+=1
                    pass
                pass
            else:
                FN+=1
                if clone_type=='1':
                    FN_T1+=1
                elif clone_type=='2c':
                    FN_T2c+=1
                elif clone_type=='2b':
                    FN_T2b+=1
                elif clone_type=='VST3':
                    FN_VST3+=1
                elif clone_type=='ST3':
                    FN_ST3+=1
                elif clone_type=='MT3':
                    FN_MT3+=1
                elif clone_type=='WT3_T4':
                    FN_WT3_T4+=1
                else:
                    FN_otherType+=1
                    pass
                pass
        elif value==0:
            if detectionResult==1:
                FP+=1
            else:
                TN+=1
        else:
            pass
        index=index+1
        if(index%10000==0):
            print('index:%d'%index)
        pass
    if ((TP_otherType!=0)or(FN_otherType!=0)):
        print('error 0')
        return 
    if((TP+TN+FP+FN+1)!=index):
        print('error 1')
        return 
    if(((TP_T1+TP_T2c+TP_T2b+TP_VST3+TP_ST3+TP_MT3+TP_WT3_T4)!=TP)or((FN_T1+FN_T2c+FN_T2b+FN_VST3+FN_ST3+FN_MT3+FN_WT3_T4)!=FN)):
        print('error 2')
        return 
    print ('Done')
    return TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4
        


def cal_accuracy(TP, TN, FP, FN):
    return (TP+TN) / (TP+TN+FP+FN)


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

def huaPR():
    
    plt.figure(1) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
     
    #y_true和y_scores分别是gt label和predict score
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([1, 1, 1, 1])
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(1)
    plt.plot(precision, recall)
    plt.show()

def analyse(distance_pkl,metrics_pkl):
    yuzhi=0.0
    jiange=0.05
    result={}
    ###########计算指标
    TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data(distance_pkl,yuzhi)
    R = cal_recall(TP, FN)
    
#    print('(TP, TN, FP, FN)')
#    print(TP, TN, FP, FN)
#    print('(acc, P, R, F1)')
#    print(acc, P, R, F1)
    result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
    
    ###########计算指标
    while(R<0.99): #只要查全率没有达到1，我们就不断的去增大阈值。当得到最大的阈值以后，我们再重新计算一次，对阈值的增长区间进行合理的划分。不然的话，我们不知道一次应该将阈值增长多少。
        print('yuzhi{},Recall:{}'.format(yuzhi,R))
#        jiange=jiange*2#加快到达Recall为1的过程。
        yuzhi=yuzhi+jiange
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data(distance_pkl,yuzhi)
        R = cal_recall(TP, FN)
        result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
        pass
    ######最后，阈值和指标就在result列表中了。
    print('yuzhi{},Recall:{}'.format(yuzhi,R))
    save_to_pkl(result,metrics_pkl)
    return result
#    xxxx=read_from_pkl(path)
def experimentID_1():
    #先把模型上计算的向量之间的距离计算出来并保存
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数（被他们错误标记），以及去重
#    idMapVector_PKL_Path_randomRoot='./vector/randomModel_IdinBigCloneBench_Map_RootVector.xiaojiepkl'
#    idMapVector_PKL_Path_randomMean='./vector/randomModel_IdinBigCloneBench_Map_MeanVector.xiaojiepkl'
#    idMapVector_PKL_Path_trainedMean='./vector/trainedModel_IdinBigCloneBench_Map_MeanVector.xiaojiepkl'
#    idMapVector_PKL_Path_trainedRoot='./vector/trainedModelIdinBigCloneBench_Map_RootVector.xiaojiepkl'
#    idMapVector_PKL_Path_trainedMean2='./vector/mdlaz_mean.xiaojiepkl'
#    idMapVector_PKL_Path_trainedRoot2='./vector/wymqn_root.xiaojiepkl'
    
#    id_pair_map_distance_pkl_randomRoot='./vector/randomModel_IdPairinBigCloneBench_Map_Distance_RootVector.xiaojiepkl'
#    id_pair_map_distance_pkl_randomMean='./vector/randomModel_IdPairinBigCloneBench_Map_Distance_MeanVector.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedMean='./vector/trainedModel_IdPairinBigCloneBench_Map_Distance_MeanVector.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedRoot='./vector/trainedModel_IdPairinBigCloneBench_Map_Distance_RootVector.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedMean2='./vector/trainedModel_IdPairinBigCloneBench_Map_Distance_MeanVector2.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedRoot2='./vector/trainedModel_IdPairinBigCloneBench_Map_Distance_RootVector2.xiaojiepkl'
    
    
    #####先根据模型去计算距离，然后保存到pkl文件中去
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_randomRoot,id_pair_map_distance_pkl=id_pair_map_distance_pkl_randomRoot)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_randomMean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_randomMean)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedMean2,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedMean2)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedRoot2,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedRoot2)
    ##保存到pkl文件以后，后续可以直接读取
#    id_pair_map_distance_randomRoot=read_from_pkl(id_pair_map_distance_pkl_randomRoot)
#    id_pair_map_distance_randomMean=read_from_pkl(id_pair_map_distance_pkl_randomMean)
    #####先根据模型去计算距离，然后保存到pkl文件中去
    
    
    metrices_random_root_pkl='./result/metrices_random_root.xiaojiepkl'
    metrices_random_mean_pkl='./result/metrices_random_mean.xiaojiepkl'
    metrices_trained_mean_pkl='./result/metrices_trained_mean.xiaojiepkl'
    metrices_trained_root_pkl='./result/metrices_trained_root.xiaojiepkl'
#    metrices_trained_mean_pkl2='./result/metrices_trained_mean2.xiaojiepkl'
#    metrices_trained_root_pkl2='./result/metrices_trained_root2.xiaojiepkl'
    #####读取pkl文件中的距离，并且根据阈值，去计算相关指标,存入相关pkl文件
#    analyse(distance_pkl=id_pair_map_distance_pkl_randomRoot,metrics_pkl=metrices_random_root_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_randomMean,metrics_pkl=metrices_random_mean_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedMean,metrics_pkl=metrices_trained_mean_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedRoot,metrics_pkl=metrices_trained_root_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedMean2,metrics_pkl=metrices_trained_mean_pkl2)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedRoot2,metrics_pkl=metrices_trained_root_pkl2)
    ##保存到pkl文件以后，后续可以直接读取
    result_random_root=read_from_pkl(metrices_random_root_pkl)
    result_random_mean=read_from_pkl(metrices_random_mean_pkl)
    result_trained_mean=read_from_pkl(metrices_trained_mean_pkl)
    result_trained_root=read_from_pkl(metrices_trained_root_pkl)
#    result_trained_mean=read_from_pkl(metrices_trained_mean_pkl2)
#    result_trained_root=read_from_pkl(metrices_trained_root_pkl2)
    #####读取pkl文件中的距离，并且根据阈值，去计算相关指标,存入相关pkl文件
##    result_random_root=read_from_pkl('./result/random_root.xiaojiepkl') #
##    result_random_mean=read_from_pkl('./result/random_mean.xiaojiepkl') #
    ##########绘制PR曲线
   
###random_root模型 
    precision_random_root=[]
    recall_random_root=[]
    yuzhi__random_root=[]
    FPR_random_root=[]
    for obj in result_random_root.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        precision_random_root.append(p)
        recall_random_root.append(r)
        yuzhi__random_root.append(key)
        FPR_random_root.append(FPR)
###random_mean模型 
    precision_random_mean=[]
    recall_random_mean=[]
    yuzhi__random_mean=[]
    FPR_random_mean=[]
    for obj in result_random_mean.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        precision_random_mean.append(p)
        recall_random_mean.append(r)
        yuzhi__random_mean.append(key)
        FPR_random_mean.append(FPR)
###trained_mean模型 
    precision_trained_mean=[]
    recall_trained_mean=[]
    yuzhi__trained_mean=[]
    FPR_trained_mean=[]
    recall_T1_trained_mean=[]
    recall_T2_trained_mean=[]
    recall_VST3_trained_mean=[]
    recall_ST3_trained_mean=[]
    recall_MT3_trained_mean=[]
    recall_WT3_T4_trained_mean=[]
    for obj in result_trained_mean.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_trained_mean.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_trained_mean.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_trained_mean.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_trained_mean.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_trained_mean.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_trained_mean.append(recall_WT3_T4)
        
        precision_trained_mean.append(p)
        recall_trained_mean.append(r)
        yuzhi__trained_mean.append(key)
        FPR_trained_mean.append(FPR)
###trained_root模型 
    precision_trained_root=[]
    recall_trained_root=[]
    yuzhi__trained_root=[]
    FPR_trained_root=[]
    recall_T1_trained_root=[]
    recall_T2_trained_root=[]
    recall_VST3_trained_root=[]
    recall_ST3_trained_root=[]
    recall_MT3_trained_root=[]
    recall_WT3_T4_trained_root=[]
    for obj in result_trained_root.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_trained_root.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_trained_root.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_trained_root.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_trained_root.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_trained_root.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_trained_root.append(recall_WT3_T4)
        
        precision_trained_root.append(p)
        recall_trained_root.append(r)
        yuzhi__trained_root.append(key)
        FPR_trained_root.append(FPR)
        
    ###绘制
    plt.figure(1) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    
    fig = plt.figure(num=1, figsize=(15, 8),dpi=80) 
    plt.plot(recall_random_root,precision_random_root,label='random_root')
    plt.plot(recall_random_mean,precision_random_mean,label='random_mean')
    plt.legend()
    plt.show()
    
    plt.figure(2) # 创建图表1
    plt.title('Recall/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=2, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi__random_root,recall_random_root,label='random_root')
    plt.plot(yuzhi__random_mean,recall_random_mean,label='random_mean')
    plt.legend()
    plt.show()
    
    plt.figure(3) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    
    fig = plt.figure(num=3, figsize=(15, 8),dpi=80) 
    plt.plot(recall_trained_mean,precision_trained_mean,label='trained_mean')
    plt.plot(recall_random_mean,precision_random_mean,label='random_mean')
    plt.legend()
    plt.show()
    
    plt.figure(4) # 创建图表3
    plt.title('Recall/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=4, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi__trained_mean,recall_trained_mean,label='trained_mean')
    plt.plot(yuzhi__random_mean,recall_random_mean,label='random_mean')
    plt.legend()
    plt.show()

    plt.figure(5) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    
    fig = plt.figure(num=5, figsize=(15, 8),dpi=80) 
    plt.plot(recall_trained_root,precision_trained_root,label='trained_root')
    plt.plot(recall_random_root,precision_random_root,label='random_root')
    plt.legend()
    plt.show()

    plt.figure(6) # 创建图表3
    plt.title('Recall/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=6, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi__trained_root,recall_trained_root,label='trained_root')
    plt.plot(yuzhi__random_root,recall_random_root,label='random_root')
    plt.legend()
    plt.show()
    
    
    
    plt.figure(7) # 创建图表3
    plt.title('Recall/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=7, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi__trained_root,recall_trained_root,label='trained_root')
    plt.plot(yuzhi__trained_mean,recall_trained_mean,label='trained_mean')
    plt.legend()
    plt.show()
    
    
    
    
    #绘制AUC曲线
    
    plt.figure(8) # 创建图表3
    plt.title('FPR/TPR')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=8, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_trained_root,recall_trained_root,label='trained_root')
#    plt.plot(FPR_random_root,recall_random_root,label='random_root')
    plt.plot(FPR_trained_mean,recall_trained_mean,label='trained_mean')
#    plt.plot(FPR_random_mean,recall_random_mean,label='random_mean')
    plt.legend()
    plt.show()
    
    
    
    plt.figure(9) # 创建图表3
    plt.title('FPR/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('FPR')
    
    fig = plt.figure(num=9, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi__trained_root,FPR_trained_root,label='trained_root')
    plt.plot(yuzhi__trained_mean,FPR_trained_mean,label='trained_mean')
    plt.legend()
    plt.show()
    
    plt.figure(10) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    
    fig = plt.figure(num=10, figsize=(15, 8),dpi=80) 
    plt.plot(recall_trained_root,precision_trained_root,label='trained_root')
    plt.plot(recall_trained_mean,precision_trained_mean,label='trained_mean') 
    plt.legend()
    plt.show()
    
    
    
    plt.figure(11) # 创建图表3

    
    fig=plt.figure(num=11, figsize=(15, 8),dpi=80) 
    plt.title('FPR/TPR')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    plt.plot(FPR_trained_root,recall_T1_trained_root,label='recall_T1_trained_root')
#    plt.plot(FPR_trained_root,recall_T2_trained_root,label='recall_T2_trained_root')
#    plt.plot(FPR_trained_root,recall_VST3_trained_root,label='recall_VST3_trained_root')
#    plt.plot(FPR_trained_root,recall_ST3_trained_root,label='recall_ST3_trained_root')
#    plt.plot(FPR_trained_root,recall_MT3_trained_root,label='recall_MT3_trained_root')
#    plt.plot(FPR_trained_root,recall_WT3_T4_trained_root,label='recall_WT3_T4_trained_root')
    
    
    plt.plot(FPR_trained_mean,recall_T1_trained_mean,label='recall_T1_trained_mean')
#    plt.plot(FPR_trained_mean,recall_T2_trained_mean,label='recall_T2_trained_mean')
#    plt.plot(FPR_trained_mean,recall_VST3_trained_mean,label='recall_VST3_trained_mean')
#    plt.plot(FPR_trained_mean,recall_ST3_trained_mean,label='recall_ST3_trained_mean')
#    plt.plot(FPR_trained_mean,recall_MT3_trained_mean,label='recall_MT3_trained_mean')
#    plt.plot(FPR_trained_mean,recall_WT3_T4_trained_mean,label='recall_WT3_T4_trained_mean')
    

#    plt.plot(FPR_random_mean,recall_random_mean,label='random_mean')
    plt.legend()
    plt.show()
    
    pass

def save_distance_file_experimentID3(temp_path,detected_clone_pairs_lineInCorpus_Path,fullCorpusLine_Map_Vector_pkl,detected_LineInfullCorpus_pair_map_distance_pkl):
    ####一开始，我是写了两个字典。一个是读取检测到的克隆对，另外一个是克隆对作为键值，然后distance作为距离。但是总是内存不足。由于两个字典并存，占用内存过大。
    #于是，先将结果写入文件，然后删除一个字典，再写入另外一个字典
    detected_clone_pairs_lineInCorpus=read_from_pkl(detected_clone_pairs_lineInCorpus_Path)
    fullCorpusLine_Map_Vector_dict=read_from_pkl(fullCorpusLine_Map_Vector_pkl)

    index=0
    with open(temp_path, 'w') as fv:
        for key in detected_clone_pairs_lineInCorpus:
            line_in_fullCorpus_1=key[0]
            line_in_fullCorpus_2=key[1]
            vector1=fullCorpusLine_Map_Vector_dict[line_in_fullCorpus_1]
            vector2=fullCorpusLine_Map_Vector_dict[line_in_fullCorpus_2]
            distance=np.linalg.norm(vector1-vector2)
            ####将下面的过程单独抽取出来。即将距离值写入一个pkl文件。
            fv.write(str(line_in_fullCorpus_1)+','+str(line_in_fullCorpus_2)+','+str(distance)+'\n')
            if(index%10000==0):
                print ('index:%d'%index)
#            ########测试
#            print(line_in_fullCorpus_1,line_in_fullCorpus_2,distance)
#            if(index==10):
#                break
            index=index+1
            ########测试
            pass
    detected_clone_pairs_lineInCorpus.clear()
    del(detected_clone_pairs_lineInCorpus)
    fullCorpusLine_Map_Vector_dict.clear()
    del(fullCorpusLine_Map_Vector_dict)
    detected_LineInfullCorpus_pair_Map_Distance_dict={}
    index=0
    with open(temp_path, 'r') as fv:
        for i,item in enumerate(fv):
            items=item.strip().split(',')
            line_in_fullCorpus_1=int(items[0])
            line_in_fullCorpus_2=int(items[1])
            key=(line_in_fullCorpus_1,line_in_fullCorpus_2)
            value=float(items[2])
#            print(line_in_fullCorpus_1,line_in_fullCorpus_2,value)
            detected_LineInfullCorpus_pair_Map_Distance_dict[key]=value
            if(index%10000==0):
                print ('index:%d'%index)
#            ########测试
#            print(line_in_fullCorpus_1,line_in_fullCorpus_2,distance)
#            if(index==10):
#                break
            index=index+1
            ########测试
            pass
    print('save_distance_file_experimentID3')
    print(index)
    save_to_pkl(detected_LineInfullCorpus_pair_Map_Distance_dict,detected_LineInfullCorpus_pair_map_distance_pkl)
    detected_LineInfullCorpus_pair_Map_Distance_dict.clear()
    del(detected_LineInfullCorpus_pair_Map_Distance_dict)
    pass
def get_basic_data_experimentID3(distance_PKL,yuzhi):
    #读取用于评测的所有id_pair
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数
    all_data_dict = read_from_pkl(bigCloneBenchIdPair_PKL_Path)
    
    all_idMapline_pkl = './SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
    all_idMapline_dict= read_from_pkl(all_idMapline_pkl)
    #读取用于评测的所有id_pair对应的clone_pair
    clone_type_pkl = './SplitDataSet/data/all_clone_id_pair_cloneType_XIAOJIE.pkl'
    all_id_pair_cloneType_dict=read_from_pkl(clone_type_pkl)
    #读取用于评测的所有id_pair对应的距离，根据模型指定
    detected_LineInfullCorpus_pair_Map_Distance_dict=read_from_pkl(distance_PKL)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    #统计各个clone类型的查全率，这里只有克隆关系，即来源于clones表。因为非克隆表中没有细分各种类型。
    TP_T1=0
    TP_T2c=0
    TP_T2b=0
    TP_VST3=0
    TP_ST3=0
    TP_MT3=0
    TP_WT3_T4=0
    TP_otherType=0

    FN_T1=0
    FN_T2c=0
    FN_T2b=0
    FN_VST3=0
    FN_ST3=0
    FN_MT3=0
    FN_WT3_T4=0
    FN_otherType=0
    #统计各个clone类型的查全率，这里只有克隆关系，即来源于clones表。因为非克隆表中没有细分各种类型。
    index=1
    yagenzhaobudao=0
    for obj in all_data_dict.items():
#        print((index))
        ####测试用,执行时注释掉。
#        if(index==1000):
#            break
        ####测试用
        key,value=obj #key是个二元组，保存的是bigCloneBench的函数ID之间的对应关系。value是标签。
        id1=str(key[0])
        id2=str(key[1])
        line1=all_idMapline_dict[id1]
        line2=all_idMapline_dict[id2]
        key_lineinfullCorpus_tuple=(line1,line2)
        key_lineinfullCorpus_tuple2=(line2,line1) #交换一下顺序，避免我们检测的结果和BigCloneBench标记的克隆对的顺序恰好相反。
        foundFlag=False
        if key_lineinfullCorpus_tuple in detected_LineInfullCorpus_pair_Map_Distance_dict:
            foundFlag=True    
            distance=detected_LineInfullCorpus_pair_Map_Distance_dict[key_lineinfullCorpus_tuple]
        elif key_lineinfullCorpus_tuple2 in detected_LineInfullCorpus_pair_Map_Distance_dict:
            foundFlag=True
            distance=detected_LineInfullCorpus_pair_Map_Distance_dict[key_lineinfullCorpus_tuple2]
        else:
            foundFlag=False
            pass
        if (foundFlag==False): #如果直接就没有检测到，我们的克隆模型，就认为这个克隆对不是克隆。即将其标记为负
            detectionResult=0
            yagenzhaobudao+=1
            #应该直接统计一下foundFlag==False，即没有找到的。因为如果用NSG算法搜索search_K个近邻之后，检测到的结果中都没有的话，那么也就说明了查全率的上限了。
        elif (foundFlag==True):
            if (distance<=yuzhi):
                detectionResult=1 #如果找到了。并且距离小于阈值，我们的模型才认为这个克隆对是克隆。
            else:
                detectionResult=0 #如果报告的距离大于阈值，我们的模型仍然不认为这个克隆对是克隆，即仍然标定为负。
        ###########使用我们的模型去进行检测
        if value==1: #如果是clone，然后我们要考虑各个clone类型的。
            id1=str(key[0]) #我们两个pkl文件中的id虽然是一个意思，但是一个是int，一个是string
            id2=str(key[1])
            keyyy=(id1,id2)
            clone_type=all_id_pair_cloneType_dict[keyyy]
            if detectionResult==1:
                TP+=1
                if clone_type=='1':
                    TP_T1+=1
                elif clone_type=='2c':
                    TP_T2c+=1
                elif clone_type=='2b':
                    TP_T2b+=1
                elif clone_type=='VST3':
                    TP_VST3+=1
                elif clone_type=='ST3':
                    TP_ST3+=1
                elif clone_type=='MT3':
                    TP_MT3+=1
                elif clone_type=='WT3_T4':
                    TP_WT3_T4+=1
                else:
                    TP_otherType+=1
                    pass
                pass
            else:
                FN+=1
                if clone_type=='1':
                    FN_T1+=1
                elif clone_type=='2c':
                    FN_T2c+=1
                elif clone_type=='2b':
                    FN_T2b+=1
                elif clone_type=='VST3':
                    FN_VST3+=1
                elif clone_type=='ST3':
                    FN_ST3+=1
                elif clone_type=='MT3':
                    FN_MT3+=1
                elif clone_type=='WT3_T4':
                    FN_WT3_T4+=1
                else:
                    FN_otherType+=1
                    pass
                pass
        elif value==0:
            if detectionResult==1:
                FP+=1
            else:
                TN+=1
        else:
            pass
        index=index+1
        if(index%10000==0):
            print('index:%d'%index)
        pass
    if ((TP_otherType!=0)or(FN_otherType!=0)):
        print('error 0')
        return 
    if((TP+TN+FP+FN+1)!=index):
        print('error 1')
        return 
    if(((TP_T1+TP_T2c+TP_T2b+TP_VST3+TP_ST3+TP_MT3+TP_WT3_T4)!=TP)or((FN_T1+FN_T2c+FN_T2b+FN_VST3+FN_ST3+FN_MT3+FN_WT3_T4)!=FN)):
        print('error 2')
        return 
    print ('Done')
    print ('yagenzhaobudao:%d'%yagenzhaobudao)
    print ('由于NSG算法设置的是返回K个近邻，我们这次实验使用的是K=15，返回的是15个近邻。无论阈值如何设置，最高查全率是：')
    print (1-(yagenzhaobudao/index))
    print ('最高查全率出现的情况就是NSG算法返回的所有克隆检测结果，无论阈值如何设置，都认为是检测出的克隆。在这个时候，这些数据占整个BigCloneBench所有标记克隆对的比例。')
    return TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4
        
def get_basic_data_experimentID4(detected_clone_pairs_lineInCorpus_Path):
    
    
    
    #############读取检测到的克隆对
    
    path=detected_clone_pairs_lineInCorpus_Path+'.xiaojiepkl'
    index=0
    detected_clone_pairs_lineInCorpus={}
    with open(detected_clone_pairs_lineInCorpus_Path, 'r') as fv:
        
        for i,item in enumerate(fv):
            items=item.strip().split(',')
            line_in_fullCorpus_1=int(items[0])
            line_in_fullCorpus_2=int(items[1])
            key=(line_in_fullCorpus_1,line_in_fullCorpus_2)
            key_2=(line_in_fullCorpus_2,line_in_fullCorpus_1)
            if(key in detected_clone_pairs_lineInCorpus):
                pass
            elif(key_2 in detected_clone_pairs_lineInCorpus):
                pass
            else:
                detected_clone_pairs_lineInCorpus[key]=1
            if(index%10000==0):
                print ('index:%d'%index)
            index=index+1
            ########测试
            pass
    print(index)
    
    print(path)

    save_to_pkl(detected_clone_pairs_lineInCorpus,path)
####    return 
    pass
    detected_clone_pairs_lineInCorpus=read_from_pkl(path)
    #读取用于评测的所有id_pair
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数
    all_data_dict = read_from_pkl(bigCloneBenchIdPair_PKL_Path)
    
    all_idMapline_pkl = './SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
    all_idMapline_dict= read_from_pkl(all_idMapline_pkl)
    #读取用于评测的所有id_pair对应的clone_pair
    clone_type_pkl = './SplitDataSet/data/all_clone_id_pair_cloneType_XIAOJIE.pkl'
    all_id_pair_cloneType_dict=read_from_pkl(clone_type_pkl)

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    #统计各个clone类型的查全率，这里只有克隆关系，即来源于clones表。因为非克隆表中没有细分各种类型。
    TP_T1=0
    TP_T2c=0
    TP_T2b=0
    TP_VST3=0
    TP_ST3=0
    TP_MT3=0
    TP_WT3_T4=0
    TP_otherType=0

    FN_T1=0
    FN_T2c=0
    FN_T2b=0
    FN_VST3=0
    FN_ST3=0
    FN_MT3=0
    FN_WT3_T4=0
    FN_otherType=0
    #统计各个clone类型的查全率，这里只有克隆关系，即来源于clones表。因为非克隆表中没有细分各种类型。
    index=1
    yagenzhaobudao=0
    for obj in all_data_dict.items():
#        print((index))
        ####测试用,执行时注释掉。
#        if(index==1000):
#            break
        ####测试用
        key,value=obj #key是个二元组，保存的是bigCloneBench的函数ID之间的对应关系。value是标签。
        id1=str(key[0])
        id2=str(key[1])
        line1=all_idMapline_dict[id1]
        line2=all_idMapline_dict[id2]
        key_lineinfullCorpus_tuple=(line1,line2)
        key_lineinfullCorpus_tuple2=(line2,line1) #交换一下顺序，避免我们检测的结果和BigCloneBench标记的克隆对的顺序恰好相反。
        foundFlag=False
        if key_lineinfullCorpus_tuple in detected_clone_pairs_lineInCorpus:
            foundFlag=True    
        elif key_lineinfullCorpus_tuple2 in detected_clone_pairs_lineInCorpus:
            foundFlag=True
        else:
            foundFlag=False
            pass
        if (foundFlag==False): #如果直接就没有检测到，我们的克隆模型，就认为这个克隆对不是克隆。即将其标记为负
            detectionResult=0
        elif (foundFlag==True):
            detectionResult=1 #如果找到了。并且距离小于阈值，我们的模型才认为这个克隆对是克隆。
        ###########使用我们的模型去进行检测
        if value==1: #如果是clone，然后我们要考虑各个clone类型的。
            id1=str(key[0]) #我们两个pkl文件中的id虽然是一个意思，但是一个是int，一个是string
            id2=str(key[1])
            keyyy=(id1,id2)
            clone_type=all_id_pair_cloneType_dict[keyyy]
            if detectionResult==1:
                TP+=1
                if clone_type=='1':
                    TP_T1+=1
                elif clone_type=='2c':
                    TP_T2c+=1
                elif clone_type=='2b':
                    TP_T2b+=1
                elif clone_type=='VST3':
                    TP_VST3+=1
                elif clone_type=='ST3':
                    TP_ST3+=1
                elif clone_type=='MT3':
                    TP_MT3+=1
                elif clone_type=='WT3_T4':
                    TP_WT3_T4+=1
                else:
                    TP_otherType+=1
                    pass
                pass
            else:
                FN+=1
                if clone_type=='1':
                    FN_T1+=1
                elif clone_type=='2c':
                    FN_T2c+=1
                elif clone_type=='2b':
                    FN_T2b+=1
                elif clone_type=='VST3':
                    FN_VST3+=1
                elif clone_type=='ST3':
                    FN_ST3+=1
                elif clone_type=='MT3':
                    FN_MT3+=1
                elif clone_type=='WT3_T4':
                    FN_WT3_T4+=1
                else:
                    FN_otherType+=1
                    pass
                pass
        elif value==0:
            if detectionResult==1:
                FP+=1
            else:
                TN+=1
        else:
            pass
        index=index+1
        if(index%10000==0):
            print('index:%d'%index)
        pass
    if ((TP_otherType!=0)or(FN_otherType!=0)):
        print('error 0')
        return 
    if((TP+TN+FP+FN+1)!=index):
        print('error 1')
        return 
    if(((TP_T1+TP_T2c+TP_T2b+TP_VST3+TP_ST3+TP_MT3+TP_WT3_T4)!=TP)or((FN_T1+FN_T2c+FN_T2b+FN_VST3+FN_ST3+FN_MT3+FN_WT3_T4)!=FN)):
        print('error 2')
        return 
    print ('Done')
    
    all_data_dict.clear()
    del(all_data_dict)
    all_idMapline_dict.clear()
    del(all_idMapline_dict)
    all_id_pair_cloneType_dict.clear()
    del(all_id_pair_cloneType_dict)
    detected_clone_pairs_lineInCorpus.clear()
    del(detected_clone_pairs_lineInCorpus)
    
    return TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4
     

def analyse_experimentID3(distance_pkl,metrics_pkl):
    yuzhi=0.0
    jiange=0.05
    result={}
    ###########计算指标
    TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data_experimentID3(distance_pkl,yuzhi)
    R = cal_recall(TP, FN)
    
#    print('(TP, TN, FP, FN)')
#    print(TP, TN, FP, FN)
#    print('(acc, P, R, F1)')
#    print(acc, P, R, F1)
    result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
    ###########计算指标
    
#    while(R<0.99): #只要查全率没有达到1，我们就不断的去增大阈值。当得到最大的阈值以后，我们再重新计算一次，对阈值的增长区间进行合理的划分。不然的话，我们不知道一次应该将阈值增长多少。
    while(yuzhi<1.7):#这里不能再用查全率去衡量。因为，我们在整个语料库上做实验，而不是在标记数据集上做实验。查全率做不到100%。如果是在标记数据集上做实验，阈值越大，查全率会越高。
        #但是，由于是在整个语料库上做实验，而且是处理NSG算法返回之后的结果，所以，即使阈值增到更大，也不可能检测到更多的克隆对。因为阈值不影响我们对近似近邻的检索。
        #因为，如果我们用阈值去干涉NSG算法，假设阈值很大，那么就没有必要用近似近邻了。阈值很大，就是检索全部了。
        print('yuzhi{},Recall:{}'.format(yuzhi,R))
#        jiange=jiange*2#加快到达Recall为1的过程。
        yuzhi=yuzhi+jiange
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data_experimentID3(distance_pkl,yuzhi)
        result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
        pass
    ######最后，阈值和指标就在result列表中了。
    print('yuzhi{},Recall:{}'.format(yuzhi,R))
    save_to_pkl(result,metrics_pkl)
    return result
#    xxxx=read_from_pkl(path)    
    
def experimentID_3():
    
    ###############################
    #第一步，我们根据NSG算法的参数设置，返回检测到的克隆对。计算出整个bcd_reduced上面的克隆对。然后与直接在标签数据集上计算出的进行比较。比较两者的相关指标。
    #设定整个bcd_reduced上面的是一个元组的列表。
#    x=(0,0)
#    y=(1,1)
#    detected_clone_pairs_lineInCorpus_trainedRoot=[] #我们检测到的，保存的是在语料库中的行号，而不是在BigCloneBench中的函数ID号。
#    detected_clone_pairs_lineInCorpus_trainedRoot.append(x)
#    detected_clone_pairs_lineInCorpus_trainedRoot.append(y)
#    detected_clone_pairs_lineInCorpus_trainedMean=[] #我们检测到的，保存的是在语料库中的行号，而不是在BigCloneBench中的函数ID号。
#    detected_clone_pairs_lineInCorpus_trainedMean.append(x)
#    detected_clone_pairs_lineInCorpus_trainedMean.append(y)
    ############detected_clone_pairs_lineInCorpus_trainedRoot.xiaojiepkl是我们在虚拟机中用NSG算法，并且是用python2保存
    search_K_inNSG=100;#返回的近邻个数
    detected_clone_pairs_lineInCorpus_trainedRoot_Path='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedRoot'+'_'+str(search_K_inNSG)+'.xiaojiepkl'
    detected_clone_pairs_lineInCorpus_trainedMean_Path='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedMean'+'_'+str(search_K_inNSG)+'.xiaojiepkl'

    ############detected_clone_pairs_lineInCorpus_trainedRoot.xiaojiepkl是我们在虚拟机中用NSG算法，并且是用python2保存
    ###############################
    #第二步，我们计算出这些克隆对的距离文件
    fullCorpusLine_Map_Vector_trainedroot='./vector/fullCorpusLine_Map_Vector_trainedroot.xiaojiepkl'
    fullCorpusLine_Map_Vector_trainedmean='./vector/fullCorpusLine_Map_Vector_trainedmean.xiaojiepkl'
    detected_LineInfullCorpus_Pair_Map_Distance_trainedroot='./vector/detected_LineInfullCorpus_Pair_Map_Distance_trainedroot.xiaojiepkl'
    detected_LineInfullCorpus_Pair_Map_Distance_trainedmean='./vector/detected_LineInfullCorpus_Pair_Map_Distance_trainedmean.xiaojiepkl'
    #如果之前已经计算过，下面就要注释掉。
#    save_distance_file_experimentID3('./vector/detected_LineInfullCorpus_Pair_Map_Distance_trainedroot.xiaojie',detected_clone_pairs_lineInCorpus_trainedRoot_Path,fullCorpusLine_Map_Vector_trainedroot,detected_LineInfullCorpus_Pair_Map_Distance_trainedroot)
#    save_distance_file_experimentID3('./vector/detected_LineInfullCorpus_Pair_Map_Distance_trainedmean.xiaojie',detected_clone_pairs_lineInCorpus_trainedMean_Path,fullCorpusLine_Map_Vector_trainedmean,detected_LineInfullCorpus_Pair_Map_Distance_trainedmean)

    ###############################
    #第三步，计算指标
    metrices_onFullCorpusWithNSG_trained_root_pkl='./result/metrices_onFullCorpusWithNSG_trained_root.xiaojiepkl'
    metrices_onFullCorpusWithNSG_trained_mean_pkl='./result/metrices_onFullCorpusWithNSG_trained_mean.xiaojiepkl'
    #####读取pkl文件中的距离，并且根据阈值，去计算相关指标,存入相关pkl文件
#    analyse_experimentID3(distance_pkl=detected_LineInfullCorpus_Pair_Map_Distance_trainedroot,metrics_pkl=metrices_onFullCorpusWithNSG_trained_root_pkl)
#    analyse_experimentID3(distance_pkl=detected_LineInfullCorpus_Pair_Map_Distance_trainedmean,metrics_pkl=metrices_onFullCorpusWithNSG_trained_mean_pkl)
    ##保存到pkl文件以后，后续可以直接读取
    result_onFullCorpusWithNSG_trained_root=read_from_pkl(metrices_onFullCorpusWithNSG_trained_root_pkl)
    result_onFullCorpusWithNSG_trained_mean=read_from_pkl(metrices_onFullCorpusWithNSG_trained_mean_pkl)
    ##########绘制PR曲线
    ###取出没有NSG的
    metrices_trained_mean_pkl='./result/metrices_trained_mean.xiaojiepkl'
    metrices_trained_root_pkl='./result/metrices_trained_root.xiaojiepkl'
    result_trained_mean=read_from_pkl(metrices_trained_mean_pkl)
    result_trained_root=read_from_pkl(metrices_trained_root_pkl)
    
    
    
    ##NSG trained_mean模型 
    precision_trained_mean_withNSG=[]
    recall_trained_mean_withNSG=[]
    yuzhi__trained_mean_withNSG=[]
    FPR_trained_mean_withNSG=[]
    recall_T1_trained_mean_withNSG=[]
    recall_T2_trained_mean_withNSG=[]
    recall_VST3_trained_mean_withNSG=[]
    recall_ST3_trained_mean_withNSG=[]
    recall_MT3_trained_mean_withNSG=[]
    recall_WT3_T4_trained_mean_withNSG=[]
    for obj in result_onFullCorpusWithNSG_trained_mean.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_trained_mean_withNSG.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_trained_mean_withNSG.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_trained_mean_withNSG.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_trained_mean_withNSG.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_trained_mean_withNSG.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_trained_mean_withNSG.append(recall_WT3_T4)
        
        precision_trained_mean_withNSG.append(p)
        recall_trained_mean_withNSG.append(r)
        yuzhi__trained_mean_withNSG.append(key)
        FPR_trained_mean_withNSG.append(FPR)
##NSG trained_root模型 
    precision_trained_root_withNSG=[]
    recall_trained_root_withNSG=[]
    yuzhi__trained_root_withNSG=[]
    FPR_trained_root_withNSG=[]
    recall_T1_trained_root_withNSG=[]
    recall_T2_trained_root_withNSG=[]
    recall_VST3_trained_root_withNSG=[]
    recall_ST3_trained_root_withNSG=[]
    recall_MT3_trained_root_withNSG=[]
    recall_WT3_T4_trained_root_withNSG=[]
    for obj in result_onFullCorpusWithNSG_trained_root.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_trained_root_withNSG.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_trained_root_withNSG.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_trained_root_withNSG.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_trained_root_withNSG.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_trained_root_withNSG.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_trained_root_withNSG.append(recall_WT3_T4)
        
        precision_trained_root_withNSG.append(p)
        recall_trained_root_withNSG.append(r)
        yuzhi__trained_root_withNSG.append(key)
        FPR_trained_root_withNSG.append(FPR)


    ###trained_mean模型，没有NSG
    precision_trained_mean=[]
    recall_trained_mean=[]
    yuzhi__trained_mean=[]
    FPR_trained_mean=[]
    recall_T1_trained_mean=[]
    recall_T2_trained_mean=[]
    recall_VST3_trained_mean=[]
    recall_ST3_trained_mean=[]
    recall_MT3_trained_mean=[]
    recall_WT3_T4_trained_mean=[]
    for obj in result_trained_mean.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_trained_mean.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_trained_mean.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_trained_mean.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_trained_mean.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_trained_mean.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_trained_mean.append(recall_WT3_T4)
        
        precision_trained_mean.append(p)
        recall_trained_mean.append(r)
        yuzhi__trained_mean.append(key)
        FPR_trained_mean.append(FPR)
    ###trained_root模型，没有NSG
    precision_trained_root=[]
    recall_trained_root=[]
    yuzhi__trained_root=[]
    FPR_trained_root=[]
    recall_T1_trained_root=[]
    recall_T2_trained_root=[]
    recall_VST3_trained_root=[]
    recall_ST3_trained_root=[]
    recall_MT3_trained_root=[]
    recall_WT3_T4_trained_root=[]
    for obj in result_trained_root.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_trained_root.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_trained_root.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_trained_root.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_trained_root.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_trained_root.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_trained_root.append(recall_WT3_T4)
        
        precision_trained_root.append(p)
        recall_trained_root.append(r)
        yuzhi__trained_root.append(key)
        FPR_trained_root.append(FPR)    
    
    plt.figure(1) # 创建图表3
    plt.title('Recall/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=1, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi__trained_root,recall_trained_root,label='trained_root')
    plt.plot(yuzhi__trained_mean,recall_trained_mean,label='trained_mean')
    plt.legend()
    plt.show()    
    
    plt.figure(2) # 创建图表3
    fig=plt.figure(num=2, figsize=(15, 8),dpi=80) 
    plt.title('Recall/threshold')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
#    plt.plot(yuzhi__trained_root_withNSG,recall_T1_trained_root_withNSG,'b--',label='recall_T1_trained_root_withNSG')
#    plt.plot(yuzhi__trained_root_withNSG,recall_T2_trained_root_withNSG,label='recall_T2_trained_root_withNSG')
    plt.plot(yuzhi__trained_root_withNSG,recall_VST3_trained_root_withNSG,label='recall_VST3_trained_root_withNSG')
#    plt.plot(yuzhi__trained_root_withNSG,recall_ST3_trained_root_withNSG,label='recall_ST3_trained_root_withNSG')
#    plt.plot(yuzhi__trained_root_withNSG,recall_MT3_trained_root_withNSG,label='recall_MT3_trained_root_withNSG')
#    plt.plot(yuzhi__trained_root_withNSG,recall_WT3_T4_trained_root_withNSG,label='recall_WT3_T4_trained_root_withNSG')
    
    
#    plt.plot(yuzhi__trained_mean_withNSG,recall_T1_trained_mean_withNSG,label='recall_T1_trained_mean_withNSG')
#    plt.plot(yuzhi__trained_mean_withNSG,recall_T2_trained_mean_withNSG,label='recall_T2_trained_mean_withNSG')
    plt.plot(yuzhi__trained_mean_withNSG,recall_VST3_trained_mean_withNSG,label='recall_VST3_trained_mean_withNSG')
#    plt.plot(yuzhi__trained_mean_withNSG,recall_ST3_trained_mean_withNSG,label='recall_ST3_trained_mean_withNSG')
#    plt.plot(yuzhi__trained_mean_withNSG,recall_MT3_trained_mean_withNSG,label='recall_MT3_trained_mean_withNSG')
#    plt.plot(yuzhi__trained_mean_withNSG,recall_WT3_T4_trained_mean_withNSG,label='recall_WT3_T4_trained_mean_withNSG')
#    
    plt.legend()
    plt.show()
    
    #与不不使用NSG算法的进行对比
    #绘制AUC曲线
    
    plt.figure(3) # 创建图表3
    plt.title('Recall/threshold')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=3, figsize=(15, 8),dpi=80)
    #需要把阈值处理一下
        
        
    plt.plot(yuzhi__trained_root_withNSG[0:11],recall_trained_root_withNSG[0:11],label='trained_root_withNSG')
    plt.plot(yuzhi__trained_root[0:11],recall_trained_root[0:11],label='trained_root')
    plt.plot(yuzhi__trained_mean_withNSG[0:11],recall_trained_mean_withNSG[0:11],label='trained_mean_withNSG')
    plt.plot(yuzhi__trained_mean[0:11],recall_trained_mean[0:11],label='trained_mean')
    plt.legend()
    plt.show()
    ##实在不行，就以表格的形式展现。

def analyse_experimentID4(metrics_pkl,num):
    
    result={}
    yuzhi=0.0
    jiange=0.05
    for i in range(num):
#        if (i!=8):
#            yuzhi+=jiange
#            continue
        print(yuzhi)
        detected_clone_pairs_lineinFullCorpus_WithNSG_WeightedRAE='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_'+str(i)+'.txt'
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data_experimentID4(detected_clone_pairs_lineinFullCorpus_WithNSG_WeightedRAE)
        R = cal_recall(TP, FN)
        result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
        print('yuzhi{},Recall:{}'.format(yuzhi,R))
        yuzhi+=jiange
        
        pass
    save_to_pkl(result,metrics_pkl)
    
    

#    detected_clone_pairs_lineinFullCorpus_WithNSG_WeightedRAE='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_'+str(5)+'_2.txt'
#    TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data_experimentID4(detected_clone_pairs_lineinFullCorpus_WithNSG_WeightedRAE)
#    precision=cal_precision(TP,FP)
#    recall_T1=cal_recall(TP_T1,FN_T1)
#    
#    
#    recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
#    
#    recall_VST3=cal_recall(TP_VST3,FN_VST3)
#    
#    recall_ST3=cal_recall(TP_ST3,FN_ST3)
#    
#    recall_MT3=cal_recall(TP_MT3,FN_MT3)
#    
#    recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
#    
#    print (precision)
#    print (recall_T1)
#    print (recall_T2)
#    print (recall_VST3)
#    print (recall_ST3)
#    print (recall_MT3)
#    print (recall_WT3_T4)
    

def experimentID_6():
    
    ###这个实验与3的区别在于，这里直接是指定阈值后NSG算法返回的克隆检测结果。而不是在NSG返回的结果中进一步进行筛选
    metrices_onFullCorpusWithNSG_WeightedRAE='./result/metrices_onFullCorpusWithNSG_WeightedRAE_differentTherolds.xiaojiepkl'
    analyse_experimentID4(metrices_onFullCorpusWithNSG_WeightedRAE)
def experimentID_4():
    
    ###这个实验与3的区别在于，这里直接是指定阈值后NSG算法返回的克隆检测结果。而不是在NSG返回的结果中进一步进行筛选
    metrices_onFullCorpusWithNSG_PVT='./result/metrices_onFullCorpusWithNSG_PVT_differentTherolds.xiaojiepkl'
#    analyse_experimentID4(metrices_onFullCorpusWithNSG_PVT,num=(14+1))
    #####去计算相关指标,存入相关pkl文件
    ##保存到pkl文件以后，后续可以直接读取
    result_onFullCorpusWithNSG_PVT=read_from_pkl(metrices_onFullCorpusWithNSG_PVT)
    ##########绘制PR曲线
    ###取出没有NSG的
    metrices_PVT_pkl='./result/diffweight/ZHIXIN_ylabx_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
    result_PVT=read_from_pkl(metrices_PVT_pkl)


###PVT模型  result_PVT
    precision_PVT=[]
    recall_PVT=[]
    yuzhi_PVT=[]
    FPR_PVT=[]
    recall_T1_PVT=[]
    recall_T2_PVT=[]
    recall_VST3_PVT=[]
    recall_ST3_PVT=[]
    recall_MT3_PVT=[]
    recall_WT3_T4_PVT=[]
    for obj in result_PVT.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_PVT.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_PVT.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_PVT.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_PVT.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_PVT.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_PVT.append(recall_WT3_T4)
        
        precision_PVT.append(p)
        recall_PVT.append(r)
        yuzhi_PVT.append(key)
        FPR_PVT.append(FPR)
        


###NSG andPVT模型 result_onFullCorpusWithNSG_PVT
    precision_PVT_NSG=[]
    recall_PVT_NSG=[]
    yuzhi_PVT_NSG=[]
    FPR_PVT_NSG=[]
    recall_T1_PVT_NSG=[]
    recall_T2_PVT_NSG=[]
    recall_VST3_PVT_NSG=[]
    recall_ST3_PVT_NSG=[]
    recall_MT3_PVT_NSG=[]
    recall_WT3_T4_PVT_NSG=[]
    for obj in result_onFullCorpusWithNSG_PVT.items():
        key,value=obj
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4=value
        p=cal_precision(TP, FP)
        r=cal_recall(TP, FN)
        FPR=cal_FPR(FP,TN)
        
        recall_T1=cal_recall(TP_T1,FN_T1)
        recall_T1_PVT_NSG.append(recall_T1)
        recall_T2=cal_recall(TP_T2c+TP_T2b,FN_T2c+FN_T2b)
        recall_T2_PVT_NSG.append(recall_T2)
        recall_VST3=cal_recall(TP_VST3,FN_VST3)
        recall_VST3_PVT_NSG.append(recall_VST3)
        recall_ST3=cal_recall(TP_ST3,FN_ST3)
        recall_ST3_PVT_NSG.append(recall_ST3)
        recall_MT3=cal_recall(TP_MT3,FN_MT3)
        recall_MT3_PVT_NSG.append(recall_MT3)
        recall_WT3_T4=cal_recall(TP_WT3_T4,FN_WT3_T4)
        recall_WT3_T4_PVT_NSG.append(recall_WT3_T4)
        
        precision_PVT_NSG.append(p)
        recall_PVT_NSG.append(r)
        yuzhi_PVT_NSG.append(key)
        FPR_PVT_NSG.append(FPR)


    ###绘制
    plt.figure(1) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')
    
    fig = plt.figure(num=1, figsize=(15, 8),dpi=80) 
    plt.plot(recall_PVT_NSG,precision_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(recall_PVT,precision_PVT,'m.-',label='PVT')
    plt.legend()
    plt.show()


    
    
    
    
    
    
    
    plt.figure(2) # 创建图表1
    plt.title('Recall/threshold Curve')# give plot a title
    plt.xlabel('threshold')# make axis labels
    plt.ylabel('Recall')
    
    fig = plt.figure(num=2, figsize=(15, 8),dpi=80) 
    plt.plot(yuzhi_PVT_NSG,recall_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(yuzhi_PVT,recall_PVT,'m.-',label='PVT')
    plt.legend()
    plt.show()
    
    #绘制AUC曲线
    
    plt.figure(3) # 创建图表3
    plt.title('TPR/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=3, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_PVT,'m.-',label='PVT')
    plt.legend()
    
            ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    
    plt.show()
    
    plt.figure(4) # 创建图表3
    plt.title('TPR_T1/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    fig = plt.figure(num=4, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_T1_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_T1_PVT,'m.-',label='PVT')
    plt.legend()
        
    ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    ###设置坐标轴
    plt.show()
    
    plt.figure(5) # 创建图表3
    plt.title('TPR_T2/FPR curve ')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=5, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_T2_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_T2_PVT,'m.-',label='PVT')
    plt.legend()
    
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    
    plt.show()
        
 
    
    plt.figure(6) # 创建图表3
    plt.title('TPR_VST3/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=6, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_VST3_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_VST3_PVT,'m.-',label='PVT')
    plt.legend()
    
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    plt.show()
   

    
    plt.figure(7) # 创建图表3
    plt.title('TPR_ST3/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    plt.ylim((0,1.1))
    
    fig = plt.figure(num=7, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_ST3_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_ST3_PVT,'m.-',label='PVT')
    plt.legend()
    
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    
    plt.show()    
    
    
    plt.figure(8) # 创建图表3
    plt.title('TPR_MT3/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=8, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_MT3_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_MT3_PVT,'m.-',label='PVT')
    plt.legend()
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    
    plt.show()    
    
    plt.figure(9) # 创建图表3
    plt.title('TPR_T4/FPR curve')# give plot a title
    plt.xlabel('FPR')# make axis labels
    plt.ylabel('TPR')
    
    fig = plt.figure(num=9, figsize=(15, 8),dpi=80) 
    plt.plot(FPR_PVT_NSG,recall_WT3_T4_PVT_NSG,'c*-',label='PVT_NSG')
    plt.plot(FPR_PVT,recall_WT3_T4_PVT,'m.-',label='PVT')
    plt.legend()
        ###设置坐标轴
    x_major_locator=MultipleLocator(0.1)#间隔为0.1
    y_major_locator=MultipleLocator(0.1)#间隔为0.1
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    
    plt.show()   
    
    
    pass        

def experimentID_5():#输出bigCloneBench上各个对，在root和mean下的各个vector之间的距离
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数
    all_data_dict = read_from_pkl(bigCloneBenchIdPair_PKL_Path)
    
    all_idMapline_pkl = './SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
    all_idMapline_dict= read_from_pkl(all_idMapline_pkl)
    
    #读取用于评测的所有id_pair对应的距离，根据模型指定


    fullCorpusLine_Map_Vector_trainedroot='./vector/fullCorpusLine_Map_Vector_trainedroot.xiaojiepkl'
    fullCorpusLine_Map_Vector_trainedmean='./vector/fullCorpusLine_Map_Vector_trainedmean.xiaojiepkl'
    fullCorpusLine_Map_Vector_dict_trainedroot=read_from_pkl(fullCorpusLine_Map_Vector_trainedroot)
    fullCorpusLine_Map_Vector_dict_trainedmean=read_from_pkl(fullCorpusLine_Map_Vector_trainedmean)
    all_pairs_id_Map_RootAndMean_Distance_pklpath='./vector/all_pairs_id_Map_RootAndMean_Distance.xiaojiepkl'
    all_pairs_id_Map_RootAndMean_Distance={}
    all_pairs_id_Map_RootAndMean_Distance_txtpath='./vector/all_pairs_id_Map_RootAndMean_Distance.txt'
    ########################################################################
    ########下面这个过程主要是验证。
#    idMapVector_PKL_Path_trainedMean='./vector/trainedModel_IdinBigCloneBench_Map_MeanVector.xiaojiepkl'
#    idMapVector_PKL_Path_trainedRoot='./vector/trainedModel_IdinBigCloneBench_Map_RootVector.xiaojiepkl'
#
#    
#    idMapVector_root=read_from_pkl(idMapVector_PKL_Path_trainedRoot)
#    idMapVector_mean=read_from_pkl(idMapVector_PKL_Path_trainedMean)
    ########
    ########################################################################
    with open(all_pairs_id_Map_RootAndMean_Distance_txtpath, 'w') as fv:
        fv.write('IdInBigCloneBench_1'+','+'IdInBigCloneBench_2'+','+'root_distance'+','+'mean_distance'+'\n')
        index=0
        for obj in all_data_dict.items(): #遍历bigCloneBench的所有ID对
    #        print((index))
            ####测试用,执行时注释掉。
    #        if(index==1000):
    #            break
            ####测试用
            key,value=obj #key是个二元组，保存的是bigCloneBench的函数ID之间的对应关系。value是标签。
            id1=str(key[0])
            id2=str(key[1])
            line1=all_idMapline_dict[id1]
            line2=all_idMapline_dict[id2]
            vector1_root=fullCorpusLine_Map_Vector_dict_trainedroot[line1]
            vector2_root=fullCorpusLine_Map_Vector_dict_trainedroot[line2]
            vector1_mean=fullCorpusLine_Map_Vector_dict_trainedmean[line1]
            vector2_mean=fullCorpusLine_Map_Vector_dict_trainedmean[line2]
#            vector1_root_2=idMapVector_root[id1]
#            vector1_mean_2=idMapVector_mean[id1]
#            vector2_root_2=idMapVector_root[id2]
#            vector2_mean_2=idMapVector_mean[id2]
#            print(vector1_root)
#            print(vector1_root_2)
#            print(vector1_mean)
#            print(vector1_mean_2)
            distance_root=np.linalg.norm(vector1_root-vector2_root)
            distance_mean=np.linalg.norm(vector1_mean-vector2_mean)
            distance_value=(distance_root,distance_mean)
            all_pairs_id_Map_RootAndMean_Distance[key]=distance_value
            fv.write(str(line1)+','+str(line2)+','+str(distance_root)+','+str(distance_mean)+'\n')
            if(index%10000==0):
                print ('index:%d'%index)
            index=index+1
            pass
    all_data_dict.clear()
    del(all_data_dict)
    all_idMapline_dict.clear()
    del(all_idMapline_dict)
    fullCorpusLine_Map_Vector_dict_trainedroot.clear()
    del(fullCorpusLine_Map_Vector_dict_trainedroot)
    fullCorpusLine_Map_Vector_dict_trainedmean.clear()
    del(fullCorpusLine_Map_Vector_dict_trainedmean)
    save_to_pkl(all_pairs_id_Map_RootAndMean_Distance,all_pairs_id_Map_RootAndMean_Distance_pklpath)
    
    all_pairs_id_Map_RootAndMean_Distance.clear()
    del(all_pairs_id_Map_RootAndMean_Distance)
    pass
    
    
    
if __name__ == "__main__":
    
#    experimentID_1()    
#    experimentID_3()
    experimentID_4()
#    experimentID_5()
#    experimentID_6()