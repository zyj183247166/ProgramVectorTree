import pickle
import random
#coding:utf-8
import matplotlib
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
def get_basic_data_chabie(id_pair_map_distance_dict_weighted_RAE_weightedAndCengci,id_pair_map_distance_dict_traditional_RAE_root,yuzhi):
    #读取用于评测的所有id_pair
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数
    all_data_dict = read_from_pkl(bigCloneBenchIdPair_PKL_Path)
    #读取用于评测的所有id_pair对应的clone_pair
    clone_type_pkl = './SplitDataSet/data/all_clone_id_pair_cloneType_XIAOJIE.pkl'
    all_id_pair_cloneType_dict=read_from_pkl(clone_type_pkl)
    #读取用于评测的所有id_pair对应的距离，根据模型指定
    id_pair_map_distance_dict_weighted_RAE_weightedAndCengci=read_from_pkl(id_pair_map_distance_dict_weighted_RAE_weightedAndCengci)
    id_pair_map_distance_dict=id_pair_map_distance_dict_weighted_RAE_weightedAndCengci
    id_pair_map_distance_dict_traditional_RAE_root=read_from_pkl(id_pair_map_distance_dict_traditional_RAE_root)
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
    write_file='./weDonotDetectButTraditionalDetect_yuzhi_'+(str)(yuzhi)+'.txt'
    with open(write_file,'w',encoding='utf-8') as fv:
        fv.write('yuzhi_'+str(yuzhi)+'\n')
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
    #                    print (keyyy)
                    elif clone_type=='2c':
                        FN_T2c+=1
    #                    print (keyyy)
                        ###如果我们的weighted模型没能检测到，而traditional RAE_ROOT模型却检测到了。我们输出这些克隆对
                        distance2=id_pair_map_distance_dict_traditional_RAE_root[key]
                        if(distance2<=yuzhi):
                            fv.write('key_1:'+str(id1)+','+'key_2:'+str(id2)+'\n')
                            print(keyyy)
                    elif clone_type=='2b':
                        FN_T2b+=1
    #                    print (keyyy)
                        distance2=id_pair_map_distance_dict_traditional_RAE_root[key]
                        if(distance2<=yuzhi):
                            fv.write('key_1:'+str(id1)+','+'key_2:'+str(id2)+'\n')
                            print(keyyy)
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
#                    print (keyyy)
                elif clone_type=='2c':
                    FN_T2c+=1
#                    print (keyyy)
                elif clone_type=='2b':
                    FN_T2b+=1
#                    print (keyyy)
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
def analyse_chabie_report():#差别分析
    
    distance_PKL_weighted_RAE_weightedAndCengci='./distance/weighted_RAE_IDPairinBigCloneBench_Map_Distance_weightedAndCengci.xiaojiepkl'
    distance_PKL_traditional_RAE_root='./distance/traditional_RAE_IdPairinBigCloneBench_Map_Distance_Root.xiaojiepkl'
    yuzhi=0.0
    jiange=0.05
    for i in range(5):
        get_basic_data_chabie(distance_PKL_weighted_RAE_weightedAndCengci,distance_PKL_traditional_RAE_root,yuzhi)
        yuzhi=yuzhi+jiange
    
    
def analyse(distance_pkl,metrics_pkl):
    yuzhi=0.0
    
    ####关于这个jiange的设置，应当取决于distance_pkl的量纲
#    jiange=0.05
    ####
    ###下面用于观察距离范围
#    id_pair_map_distance_dict=read_from_pkl(distance_pkl)
#    for obj in id_pair_map_distance_dict.items():
#        print (obj[1])
    jiange=0.05
    result={}
    ###########计算指标
    TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data(distance_pkl,yuzhi)
    R = cal_recall(TP, FN)
    FPR=cal_FPR(FP, TN)
#    print('(TP, TN, FP, FN)')
#    print(TP, TN, FP, FN)
#    print('(acc, P, R, F1)')
#    print(acc, P, R, F1)
    result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
    
    ###########计算指标
    while(R<0.99 or FPR<0.99): #只要查全率没有达到1，我们就不断的去增大阈值。当得到最大的阈值以后，我们再重新计算一次，对阈值的增长区间进行合理的划分。不然的话，我们不知道一次应该将阈值增长多少。
        
        print('yuzhi{},Recall:{},FPR:{}'.format(yuzhi,R,FPR))
#        jiange=jiange*2#加快到达Recall为1的过程。
        yuzhi=yuzhi+jiange
        TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4= get_basic_data(distance_pkl,yuzhi)
        R = cal_recall(TP, FN)
        FPR=cal_FPR(FP, TN)
        result[yuzhi]=(TP,TN,FP,FN,TP_T1,TP_T2c,TP_T2b,TP_VST3,TP_ST3,TP_MT3,TP_WT3_T4,FN_T1,FN_T2c,FN_T2b,FN_VST3,FN_ST3,FN_MT3,FN_WT3_T4)
        if (R>=0.99):
            jiange=1
        pass
    ######最后，阈值和指标就在result列表中了。
    print('yuzhi{},Recall:{},FPR:{}'.format(yuzhi,R,FPR))
    save_to_pkl(result,metrics_pkl)
    return result
#    xxxx=read_from_pkl(path)
def experimentID_1():
    #先把模型上计算的向量之间的距离计算出来并保存
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数（被他们错误标记），以及去重

#    idMapVector_PKL_Path_trainedRoot='./vector/tkqnm_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
#    idMapVector_PKL_Path_trainedWeighted='./vector/gteon_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl'
#    idMapVector_PKL_Path_trainedWeighted_root='./vector/psqby_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
##    idMapVector_PKL_Path_trainedWeighted_meanAndRootMost='./vector/yalfo_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_meanAndRootMost.xiaojiepkl'
#    idMapVector_PKL_Path_trainedWeighted_meanAndRootMost='./vector/mauvn_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_meanAndRootMost.xiaojiepkl'
##    idMapVector_PKL_Path_trainedWeighted_weightedAndCengci='./vector/wxart_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl'
#    idMapVector_PKL_Path_trainedWeighted_weightedAndCengci='./vector/sfgiy_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl'
    ##########
#    id_pair_map_distance_pkl_trainedRoot='./distance/traditional_RAE_IdPairinBigCloneBench_Map_Distance_Root.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted='./distance/weighted_RAE_IDPairinBigCloneBench_Map_Distance_weighted.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_root='./distance/weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
##    id_pair_map_distance_pkl_trainedWeighted_meanAndRootMost='./distance/weighted_RAE_IDPairinBigCloneBench_Map_Distance_meanAndRootMost.xiaojiepkl'
##    id_pair_map_distance_pkl_trainedWeighted_meanAndRootMost='./distance/weighted_RAE_IDPairinBigCloneBench_Map_Distance_meanAndRootMost2.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci='./distance/weighted_RAE_IDPairinBigCloneBench_Map_Distance_weightedAndCengci.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
    #####先根据模型去计算距离，然后保存到pkl文件中去
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedRoot,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedRoot)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_meanAndRootMost,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_meanAndRootMost)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_weightedAndCengci,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci)
    ##保存到pkl文件以后，后续可以直接读取
#    id_pair_map_distance_randomRoot=read_from_pkl(id_pair_map_distance_pkl_randomRoot)
#    id_pair_map_distance_randomMean=read_from_pkl(id_pair_map_distance_pkl_randomMean)
    #####先根据模型去计算距离，然后保存到pkl文件中去
#     metrices_traditionalRAE_Root_pkl='./result/metrices_traditionalRAE_root.xiaojiepkl'
##    metrices_weightedRAE_Weighted_pkl='./result/metrices_weightedRAE_weighted.xiaojiepkl'
#    metrices_weightedRAE_root_pkl='./result/metrices_weightedRAE_root.xiaojiepkl'
#    metrices_weightedRAE_meanAndRootMost_pkl='./result/metrices_weightedRAE_meanAndRootMost2.xiaojiepkl'
##    metrices_weightedRAE_trainedWeighted_weightedAndCengci='./result/metrices_weightedRAE_trainedWeighted_weightedAndCengci.xiaojiepkl'
#    metrices_weightedRAE_trainedWeighted_weightedAndCengci='./result/metrices_weightedRAE_trainedWeighted_weightedAndCengci.xiaojiepkl'
#    #####读取pkl文件中的距离，并且根据阈值，去计算相关指标,存入相关pkl文件
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedRoot,metrics_pkl=metrices_traditionalRAE_Root_pkl)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted,metrics_pkl=metrices_weightedRAE_Weighted_pkl)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_meanAndRootMost,metrics_pkl=metrices_weightedRAE_meanAndRootMost_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci,metrics_pkl=metrices_weightedRAE_trainedWeighted_weightedAndCengci)
#    

#####另一种权重模型，即计算父亲节点交叉熵的时候，考虑到子节点的权重。
#    idMapVector_PKL_Path_trainedWeighted_root='./vector/childparentweight/crzix_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
#    idMapVector_PKL_Path_trainedWeighted_mean='./vector/childparentweight/mzduw_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl'
#    idMapVector_PKL_Path_trainedWeighted_weightedAndCengci='./vector/childparentweight/vifhz_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl'
#    
#    id_pair_map_distance_pkl_trainedWeighted_root='./distance/childparentweight/weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_mean='./distance/childparentweight/weighted_RAE_IDPairinBigCloneBench_Map_Distance_mean.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci='./distance/childparentweight/weighted_RAE_IDPairinBigCloneBench_Map_Distance_weightedAndCengci.xiaojiepkl'

#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_mean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean)
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_weightedAndCengci,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci)    
    
    
#    metrices_weightedRAE_root_pkl='./result/childparentweight/metrices_weightedRAE_root.xiaojiepkl'
#    metrices_weightedRAE_mean_pkl='./result/childparentweight/metrices_weightedRAE_mean.xiaojiepkl'
#    metrices_weightedRAE_weightedAndCengci='./result/childparentweight/metrices_weightedRAE_trainedWeighted_weightedAndCengci.xiaojiepkl'
    
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean,metrics_pkl=metrices_weightedRAE_mean_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci,metrics_pkl=metrices_weightedRAE_weightedAndCengci)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_meanAndRootMost,metrics_pkl=metrices_weightedRAE_meanAndRootMost_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_weightedAndCengci,metrics_pkl=metrices_weightedRAE_trainedWeighted_weightedAndCengci)
#   
#    ./vector/childparentweight/iwtzf_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl
#./vector/childparentweight/cweav_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl
#./vector/childparentweight/uqyag_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl
#./vector/childparentweight/hqbym_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_meanAndRootMost.xiaojiepkl
#./vector/childparentweight/fcznt_using_Weigthed_RAE_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl
    ####计算传统RAE模型
    
    


#    idMapVector_PKL_Path_traditionalRAE_root='./vector/oacnr_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
#    idMapVector_PKL_Path_traditionalRAE_mean='./vector/jtzev_BigCloneBench_traditionalRAE_ID_Map_Vector_mean.xiaojiepkl'
#    #########
#    id_pair_map_distance_pkl_traditionalRAE_root='./distance/20190622traditional_RAE_IdPairinBigCloneBench_Map_Distance_Root.xiaojiepkl'
#    id_pair_map_distance_pkl_traditionalRAE_mean='./distance/20190622traditional_RAE_IDPairinBigCloneBench_Map_Distance_weighted.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_traditionalRAE_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_traditionalRAE_root)
#    ####先根据模型去计算距离，然后保存到pkl文件中去
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_traditionalRAE_mean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_traditionalRAE_mean)
#    metrices_weightedRAE_root_pkl='./result/20190622traditional_RAE_metrices_weightedRAE_root.xiaojiepkl'
#    metrices_weightedRAE_mean_pkl='./result/20190622traditional_RAE_metrices_weightedRAE_mean.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_traditionalRAE_root,metrics_pkl=metrices_weightedRAE_root_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_traditionalRAE_mean,metrics_pkl=metrices_weightedRAE_mean_pkl)
    #保存到pkl文件以后，后续可以直接读取

    
    ####计算加权模型，随机化参数，不训练
#    idMapVector_PKL_Path_trainedWeighted_root='./vector/childparentweight/ctvgn_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
##    idMapVector_PKL_Path_trainedWeighted_mean='./vector/childparentweight/kglub_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_root='./distance/childparentweight/ctvgn_weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
##    id_pair_map_distance_pkl_trainedWeighted_mean='./distance/childparentweight/kglub_weighted_RAE_IDPairinBigCloneBench_Map_Distance_mean.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
##    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_mean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean)
#    metrices_weightedRAE_root_pkl='./result/childparentweight/ctvgn_metrices_weightedRAE_root.xiaojiepkl'
##    metrices_weightedRAE_mean_pkl='./result/childparentweight/cweav_metrices_weightedRAE_mean.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean,metrics_pkl=metrices_weightedRAE_mean_pkl)
 
    ####计算加权模型，训练后模型
#    idMapVector_PKL_Path_trainedWeighted_root='./vector/childparentweight/tlped_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
##    idMapVector_PKL_Path_trainedWeighted_mean='./vector/childparentweight/kglub_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_root='./distance/childparentweight/tlped_weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
##    id_pair_map_distance_pkl_trainedWeighted_mean='./distance/childparentweight/kglub_weighted_RAE_IDPairinBigCloneBench_Map_Distance_mean.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
##    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_mean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean)
#    metrices_weightedRAE_root_pkl='./result/childparentweight/tlped_metrices_weightedRAE_root.xiaojiepkl'
##    metrices_weightedRAE_mean_pkl='./result/childparentweight/cweav_metrices_weightedRAE_mean.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean,metrics_pkl=metrices_weightedRAE_mean_pkl)

    ####计算diff-weighted模型，各个模型下的距离
#    xiaojie_lists=['diffW_random_quyko']
#    for i,item in enumerate(xiaojie_lists):
#        idMapVector_root='./vector/diffweight/'+item+'_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
#        id_pair_map_distance_pkl_root='./distance/diffweight/'+item+'_weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
#        save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_root)
#        metrices_root_pkl='./result/diffweight/'+item+'_metrices_weightedRAE_root.xiaojiepkl'
#        analyse(distance_pkl=id_pair_map_distance_pkl_root,metrics_pkl=metrices_root_pkl)
    
    ####计算weighted模型，各个模型下的距离
#    ./vector/childparentweight/weighted_epoch_29_jwytx_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl
##    xiaojie_lists=['apywz']
#    xiaojie_lists=['weighted_epoch_29_jwytx']
#    for i,item in enumerate(xiaojie_lists):
#        idMapVector_root='./vector/childparentweight/'+item+'_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
#        id_pair_map_distance_pkl_root='./distance/childparentweight/'+item+'_weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
##        save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_root)
#        metrices_root_pkl='./result/childparentweight/'+item+'_metrices_weightedRAE_root.xiaojiepkl'
#        analyse(distance_pkl=id_pair_map_distance_pkl_root,metrics_pkl=metrices_root_pkl)
    
    
    ######实验最终章
    ######随机模型
    
#    idMapVector_root='./vector/random_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
#    id_pair_map_distance_pkl_root='./distance/random_BigCloneBench_traditionalRAE_distance_root.xiaojiepkl'
##    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_root)
#    metrices_root_pkl='./result/random_BigCloneBench_traditionalRAE_metrics_root.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_root,metrics_pkl=metrices_root_pkl)  
    ######unweighted模型
#    idMapVector_root='./vector/unweighted_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
#    id_pair_map_distance_pkl_root='./distance/unweighted_BigCloneBench_traditionalRAE_distance_root.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_root)
#    metrices_root_pkl='./result/unweighted_BigCloneBench_traditionalRAE_metrics_root.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_root,metrics_pkl=metrices_root_pkl)  
        
    
    ######weighted模型
#    idMapVector_root='./vector/childparentweight/enoyz_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl'
#    id_pair_map_distance_pkl_root='./distance/weighted_BigCloneBench_traditionalRAE_distance_TF-IDF.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_root)
#    metrices_root_pkl='./result/weighted_BigCloneBench_traditionalRAE_metrics_root_TF-IDF.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_root,metrics_pkl=metrices_root_pkl) 
#    ./vector/childparentweight/enoyz_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl
#    ./vector/childparentweight/enoyz_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl
    ######实验最终章    
    ##########质心模型
#    idMapVector_PKL_Path_trainedWeighted_root='./vector/diffweight/ZHIXIN_zwimn_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_root='./distance/diffweight//ZHIXIN_zwimn_weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
#    metrices_weightedRAE_root_pkl='./result/diffweight//ZHIXIN_zwimn_metrices_weightedRAE_root.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
    
    
    #########程序向量树
    idMapVector_PKL_Path_trainedWeighted_root='./vector/diffweight/ZHIXIN_ylabx_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
    id_pair_map_distance_pkl_trainedWeighted_root='./distance/diffweight/ZHIXIN_ylabx_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
    metrices_weightedRAE_root_pkl='./result/diffweight/ZHIXIN_ylabx_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
        


    ####计算加权模型，训练后模型，含TF-IDF
#    idMapVector_PKL_Path_trainedWeighted_root='./vector/childparentweight/wRAETF_IDF_uiyvq_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'
##    idMapVector_PKL_Path_trainedWeighted_mean='./vector/childparentweight/kglub_using_Weigthed_RAE_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl'
#    id_pair_map_distance_pkl_trainedWeighted_root='./distance/childparentweight/wRAETF_IDF_uiyvq_weighted_RAE_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
##    id_pair_map_distance_pkl_trainedWeighted_mean='./distance/childparentweight/kglub_weighted_RAE_IDPairinBigCloneBench_Map_Distance_mean.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root)
##    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedWeighted_mean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean)
#    metrices_weightedRAE_root_pkl='./result/childparentweight/wRAETF_IDF_uiyvq_metrices_weightedRAE_root.xiaojiepkl'
##    metrices_weightedRAE_mean_pkl='./result/childparentweight/cweav_metrices_weightedRAE_mean.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_root,metrics_pkl=metrices_weightedRAE_root_pkl)
##    analyse(distance_pkl=id_pair_map_distance_pkl_trainedWeighted_mean,metrics_pkl=metrices_weightedRAE_mean_pkl)
#

    
    
    ####计算随机模型
#    idMapVector_PKL_Path_random_root='./vector/gsayj_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
#    
##    ./vector/kwbyx_BigCloneBench_traditionalRAE_ID_Map_Vector_mean.xiaojiepkl
##    idMapVector_PKL_Path_random_root='./vector/kziud_BigCloneBench_traditionalRAE_ID_Map_Vector_root.xiaojiepkl'
##    idMapVector_PKL_Path_random_mean='./vector/cbqep_BigCloneBench_traditionalRAE_ID_Map_Vector_mean.xiaojiepkl'
#    id_pair_map_distance_pkl_random_root='./distance/gsayj_IDPairinBigCloneBench_Map_Distance_root.xiaojiepkl'
##    id_pair_map_distance_pkl_random_mean='./distance/cbqep_IDPairinBigCloneBench_Map_Distance_mean.xiaojiepkl'
#    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_random_root,id_pair_map_distance_pkl=id_pair_map_distance_pkl_random_root)
##    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_random_mean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_random_mean)
#    metrices_random_root_pkl='./result/20190622gsayj_random_root.xiaojiepkl'
##    metrices_random_mean_pkl='./result/kziud_random_mean.xiaojiepkl'
#    analyse(distance_pkl=id_pair_map_distance_pkl_random_root,metrics_pkl=metrices_random_root_pkl)
#    analyse(distance_pkl=id_pair_map_distance_pkl_random_mean,metrics_pkl=metrices_random_mean_pkl)
#    
    
    
    ############差别分析
#   analyse_chabie_report()
def experimentID_2():   
    bigCloneBenchIdPair_PKL_Path = './SplitDataSet/data/all_pairs_id_XIAOJIE.pkl' #已经去除没有对应预料库编号的bigclonebench中的函数（被他们错误标记），以及去重

    idMapVector_PKL_Path_trainedMean='./vector/lxeip_BigCloneBench_traditionalRAE_ID_Map_Vector_mean.xiaojiepkl'
    
    id_pair_map_distance_pkl_trainedMean='./distance/traditional_RAE_IdPairinBigCloneBench_Map_Distance_Mean.xiaojiepkl'

    #####先根据模型去计算距离，然后保存到pkl文件中去
    save_distance_file(bigCloneBenchIdPair_PKL_Path=bigCloneBenchIdPair_PKL_Path,idMapVector_PKL_Path=idMapVector_PKL_Path_trainedMean,id_pair_map_distance_pkl=id_pair_map_distance_pkl_trainedMean)
    
    
    metrices_traditionalRAE_mean_pkl='./result/metrices_traditionalRAE_Mean.xiaojiepkl'
    
#    analyse(distance_pkl=id_pair_map_distance_pkl_trainedMean,metrics_pkl=metrices_traditionalRAE_mean_pkl)
   
    
if __name__ == "__main__":
    experimentID_1()    
#    experimentID_2()   