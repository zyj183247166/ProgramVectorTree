# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:21:59 2019

@author: Administrator
"""

import os
import struct
from os.path import getsize
import numpy as np
import time
import pickle
def read_from_pkl(pickle_name):
    with open(pickle_name, 'rb') as pickle_f:
        python_content = pickle.load(pickle_f)
    return python_content
def save_to_pkl(python_content, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(python_content, pickle_f)
def read_fvecs_to_Numpy(FilePath,numpyPath):
    
    FeactureSize = 4
    FileSize = getsize(FilePath)
    print(FileSize)
    fobject = open(FilePath,'rb')
    index = 0;
    Hex = fobject.read(4)
    Hex = struct.unpack_from('<I' , Hex , index) # Key point '<'(Little) or '>'(Big)
    index += struct.calcsize('<I')
    print (Hex[0])
    print (index)
    FeactureDim = Hex[0]
    print (FeactureDim)
    RowSize = 1*4 + FeactureDim*FeactureSize
    print (RowSize)
    SampleNum = (int)(FileSize/RowSize)
    print (SampleNum)
    fobject.close()
    fobject = open(FilePath,'rb')
    dimensionNum=Hex[0]
    returnedNumpy=[]
    for i in range(SampleNum):
#    for i in range(100):
        fobject.read(4)#读取行的元素个数，然后不用
        x=[]
        for j in range(dimensionNum):
            value=fobject.read(4)
            value=struct.unpack_from('<f' , value , 0) #读出浮点数
            value=value[0]
            x.append(value)
            pass
        if(i%100==0):
            print ('i:%d'%i)
        returnedNumpy.append(x)
    fobject.close()
    returnedNumpy=np.array(returnedNumpy)
    ####保存到文件
#    np.save(numpyPath,RESULT_fullCorpusLine_Pair_returnedByNSG)
    pass
#fullCorpusLine_Map_Vector_numpy_trainedroot_numpy=np.load('./vector/fullCorpusLine_Map_Vector_numpy_trainedroot.npy')
pass
########################################################################################################################################################################
##############m将NSG返回的结果从ivecs转换成numpy矩阵并报错
def convert_NSGresult_to_Numpy(FilePath,numpyPath): #这个只能处理每个查询都返回相同数目查询值的。
    
    FeactureSize = 4
    FileSize = getsize(FilePath)
    print(FileSize)
    fobject = open(FilePath,'rb')
    index = 0;
    Hex = fobject.read(4)
    Hex = struct.unpack_from('<I' , Hex , index) # Key point '<'(Little) or '>'(Big)
    index += struct.calcsize('<I')
    print (Hex[0])
    print (index)
    FeactureDim = Hex[0]
    print (FeactureDim)
    RowSize = 1*4 + FeactureDim*FeactureSize
    print (RowSize)
    SampleNum = (int)(FileSize/RowSize)
    print (SampleNum)
    fobject.close()
    fobject = open(FilePath,'rb')
    dimensionNum=Hex[0]
    RESULT_fullCorpusLine_Pair_returnedByNSG=[]
    for i in range(SampleNum):
#    for i in range(100):
        fobject.read(4)#读取行的元素个数，然后不用
        x=[]
        for j in range(dimensionNum):
            value=fobject.read(4)
            value=struct.unpack_from('<I' , value , 0)
            value=value[0]
            x.append(value)
            pass
        if(i%100==0):
            print ('i:%d'%i)
        RESULT_fullCorpusLine_Pair_returnedByNSG.append(x)
    fobject.close()
    RESULT_fullCorpusLine_Pair_returnedByNSG=np.array(RESULT_fullCorpusLine_Pair_returnedByNSG)
    ####保存到文件
    np.save(numpyPath,RESULT_fullCorpusLine_Pair_returnedByNSG)

def convert_NSGresult_to_dict_notFixedQueryNum(FilePath,dictPKLPath): #当每个查询按阈值返回时，就不是固定数目的，我们只能放入字典中
    
#    FeactureSize = 4
    FileSize = getsize(FilePath)
    readSize = 0 
    fobject = open(FilePath,'rb')
    RESULT_fullCorpusLine_Pair_returnedByNSG={}
    rowNum=0
    while(readSize!=FileSize):
        Hex = fobject.read(4)
        readSize+=4
        Hex = struct.unpack_from('<I' , Hex , 0) # Key point '<'(Little) or '>'(Big)
        returnedQueryResultNum=Hex[0]
        for i in range(returnedQueryResultNum):
            value = fobject.read(4)#读取行的元素个数，然后不用
            readSize+=4    
            value=struct.unpack_from('<I' , value , 0)
            value=value[0]
            if(value==rowNum): #自己跟自己相似，我们不认为是克隆
                continue
            key=(rowNum,value)
            key_reversed=(value,rowNum)
            if(key in RESULT_fullCorpusLine_Pair_returnedByNSG):
                RESULT_fullCorpusLine_Pair_returnedByNSG[key]=RESULT_fullCorpusLine_Pair_returnedByNSG[key]+1 #记录碰撞次数
                continue
            if(key_reversed in RESULT_fullCorpusLine_Pair_returnedByNSG):
                RESULT_fullCorpusLine_Pair_returnedByNSG[key_reversed]=RESULT_fullCorpusLine_Pair_returnedByNSG[key_reversed]+1 #记录碰撞次数
                continue
            #上面的两个if流程保证了当(a,b)作为键值在字典中以后，(b,a)是不可能作为键字加入到字典中去的。
            RESULT_fullCorpusLine_Pair_returnedByNSG[key]=1
        pass
        pass
        if(rowNum%10000==0):
            print ('rowNum:%d'%rowNum)
        rowNum+=1
        pass
    fobject.close()
    save_to_pkl(RESULT_fullCorpusLine_Pair_returnedByNSG,dictPKLPath)
    RESULT_fullCorpusLine_Pair_returnedByNSG.clear()
    del(RESULT_fullCorpusLine_Pair_returnedByNSG)
    return rowNum

def convert_NSGresult_to_dict_notFixedQueryNum_with_different_therolds(FilePath,jiange): #当每个查询按阈值返回时，就不是固定数目的，我们只能放入字典中
    
#    FeactureSize = 4
    pickle_name='./vector/using_PVT_fullCorpusLine_Map_Vector.xiaojiepkl'
    fullCorpusLine_Map_Vector=read_from_pkl(pickle_name)
    ########
    filename_0='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_0.txt'
    filename_1='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_1.txt'
    filename_2='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_2.txt'
    filename_3='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_3.txt'
    filename_4='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_4.txt'
    filename_5='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_5.txt'
    filename_6='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_6.txt'
    filename_7='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_7.txt'
    filename_8='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_8.txt'
    filename_9='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_9.txt'
    filename_10='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_10.txt'
    filename_11='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_11.txt'
    filename_12='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_12.txt'
    filename_13='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_13.txt'
    filename_14='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineinFullCorpus_14.txt'
    with open(filename_0, 'w') as f0:
        with open(filename_1, 'w') as f1:
            with open(filename_2, 'w') as f2:
                with open(filename_3, 'w') as f3:
                    with open(filename_4, 'w') as f4:
                        with open(filename_5, 'w') as f5:
                            with open(filename_6, 'w') as f6:
                                with open(filename_7, 'w') as f7:
                                    with open(filename_8, 'w') as f8:
                                        with open(filename_9, 'w') as f9:
                                            with open(filename_10, 'w') as f10:
                                                with open(filename_11, 'w') as f11:
                                                    with open(filename_12, 'w') as f12:
                                                        with open(filename_13, 'w') as f13:
                                                            with open(filename_14, 'w') as f14:
                                                                
                                                                FileSize = getsize(FilePath)
                                                                readSize = 0 
                                                                fobject = open(FilePath,'rb')
                        #                                        RESULT_fullCorpusLine_Pair_returnedByNSG={}
                                                                rowNum=0
                                                                while(readSize!=FileSize):
                                                                    Hex = fobject.read(4)
                                                                    readSize+=4
                                                                    Hex = struct.unpack_from('<I' , Hex , 0) # Key point '<'(Little) or '>'(Big)
                                                                    returnedQueryResultNum=Hex[0]
                                                                    for i in range(returnedQueryResultNum):
                                                                        value = fobject.read(4)#读取行的元素个数，然后不用
                                                                        readSize+=4    
                                                                        value=struct.unpack_from('<I' , value , 0)
                                                                        value=value[0]
                                                                        if(value==rowNum): #自己跟自己相似，我们不认为是克隆
                                                                            continue
                                                                        key=(rowNum,value)
                                                                        ########现在对其进行阈值判定，将符合指定阈值的放入到字典中，然后分别保存。
                                                                        key1=key[0]
                                                                        key2=key[1]
                                                                        vector1=fullCorpusLine_Map_Vector[key1]
                                                                        vector2=fullCorpusLine_Map_Vector[key2]
                                                                        distance=np.linalg.norm(vector1-vector2)
                                                                        if(distance<=(0*jiange)):
                                                                            f0.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(1*jiange)):
                                                                            f1.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(2*jiange)):
                                                                            f2.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(3*jiange)):
                                                                            f3.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(4*jiange)):
                                                                            f4.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(5*jiange)):
                                                                            f5.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(6*jiange)):
                                                                            f6.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(7*jiange)):
                                                                            f7.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(8*jiange)):
                                                                            f8.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(9*jiange)):
                                                                            f9.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(10*jiange)):
                                                                            f10.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(11*jiange)):
                                                                            f11.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(12*jiange)):
                                                                            f12.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(13*jiange)):
                                                                            f13.write(str(key1)+','+str(key2)+'\n')
                                                                        if(distance<=(14*jiange)):
                                                                            f14.write(str(key1)+','+str(key2)+'\n')
         
                                                                        
                        #                                                key_reversed=(value,rowNum)
                        #                                                if(key in RESULT_fullCorpusLine_Pair_returnedByNSG):
                        #                                                    RESULT_fullCorpusLine_Pair_returnedByNSG[key]=RESULT_fullCorpusLine_Pair_returnedByNSG[key]+1 #记录碰撞次数
                        #                                                    continue
                        #                                                if(key_reversed in RESULT_fullCorpusLine_Pair_returnedByNSG):
                        #                                                    RESULT_fullCorpusLine_Pair_returnedByNSG[key_reversed]=RESULT_fullCorpusLine_Pair_returnedByNSG[key_reversed]+1 #记录碰撞次数
                        #                                                    continue
                        #                                                #上面的两个if流程保证了当(a,b)作为键值在字典中以后，(b,a)是不可能作为键字加入到字典中去的。
                        #                                                RESULT_fullCorpusLine_Pair_returnedByNSG[key]=1
                                                                    pass
                                                                    pass
                                                                    if(rowNum%10000==0):
                                                                        print ('rowNum:%d'%rowNum)
                                                                    rowNum+=1
                                                                    f0.flush()
                                                                    f1.flush()
                                                                    f2.flush()
                                                                    f3.flush()
                                                                    f4.flush()
                                                                    f5.flush()
                                                                    f6.flush()
                                                                    f7.flush()
                                                                    f8.flush()
                                                                    f9.flush()
                                                                    f10.flush()
                                                                    f11.flush()
                                                                    f12.flush()
                                                                    f13.flush()
                                                                    f14.flush()
                                                                    pass
                                                                fobject.close()
                        #                                        save_to_pkl(RESULT_fullCorpusLine_Pair_returnedByNSG,dictPKLPath)
                        #                                        RESULT_fullCorpusLine_Pair_returnedByNSG.clear()
                        #                                        del(RESULT_fullCorpusLine_Pair_returnedByNSG)
                                                                return rowNum
    pass

nowTime = lambda:int(round(time.time() * 1000))
start_time = nowTime()
FilePath_weighted_RAE_NSG_RESULT='./nsg_result/RESULT_using_PVT_fullCorpusLine_Map_Vector.ivecs'
rowNum=convert_NSGresult_to_dict_notFixedQueryNum_with_different_therolds(FilePath_weighted_RAE_NSG_RESULT,jiange=0.05)
if (rowNum!=785438):
    print('error')
print('processing time: {} ms'.format(nowTime() - start_time))


#def convert_NSGresult_to_dict_notFixedQueryNum_with_different_therolds_2_for_tongjishijian(FilePath,jiange): #当每个查询按阈值返回时，就不是固定数目的，我们只能放入字典中
#    
##    FeactureSize = 4
#    pickle_name='./vector/using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.xiaojiepkl'
#    fullCorpusLine_Map_Vector=read_from_pkl(pickle_name)
#    ########
#
#    FileSize = getsize(FilePath)
#    readSize = 0 
#    fobject = open(FilePath,'rb')
##                                        RESULT_fullCorpusLine_Pair_returnedByNSG={}
#    rowNum=0
#    while(readSize!=FileSize):
#        Hex = fobject.read(4)
#        readSize+=4
#        Hex = struct.unpack_from('<I' , Hex , 0) # Key point '<'(Little) or '>'(Big)
#        returnedQueryResultNum=Hex[0]
#        for i in range(returnedQueryResultNum):
#            value = fobject.read(4)#读取行的元素个数，然后不用
#            readSize+=4    
#            value=struct.unpack_from('<I' , value , 0)
#            value=value[0]
#            if(value==rowNum): #自己跟自己相似，我们不认为是克隆
#                continue
#            key=(rowNum,value)
#            ########现在对其进行阈值判定，将符合指定阈值的放入到字典中，然后分别保存。
#            key1=key[0]
#            key2=key[1]
#            vector1=fullCorpusLine_Map_Vector[key1]
#            vector2=fullCorpusLine_Map_Vector[key2]
#            distance=np.linalg.norm(vector1-vector2)
#            if(distance<=(0)):
#                pass
#
#            
##                                                key_reversed=(value,rowNum)
##                                                if(key in RESULT_fullCorpusLine_Pair_returnedByNSG):
##                                                    RESULT_fullCorpusLine_Pair_returnedByNSG[key]=RESULT_fullCorpusLine_Pair_returnedByNSG[key]+1 #记录碰撞次数
##                                                    continue
##                                                if(key_reversed in RESULT_fullCorpusLine_Pair_returnedByNSG):
##                                                    RESULT_fullCorpusLine_Pair_returnedByNSG[key_reversed]=RESULT_fullCorpusLine_Pair_returnedByNSG[key_reversed]+1 #记录碰撞次数
##                                                    continue
##                                                #上面的两个if流程保证了当(a,b)作为键值在字典中以后，(b,a)是不可能作为键字加入到字典中去的。
##                                                RESULT_fullCorpusLine_Pair_returnedByNSG[key]=1
#        pass
#        pass
#        if(rowNum%10000==0):
#            print ('rowNum:%d'%rowNum)
#        rowNum+=1
#
#        pass
#    fobject.close()
##                                        save_to_pkl(RESULT_fullCorpusLine_Pair_returnedByNSG,dictPKLPath)
##                                        RESULT_fullCorpusLine_Pair_returnedByNSG.clear()
##                                        del(RESULT_fullCorpusLine_Pair_returnedByNSG)
#    return rowNum
#pass
#nowTime = lambda:int(round(time.time() * 1000))
#start_time = nowTime()
#    
#
#FilePath_weighted_RAE_NSG_RESULT='./nsg_result/RESULT_using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.ivecs'
#rowNum=convert_NSGresult_to_dict_notFixedQueryNum_with_different_therolds_2_for_tongjishijian(FilePath_weighted_RAE_NSG_RESULT,jiange=0.05)
#if (rowNum!=785438):
#    print('error')
#
#
########统计时间用
#
#print('processing time: {} ms'.format(nowTime() - start_time))


###处理定长查询    
#FilePath_trainedroot='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedroot.ivecs'
#numpyPath_trainedroot='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedroot.npy'
#FilePath_trainedmean='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedmean.ivecs'
#numpyPath_trainedmean='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedmean.npy'
#convert_NSGresult_to_Numpy(FilePath_trainedroot,numpyPath_trainedroot)
#convert_NSGresult_to_Numpy(FilePath_trainedmean,numpyPath_trainedmean)
###处理指定阈值下的不定长查询
#yuzhi=0.0
#FilePath_trainedroot='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedroot_'+str(yuzhi)+'.ivecs'
#FilePath_trainedmean='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedmean_'+str(yuzhi)+'.ivecs'
#dictPath_trainedmean='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedRoot'+str(yuzhi)+'.xiaojiepkl'
#dictPath_trainedroot='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedMean'+str(yuzhi)+'.xiaojiepkl'
#rowNum1=convert_NSGresult_to_dict_notFixedQueryNum(FilePath_trainedroot,dictPath_trainedroot)
#rowNum2=convert_NSGresult_to_dict_notFixedQueryNum(FilePath_trainedmean,dictPath_trainedmean)
#yuzhi=0.00
#jiange=0.05
#rootErrorFlag=False
#for i in range(4):
#    
#    yuzhi_name=round(yuzhi,3) #截取两位小数
#    print(yuzhi_name)
#    FilePath_trainedroot='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedroot_'+str(yuzhi_name)+'.ivecs'
#    dictPath_trainedroot='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedRoot'+str(yuzhi_name)+'.xiaojiepkl'    
#    rowNum1=convert_NSGresult_to_dict_notFixedQueryNum(FilePath_trainedroot,dictPath_trainedroot)            
#    if (rowNum1!=785438):
#        print('i:%d,error'%i)        
#        rootErrorFlag=True
#    yuzhi=yuzhi+jiange
#    pass
#yuzhi=0.00
#jiange=0.05
#meanErrorFlag=False
#for i in range(4):
#    yuzhi_name=round(yuzhi,3) #截取两位小数
#    print(yuzhi_name)
#    FilePath_trainedmean='./nsg_result/RESULT_fullCorpusLine_Map_Vector_trainedmean_'+str(yuzhi_name)+'.ivecs'
#    dictPath_trainedmean='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedmean'+str(yuzhi_name)+'.xiaojiepkl'
#    rowNum2=convert_NSGresult_to_dict_notFixedQueryNum(FilePath_trainedmean,dictPath_trainedmean)            
#    if (rowNum2!=785438):
#        print('i:%d,error'%i)
#        meanErrorFlag=True               
#    yuzhi=yuzhi+jiange
#    pass
#print(rootErrorFlag)
#print(meanErrorFlag)
##############m将NSG返回的根据阈值的结果结果，转换成克隆检测的结果。
########################################################################################################################################################################
#前面返回的结果就是要报告的而结果
#detected_clone_pairs_lineInCorpus_trainedRoot_Path=dictPath_trainedroot
#detected_clone_pairs_lineInCorpus_trainedMean_Path=dictPath_trainedmean



########################################################################################################################################################################
##############m得到克隆结果

def get_detected_clone_pairs(RESULT_fullCorpusLine_ClonePair_Lines,detected_clone_pairs_lineInCorpus_PKLPATH):
    #输出第0行的克隆行的编号，肯定会返回自身与自身的克隆关系。
#    print (len(RESULT_fullCorpusLine_ClonePair_Lines))
#    print (len(RESULT_fullCorpusLine_ClonePair_Lines[0]))
#    print (RESULT_fullCorpusLine_ClonePair_Lines[0])
#    print (RESULT_fullCorpusLine_ClonePair_Lines[-1])
#    print (RESULT_fullCorpusLine_ClonePair_Lines[99])
    dimension=len(RESULT_fullCorpusLine_ClonePair_Lines[0])
    rowNum=len(RESULT_fullCorpusLine_ClonePair_Lines)
#    detected_clone_pairs_lineInCorpus=[]
    detected_clone_pairs_lineInCorpus={}
    #字典的索引效率要远远大于列表，矩阵的索引速度也大于列表
    #################################################
    ######对矩阵的访问，经过测试，还是先选定行，再选定列的方式比较快。
    t0=time.time()
    index=0
    for i in range(rowNum):
#    for i in range(10000):
        key1=i
        for j in range(dimension):            
            key2=RESULT_fullCorpusLine_ClonePair_Lines[i][j]
            index+=1
            if (index%100000==0):
                print('index:%d'%index)
            if(key1==key2):
                continue
            ################Python字典查找指定key值是否存在对于大数据处理非常费时，能不用的时候建议不要使用。
            key=(key1,key2)
            key_reversed=(key2,key1)
            if(key in detected_clone_pairs_lineInCorpus):
                detected_clone_pairs_lineInCorpus[key]=detected_clone_pairs_lineInCorpus[key]+1 #记录碰撞次数
                continue
            if(key_reversed in detected_clone_pairs_lineInCorpus):
                detected_clone_pairs_lineInCorpus[key_reversed]=detected_clone_pairs_lineInCorpus[key_reversed]+1 #记录碰撞次数
                continue
            #上面的两个if流程保证了当(a,b)作为键值在字典中以后，(b,a)是不可能作为键字加入到字典中去的。
            detected_clone_pairs_lineInCorpus[key]=1
        pass
    t1=time.time()
    save_to_pkl(detected_clone_pairs_lineInCorpus,detected_clone_pairs_lineInCorpus_PKLPATH)
    dictlen=len(detected_clone_pairs_lineInCorpus)
    del(detected_clone_pairs_lineInCorpus)
    return ((t1-t0),dictlen)

search_K_inNSG=100;#返回的近邻个数
#RESULT_fullCorpusLine_Pair_returnedByNSG_trainedroot=np.load(numpyPath_trainedroot)
#RESULT_fullCorpusLine_Pair_returnedByNSG_trainedmean=np.load(numpyPath_trainedmean)
#detected_clone_pairs_lineInCorpus_trainedRoot_Path='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedRoot'+'_'+str(search_K_inNSG)+'.xiaojiepkl'
#detected_clone_pairs_lineInCorpus_trainedMean_Path='./detected_clone_pairs_line_in_fullCoprus/detected_clone_pairs_lineInCorpus_trainedMean'+'_'+str(search_K_inNSG)+'.xiaojiepkl'
#time1,dictlen1=get_detected_clone_pairs(RESULT_fullCorpusLine_Pair_returnedByNSG_trainedroot,detected_clone_pairs_lineInCorpus_trainedRoot_Path)
#time2,dictlen2=get_detected_clone_pairs(RESULT_fullCorpusLine_Pair_returnedByNSG_trainedmean,detected_clone_pairs_lineInCorpus_trainedMean_Path)

#print(time1,dictlen1)
#print(time2,dictlen2)
#print(time2)
#xx=read_from_pkl(detected_clone_pairs_lineInCorpus_trainedRoot_Path)
##########################################
pass
    