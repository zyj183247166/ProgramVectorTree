# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:05:01 2019

@author: Administrator
"""
import pickle
import numpy as np
def save_to_pkl(python_content, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(python_content, pickle_f)
def read_from_pkl(pickle_name):
    with open(pickle_name, 'rb') as pickle_f:
        python_content = pickle.load(pickle_f)
    return python_content

pickle_name='./vector/using_PVT_fullCorpusLine_Map_Vector.xiaojiepkl'
#合并mean的
dict_all_mean={}
for i in range(13):
    filename='./vector/'+str(i)+'_fullCorpusLine_PVT_root.xiaojiepkl'
    dict_partial=read_from_pkl(filename)
    for key,value in dict_partial.items():
        dict_all_mean[key]=value
    del(dict_partial)

save_to_pkl(dict_all_mean, pickle_name)
dict_all_mean=read_from_pkl(pickle_name)
dict_mean_len=len(dict_all_mean)
print(dict_mean_len)
x=[]
for i in range(dict_mean_len):
    if i not in dict_all_mean:
        print(i)
    else:
        x.append(dict_all_mean[i])
numpy_mean=np.array(x)
np.save('./vector/using_PVT_fullCorpusLine_Map_Vector.npy',numpy_mean)
numpy_mean_2 = np.load("./vector/using_PVT_fullCorpusLine_Map_Vector.npy")
xiaojie=numpy_mean_2[0]
length_xiaojie = np.sqrt(np.sum(np.square(xiaojie)))
print (xiaojie.shape)
print (length_xiaojie)
numpy_mean_2 = np.load("G:\\code_of_zengjie\DeepLearningCodeOfXiaoJie\\Recursive_autoencoder_xiaojie_296_dimension\\vector/using_Weigthed_RAE_fullCorpusLine_Map_Vector_weighted.npy")
xiaojie=numpy_mean_2[0]
length_xiaojie = np.sqrt(np.sum(np.square(xiaojie)))
print (xiaojie.shape)
print (length_xiaojie)
numpy_mean_2 = np.load("G:\\code_of_zengjie\DeepLearningCodeOfXiaoJie\\Recursive_autoencoder_xiaojie_296_dimension\\vector/using_traditional_RAE_fullCorpusLine_Map_Vector_root.npy")
xiaojie=numpy_mean_2[0]
length_xiaojie = np.sqrt(np.sum(np.square(xiaojie)))
print (xiaojie.shape)
print (length_xiaojie)
pass
