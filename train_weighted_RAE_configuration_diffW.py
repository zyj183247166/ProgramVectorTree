# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:12:28 2019

@author: Administrator
"""

configuration={}
configuration['label_size']=2
configuration['early_stopping']=5
configuration['max_epochs']=20
configuration['anneal_threshold']=0.99
configuration['anneal_by']=1.5
configuration['lr']=0.01
configuration['l2']=0.02
configuration['embed_size']=296
configuration['model_name']='weighted_RAE_diffW_embed=%d_l2=%f_lr=%f.weights'%(configuration['embed_size'], configuration['lr'], configuration['l2'])
configuration['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie_296_dimension/2word2vecOutData/'
configuration['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie_296_dimension/3RvNNoutData/'
configuration['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie_296_dimension/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
configuration['batch_size']=10 
configuration['batch_size_using_model_notTrain']=400 
configuration['MAX_SENTENCE_LENGTH_for_Bigclonebench']=600
configuration['corpus_fixed_tree_construction_parentType_weight_file']='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'
configuration['corpus_word_TF_IDF_file']='./1corpusData/corpus_bcb_reduced.method.wordTFIDF.txt'