# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 09:13:58 2018

@author: a
"""
import time
import numpy as np
from xiaojie_log import xiaojie_log_class
class corpora:
    pass

global logs
logs=xiaojie_log_class()

def get_fixed_tree_constructionorder_from_file(corpus_fixed_tree_constructionorder_file,training=False):
    with open(corpus_fixed_tree_constructionorder_file, 'r') as fw:    
        sentences_fixed_tree_constructionorder=[]
        for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
            sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
            sentence_fixed_tree_constructionorder=[]
            for i in range(len(sentence_fixed_tree_constructionorder_list)):
                one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                one_time_construction_order=one_time_construction_order.strip('(')
                one_time_construction_order=one_time_construction_order.strip(')')
                one_time_construction_order_list=one_time_construction_order.split(',')
                child_id1=int(one_time_construction_order_list[0])
#                type1=one_time_construction_order_list[1]
                child_id2=int(one_time_construction_order_list[2])
#                type2=one_time_construction_order_list[3]
                parent_id=int(one_time_construction_order_list[4])
#                type3=one_time_construction_order_list[5]
                sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                pass
            b=np.array(sentence_fixed_tree_constructionorder)
            b=b.transpose()
            sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
            pass
    return sentences_fixed_tree_constructionorder #numpy矩阵的的列表。由于每个numpy矩阵大小不同，所以最高纬度只能用列表。

def preprocess_withAST_weight(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,corpus_fixed_tree_construction_parentType_weight_file,MAX_SENTENCE_LENGTH): #有抽象语法树结构的，需要提取抽象语法树。
    nowTime = lambda:int(round(time.time() * 1000))
    start_time = nowTime()

    logs.log('processing wird2vecOutdata')
    trainCorpus,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder,trainCorpus_fixed_tree_constructionorder_parentType_weight=transform_withAST_weight(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,corpus_fixed_tree_construction_parentType_weight_file,MAX_SENTENCE_LENGTH,'corpus', True)
    fullCorpus,max_sentence_length_full_Corpus,fullCorpus_fixed_tree_constructionorder,fullCorpus_fixed_tree_constructionorder_parentType_weight=transform_withAST_weight(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,corpus_fixed_tree_construction_parentType_weight_file ,MAX_SENTENCE_LENGTH,'corpus', False)
    vocabulary=readVocabulary(IDIR)
    We=loadWordEmbedding(IDIR)
    logs.log('processing wird2vecOutdata time: {} ms'.format(nowTime() - start_time))
    return (trainCorpus,fullCorpus,vocabulary,We,max_sentence_length_train_Corpus,max_sentence_length_full_Corpus,trainCorpus_fixed_tree_constructionorder,fullCorpus_fixed_tree_constructionorder,trainCorpus_fixed_tree_constructionorder_parentType_weight,fullCorpus_fixed_tree_constructionorder_parentType_weight)
def transform_withAST_weight(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,corpus_fixed_tree_construction_parentType_weight_file,MAX_SENTENCE_LENGTH,granularity, training):
    listStr=[IDIR,'/',granularity,'.int'] #所有的语料经过word2vec等处理后，都要变成corpus.int文件
    filename=''.join(listStr)    
    count = 0
    for index, line in enumerate(open(filename,'r')):
        count += 1
    numlines=count
    logs.log("corpus.int的文件行数也就是sentence个数为%d"%numlines)
    with open(corpus_fixed_tree_constructionorder_file, 'r') as fw:
        with open(corpus_fixed_tree_construction_parentType_weight_file, 'r') as fw2:    
            with open(filename, 'r') as fid:    
                corpus = [str2int(l) for l in fid.readlines()]
            if training:
                logs.log('设置用于从语料库中筛选出训练集的max sentence length=%d'%MAX_SENTENCE_LENGTH)
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                g=lambda x: (len(x) < MAX_SENTENCE_LENGTH) and (len(x)>1)
                
                training_data=[element for element in corpus if g(element)]
                logs.log('小于MAX_SENTENCE_LENGTH的training corpus has %d sentences'%(len(training_data)))
                #logs.log(''%((len)training_data))
                sizes = list(map(len,training_data))
                logs.log('training corpus最长的sentencde的长度为%d'%(max(sizes)))
                
                #如果一个训练语料是10个单词，那么根据它构建抽象语法树的次数需要9次，即对应的文件中该行保存的抽象语法树结构共九次构建。故而比较基准为减去一。
                g2=lambda x: (len(x) < MAX_SENTENCE_LENGTH-1) and (len(x)>0)
                sentences_fixed_tree_constructionorder=[]
                for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree_constructionorder=[]
                    if(g2(sentence_fixed_tree_constructionorder_list)==False):
                        continue
                    for i in range(len(sentence_fixed_tree_constructionorder_list)):
                        one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                        one_time_construction_order=one_time_construction_order.strip('(')
                        one_time_construction_order=one_time_construction_order.strip(')')
                        one_time_construction_order_list=one_time_construction_order.split(',')
                        child_id1=int(one_time_construction_order_list[0])
        #                type1=one_time_construction_order_list[1]
                        child_id2=int(one_time_construction_order_list[2])
        #                type2=one_time_construction_order_list[3]
                        parent_id=int(one_time_construction_order_list[4])
        #                type3=one_time_construction_order_list[5]
                        sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                        pass
                    b=np.array(sentence_fixed_tree_constructionorder)
                    b=b.transpose()
                    sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                sentences_fixed_tree_constructionorder_parentType_weight=[]
                for i, item in enumerate(fw2, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_parentType_weight_list=item.strip('\n').strip(' ').split(' ')
                    item=item.strip('\n').strip(' ')
                    if(g2(sentence_fixed_tree_constructionorder_parentType_weight_list)==False):
                        continue
                    sentence_fixed_tree_constructionorder_parentType_weight=str2double(item)
                    sentences_fixed_tree_constructionorder_parentType_weight.append(sentence_fixed_tree_constructionorder_parentType_weight)        
                pass
                return training_data,max(sizes),sentences_fixed_tree_constructionorder,sentences_fixed_tree_constructionorder_parentType_weight
            else:
                g=lambda x:(len(x)>1) #对于整个语料库而言，如果不是用于训练，就不需要限制上限，即句子最大长度。因为训练需要对网络的规模进行限制，而句子长度就决定了网络的规模。
                all_data=[element for element in corpus if g(element)]
                sizes = list(map(len,all_data))
                logs.log('full corpus最长的sentencde的长度为%d'%(max(sizes)))
                pass
                sentences_fixed_tree_constructionorder=[]
                for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree_constructionorder=[]
                    for i in range(len(sentence_fixed_tree_constructionorder_list)):
                        one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                        one_time_construction_order=one_time_construction_order.strip('(')
                        one_time_construction_order=one_time_construction_order.strip(')')
                        one_time_construction_order_list=one_time_construction_order.split(',')
                        child_id1=int(one_time_construction_order_list[0])
        #                type1=one_time_construction_order_list[1]
                        child_id2=int(one_time_construction_order_list[2])
        #                type2=one_time_construction_order_list[3]
                        parent_id=int(one_time_construction_order_list[4])
        #                type3=one_time_construction_order_list[5]
                        sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                        pass
                    b=np.array(sentence_fixed_tree_constructionorder)
                    b=b.transpose()
                    sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                sentences_fixed_tree_constructionorder_parentType_weight=[]
                for i, item in enumerate(fw2, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_parentType_weight_list=item.strip('\n').strip(' ').split(' ')
                    item=item.strip('\n').strip(' ')
                    sentence_fixed_tree_constructionorder_parentType_weight=str2double(item)
                    sentences_fixed_tree_constructionorder_parentType_weight.append(sentence_fixed_tree_constructionorder_parentType_weight)        
                pass
                return all_data,max(sizes),sentences_fixed_tree_constructionorder,sentences_fixed_tree_constructionorder_parentType_weight

def preprocess_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH): #有抽象语法树结构的，需要提取抽象语法树。
    nowTime = lambda:int(round(time.time() * 1000))
    start_time = nowTime()

    logs.log('processing wird2vecOutdata')
    vocabulary=readVocabulary(IDIR)
    We=loadWordEmbedding(IDIR)
    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,'corpus', True)
    fullCorpus,fullCorpus_sentence_length,max_sentence_length_full_Corpus,fullCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file ,MAX_SENTENCE_LENGTH,'corpus', False)
    
    logs.log('processing wird2vecOutdata time: {} ms'.format(nowTime() - start_time))
    return (trainCorpus,fullCorpus,trainCorpus_sentence_length,fullCorpus_sentence_length,vocabulary,We,max_sentence_length_train_Corpus,max_sentence_length_full_Corpus,trainCorpus_fixed_tree_constructionorder,fullCorpus_fixed_tree_constructionorder)

def preprocess_withAST_experimentID_1(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH): #有抽象语法树结构的，需要提取抽象语法树。
    nowTime = lambda:int(round(time.time() * 1000))
    start_time = nowTime()

    logs.log('processing wird2vecOutdata')
    vocabulary=readVocabulary(IDIR)
    We=loadWordEmbedding(IDIR)
#    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,'corpus', True)
#    fullCorpus,fullCorpus_sentence_length,max_sentence_length_full_Corpus,fullCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file ,MAX_SENTENCE_LENGTH,'corpus', False)
    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,'corpus', False)
    ####不同于别的筛选策略，其实是将全部的corpus作为trainCorpus返回。
    logs.log('processing wird2vecOutdata time: {} ms'.format(nowTime() - start_time))
    return (trainCorpus,trainCorpus_sentence_length,vocabulary,We,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder)

def transform_withAST11(IDIR,ODIR,corpus_word_TF_IDF_file,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,corpus_fixed_tree_construction_parentType_weight_file,granularity, training):
    listStr=[IDIR,'/',granularity,'.int'] #所有的语料经过word2vec等处理后，都要变成corpus.int文件
    filename=''.join(listStr)    
    count = 0
    for index, line in enumerate(open(filename,'r')):
        count += 1
    numlines=count
    logs.log("corpus.int的文件行数也就是sentence个数为%d"%numlines)
    with open(corpus_fixed_tree_constructionorder_file, 'r') as fw:  
        with open(corpus_fixed_tree_construction_parentType_weight_file,'r')as fxiaojie:
            with open(filename, 'r') as fid:    
                corpus = [str2int(l) for l in fid.readlines()]
            with open(corpus_word_TF_IDF_file, 'r') as fid:    
                corpus_word_TF_IDF = [str2double(l) for l in fid.readlines()]                
            if training:
                logs.log('设置用于从语料库中筛选出训练集的max sentence length=%d'%MAX_SENTENCE_LENGTH)
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                g=lambda x: (len(x) < MAX_SENTENCE_LENGTH) and (len(x)>1)
                
                training_data=[element for element in corpus if g(element)]
                training_data_TF_IDF=[element for element in corpus_word_TF_IDF if g(element)]
                
                
                logs.log('小于MAX_SENTENCE_LENGTH的training corpus has %d sentences'%(len(training_data)))
                #logs.log(''%((len)training_data))
                sizes = list(map(len,training_data))
                logs.log('training corpus最长的sentencde的长度为%d'%(max(sizes)))
                
                #如果一个训练语料是10个单词，那么根据它构建抽象语法树的次数需要9次，即对应的文件中该行保存的抽象语法树结构共九次构建。故而比较基准为减去一。
                g2=lambda x: (len(x) < MAX_SENTENCE_LENGTH-1) and (len(x)>0)
                sentences_fixed_tree_constructionorder=[]
                for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree_constructionorder=[]
                    if(g2(sentence_fixed_tree_constructionorder_list)==False):
                        continue
                    for i in range(len(sentence_fixed_tree_constructionorder_list)):
                        one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                        one_time_construction_order=one_time_construction_order.strip('(')
                        one_time_construction_order=one_time_construction_order.strip(')')
                        one_time_construction_order_list=one_time_construction_order.split(',')
                        child_id1=int(one_time_construction_order_list[0])
        #                type1=one_time_construction_order_list[1]
                        child_id2=int(one_time_construction_order_list[2])
        #                type2=one_time_construction_order_list[3]
                        parent_id=int(one_time_construction_order_list[4])
        #                type3=one_time_construction_order_list[5]
                        sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                        pass
                    b=np.array(sentence_fixed_tree_constructionorder)
                    b=b.transpose()
                    sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                    pass
                pass
            
                g2=lambda x: (len(x) < MAX_SENTENCE_LENGTH-1) and (len(x)>0)
                sentences_fixed_tree__parentTypes=[]
                for i, item in enumerate(fxiaojie, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree__parentTypes_weight_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree__parentTypes_weight=[]
                    if(g2(sentence_fixed_tree__parentTypes_weight_list)==False):
                        continue
                    for j in range(len(sentence_fixed_tree__parentTypes_weight_list)):
                        parentTypeWeight=(float)(sentence_fixed_tree__parentTypes_weight_list[j])
                        sentence_fixed_tree__parentTypes_weight.append(parentTypeWeight)
                        pass
                    b=np.array(sentence_fixed_tree__parentTypes_weight)
                    sentences_fixed_tree__parentTypes.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                
                return training_data,sizes,max(sizes),sentences_fixed_tree_constructionorder,sentences_fixed_tree__parentTypes,training_data_TF_IDF
            else:
                g=lambda x:(len(x)>1) #对于整个语料库而言，如果不是用于训练，就不需要限制上限，即句子最大长度。因为训练需要对网络的规模进行限制，而句子长度就决定了网络的规模。
                all_data=[element for element in corpus if g(element)]
                all_data_TF_IDF=[element for element in corpus_word_TF_IDF if g(element)]
                sizes = list(map(len,all_data))
                logs.log('full corpus最长的sentencde的长度为%d'%(max(sizes)))
                pass
                sentences_fixed_tree_constructionorder=[]
                for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree_constructionorder=[]
                    for i in range(len(sentence_fixed_tree_constructionorder_list)):
                        one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                        one_time_construction_order=one_time_construction_order.strip('(')
                        one_time_construction_order=one_time_construction_order.strip(')')
                        one_time_construction_order_list=one_time_construction_order.split(',')
                        child_id1=int(one_time_construction_order_list[0])
        #                type1=one_time_construction_order_list[1]
                        child_id2=int(one_time_construction_order_list[2])
        #                type2=one_time_construction_order_list[3]
                        parent_id=int(one_time_construction_order_list[4])
        #                type3=one_time_construction_order_list[5]
                        sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                        pass
                    b=np.array(sentence_fixed_tree_constructionorder)
                    b=b.transpose()
                    sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                
                sentences_fixed_tree__parentTypes=[]
                for i, item in enumerate(fxiaojie, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree__parentTypes_weight_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree__parentTypes_weight=[]
                    for j in range(len(sentence_fixed_tree__parentTypes_weight_list)):
                        parentTypeWeight=(float)(sentence_fixed_tree__parentTypes_weight_list[j])
                        sentence_fixed_tree__parentTypes_weight.append(parentTypeWeight)
                        pass
                    b=np.array(sentence_fixed_tree__parentTypes_weight)
                    sentences_fixed_tree__parentTypes.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                return all_data,sizes,max(sizes),sentences_fixed_tree_constructionorder,sentences_fixed_tree__parentTypes,all_data_TF_IDF
def preprocess_withAST_experimentID_11(IDIR,ODIR,corpus_word_TF_IDF_file,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,corpus_fixed_tree_construction_parentType_weight_file): #有抽象语法树结构的，需要提取抽象语法树。
    nowTime = lambda:int(round(time.time() * 1000))
    start_time = nowTime()

    logs.log('processing wird2vecOutdata')
    vocabulary=readVocabulary(IDIR)
    We=loadWordEmbedding(IDIR)
#    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,'corpus', True)
#    fullCorpus,fullCorpus_sentence_length,max_sentence_length_full_Corpus,fullCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file ,MAX_SENTENCE_LENGTH,'corpus', False)
    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder,trainCorpus_fixed_tree_parentTypes_weight,trainCorpus_TF_IDF=transform_withAST11(IDIR,ODIR,corpus_word_TF_IDF_file,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,corpus_fixed_tree_construction_parentType_weight_file,'corpus', False)
    ####不同于别的筛选策略，其实是将全部的corpus作为trainCorpus返回。
    logs.log('processing wird2vecOutdata time: {} ms'.format(nowTime() - start_time))
    return (trainCorpus,trainCorpus_sentence_length,vocabulary,We,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder,trainCorpus_fixed_tree_parentTypes_weight,trainCorpus_TF_IDF)



def preprocess_withAST_experimentID_10(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,corpus_fixed_tree_construction_parentType_weight_file): #有抽象语法树结构的，需要提取抽象语法树。
    nowTime = lambda:int(round(time.time() * 1000))
    start_time = nowTime()

    logs.log('processing wird2vecOutdata')
    vocabulary=readVocabulary(IDIR)
    We=loadWordEmbedding(IDIR)
#    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,'corpus', True)
#    fullCorpus,fullCorpus_sentence_length,max_sentence_length_full_Corpus,fullCorpus_fixed_tree_constructionorder=transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file ,MAX_SENTENCE_LENGTH,'corpus', False)
    trainCorpus,trainCorpus_sentence_length,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder,trainCorpus_fixed_tree_parentTypes_weight=transform_withAST10(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,corpus_fixed_tree_construction_parentType_weight_file,'corpus', False)
    ####不同于别的筛选策略，其实是将全部的corpus作为trainCorpus返回。
    logs.log('processing wird2vecOutdata time: {} ms'.format(nowTime() - start_time))
    return (trainCorpus,trainCorpus_sentence_length,vocabulary,We,max_sentence_length_train_Corpus,trainCorpus_fixed_tree_constructionorder,trainCorpus_fixed_tree_parentTypes_weight)


def transform_withAST10(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,corpus_fixed_tree_construction_parentType_weight_file,granularity, training):
    listStr=[IDIR,'/',granularity,'.int'] #所有的语料经过word2vec等处理后，都要变成corpus.int文件
    filename=''.join(listStr)    
    count = 0
    for index, line in enumerate(open(filename,'r')):
        count += 1
    numlines=count
    logs.log("corpus.int的文件行数也就是sentence个数为%d"%numlines)
    with open(corpus_fixed_tree_constructionorder_file, 'r') as fw:  
        with open(corpus_fixed_tree_construction_parentType_weight_file,'r')as fxiaojie:
            with open(filename, 'r') as fid:    
                corpus = [str2int(l) for l in fid.readlines()]
            if training:
                logs.log('设置用于从语料库中筛选出训练集的max sentence length=%d'%MAX_SENTENCE_LENGTH)
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                g=lambda x: (len(x) < MAX_SENTENCE_LENGTH) and (len(x)>1)
                
                training_data=[element for element in corpus if g(element)]
                logs.log('小于MAX_SENTENCE_LENGTH的training corpus has %d sentences'%(len(training_data)))
                #logs.log(''%((len)training_data))
                sizes = list(map(len,training_data))
                logs.log('training corpus最长的sentencde的长度为%d'%(max(sizes)))
                
                #如果一个训练语料是10个单词，那么根据它构建抽象语法树的次数需要9次，即对应的文件中该行保存的抽象语法树结构共九次构建。故而比较基准为减去一。
                g2=lambda x: (len(x) < MAX_SENTENCE_LENGTH-1) and (len(x)>0)
                sentences_fixed_tree_constructionorder=[]
                for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree_constructionorder=[]
                    if(g2(sentence_fixed_tree_constructionorder_list)==False):
                        continue
                    for i in range(len(sentence_fixed_tree_constructionorder_list)):
                        one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                        one_time_construction_order=one_time_construction_order.strip('(')
                        one_time_construction_order=one_time_construction_order.strip(')')
                        one_time_construction_order_list=one_time_construction_order.split(',')
                        child_id1=int(one_time_construction_order_list[0])
        #                type1=one_time_construction_order_list[1]
                        child_id2=int(one_time_construction_order_list[2])
        #                type2=one_time_construction_order_list[3]
                        parent_id=int(one_time_construction_order_list[4])
        #                type3=one_time_construction_order_list[5]
                        sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                        pass
                    b=np.array(sentence_fixed_tree_constructionorder)
                    b=b.transpose()
                    sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                    pass
                pass
            
                g2=lambda x: (len(x) < MAX_SENTENCE_LENGTH-1) and (len(x)>0)
                sentences_fixed_tree__parentTypes=[]
                for i, item in enumerate(fxiaojie, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree__parentTypes_weight_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree__parentTypes_weight=[]
                    if(g2(sentence_fixed_tree__parentTypes_weight_list)==False):
                        continue
                    for j in range(len(sentence_fixed_tree__parentTypes_weight_list)):
                        parentTypeWeight=(float)(sentence_fixed_tree__parentTypes_weight_list[j])
                        sentence_fixed_tree__parentTypes_weight.append(parentTypeWeight)
                        pass
                    b=np.array(sentence_fixed_tree__parentTypes_weight)
                    sentences_fixed_tree__parentTypes.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                
                return training_data,sizes,max(sizes),sentences_fixed_tree_constructionorder,sentences_fixed_tree__parentTypes
            else:
                g=lambda x:(len(x)>1) #对于整个语料库而言，如果不是用于训练，就不需要限制上限，即句子最大长度。因为训练需要对网络的规模进行限制，而句子长度就决定了网络的规模。
                all_data=[element for element in corpus if g(element)]
                sizes = list(map(len,all_data))
                logs.log('full corpus最长的sentencde的长度为%d'%(max(sizes)))
                pass
                sentences_fixed_tree_constructionorder=[]
                for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree_constructionorder=[]
                    for i in range(len(sentence_fixed_tree_constructionorder_list)):
                        one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                        one_time_construction_order=one_time_construction_order.strip('(')
                        one_time_construction_order=one_time_construction_order.strip(')')
                        one_time_construction_order_list=one_time_construction_order.split(',')
                        child_id1=int(one_time_construction_order_list[0])
        #                type1=one_time_construction_order_list[1]
                        child_id2=int(one_time_construction_order_list[2])
        #                type2=one_time_construction_order_list[3]
                        parent_id=int(one_time_construction_order_list[4])
        #                type3=one_time_construction_order_list[5]
                        sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                        pass
                    b=np.array(sentence_fixed_tree_constructionorder)
                    b=b.transpose()
                    sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                
                sentences_fixed_tree__parentTypes=[]
                for i, item in enumerate(fxiaojie, start=1):  # MATLAB is 1-indexed
                    sentence_fixed_tree__parentTypes_weight_list=item.strip('\n').strip(' ').split(' ')
                    sentence_fixed_tree__parentTypes_weight=[]
                    for j in range(len(sentence_fixed_tree__parentTypes_weight_list)):
                        parentTypeWeight=(float)(sentence_fixed_tree__parentTypes_weight_list[j])
                        sentence_fixed_tree__parentTypes_weight.append(parentTypeWeight)
                        pass
                    b=np.array(sentence_fixed_tree__parentTypes_weight)
                    sentences_fixed_tree__parentTypes.append(b) #列表转变成了numpy矩阵
                    pass
                pass
                return all_data,sizes,max(sizes),sentences_fixed_tree_constructionorder,sentences_fixed_tree__parentTypes


def transform_withAST(IDIR,ODIR,corpus_fixed_tree_constructionorder_file,MAX_SENTENCE_LENGTH,granularity, training):
    listStr=[IDIR,'/',granularity,'.int'] #所有的语料经过word2vec等处理后，都要变成corpus.int文件
    filename=''.join(listStr)    
    count = 0
    for index, line in enumerate(open(filename,'r')):
        count += 1
    numlines=count
    logs.log("corpus.int的文件行数也就是sentence个数为%d"%numlines)
    with open(corpus_fixed_tree_constructionorder_file, 'r') as fw:  
        with open(filename, 'r') as fid:    
            corpus = [str2int(l) for l in fid.readlines()]
        if training:
            logs.log('设置用于从语料库中筛选出训练集的max sentence length=%d'%MAX_SENTENCE_LENGTH)
            #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
            g=lambda x: (len(x) < MAX_SENTENCE_LENGTH) and (len(x)>1)
            
            training_data=[element for element in corpus if g(element)]
            logs.log('小于MAX_SENTENCE_LENGTH的training corpus has %d sentences'%(len(training_data)))
            #logs.log(''%((len)training_data))
            sizes = list(map(len,training_data))
            logs.log('training corpus最长的sentencde的长度为%d'%(max(sizes)))
            
            #如果一个训练语料是10个单词，那么根据它构建抽象语法树的次数需要9次，即对应的文件中该行保存的抽象语法树结构共九次构建。故而比较基准为减去一。
            g2=lambda x: (len(x) < MAX_SENTENCE_LENGTH-1) and (len(x)>0)
            sentences_fixed_tree_constructionorder=[]
            for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                sentence_fixed_tree_constructionorder=[]
                if(g2(sentence_fixed_tree_constructionorder_list)==False):
                    continue
                for i in range(len(sentence_fixed_tree_constructionorder_list)):
                    one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                    one_time_construction_order=one_time_construction_order.strip('(')
                    one_time_construction_order=one_time_construction_order.strip(')')
                    one_time_construction_order_list=one_time_construction_order.split(',')
                    child_id1=int(one_time_construction_order_list[0])
    #                type1=one_time_construction_order_list[1]
                    child_id2=int(one_time_construction_order_list[2])
    #                type2=one_time_construction_order_list[3]
                    parent_id=int(one_time_construction_order_list[4])
    #                type3=one_time_construction_order_list[5]
                    sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                    pass
                b=np.array(sentence_fixed_tree_constructionorder)
                b=b.transpose()
                sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                pass
            pass
            return training_data,sizes,max(sizes),sentences_fixed_tree_constructionorder
        else:
            g=lambda x:(len(x)>1) #对于整个语料库而言，如果不是用于训练，就不需要限制上限，即句子最大长度。因为训练需要对网络的规模进行限制，而句子长度就决定了网络的规模。
            all_data=[element for element in corpus if g(element)]
            sizes = list(map(len,all_data))
            logs.log('full corpus最长的sentencde的长度为%d'%(max(sizes)))
            pass
            sentences_fixed_tree_constructionorder=[]
            for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                sentence_fixed_tree_constructionorder_list=item.strip('\n').strip(' ').split(' ')
                sentence_fixed_tree_constructionorder=[]
                for i in range(len(sentence_fixed_tree_constructionorder_list)):
                    one_time_construction_order=sentence_fixed_tree_constructionorder_list[i];
                    one_time_construction_order=one_time_construction_order.strip('(')
                    one_time_construction_order=one_time_construction_order.strip(')')
                    one_time_construction_order_list=one_time_construction_order.split(',')
                    child_id1=int(one_time_construction_order_list[0])
    #                type1=one_time_construction_order_list[1]
                    child_id2=int(one_time_construction_order_list[2])
    #                type2=one_time_construction_order_list[3]
                    parent_id=int(one_time_construction_order_list[4])
    #                type3=one_time_construction_order_list[5]
                    sentence_fixed_tree_constructionorder.append([child_id1,child_id2,parent_id])
                    pass
                b=np.array(sentence_fixed_tree_constructionorder)
                b=b.transpose()
                sentences_fixed_tree_constructionorder.append(b) #列表转变成了numpy矩阵
                pass
            pass
            return all_data,sizes,max(sizes),sentences_fixed_tree_constructionorder

def preprocess(IDIR,ODIR,MAX_SENTENCE_LENGTH): #对于没有抽象语法树结构的，只需要提取语料即可，就不需要抽象语法树。
    nowTime = lambda:int(round(time.time() * 1000))
    start_time = nowTime()

    logs.log('processing wird2vecOutdata')
    trainCorpus,max_sentence_length_train_Corpus=transform(IDIR,ODIR,MAX_SENTENCE_LENGTH,'corpus', True)
    fullCorpus,max_sentence_length_full_Corpus=transform(IDIR,ODIR,MAX_SENTENCE_LENGTH,'corpus', False)
    vocabulary=readVocabulary(IDIR)
    We=loadWordEmbedding(IDIR)
    logs.log('processing wird2vecOutdata time: {} ms'.format(nowTime() - start_time))
    return (trainCorpus,fullCorpus,vocabulary,We,max_sentence_length_train_Corpus,max_sentence_length_full_Corpus)
def loadWordEmbedding(IDIR):
    listStr=[IDIR,'/embed.txt']
    filename=''.join(listStr)
    with open(filename, 'r') as fid:
        We = [str2double(l) for l in fid.readlines()]
    We=np.transpose(We).tolist()#对列表表示的矩阵进行转置，之后仍然得到列表。转置以后用每一列存放一个单词的词向量。即行数是固定的，300行。
    return We
def readVocabulary(IDIR):
    listStr=[IDIR,'/vocab.txt']
    filename=''.join(listStr)
    with open(filename, 'r') as fid:    
        vocabulary = [l.strip('\n') for l in fid.readlines()]
    return vocabulary
    
def transform(IDIR,ODIR,MAX_SENTENCE_LENGTH,granularity, training):
    listStr=[IDIR,'/',granularity,'.int']
    filename=''.join(listStr)    
    count = 0
    for index, line in enumerate(open(filename,'r')):
        count += 1
    numlines=count
    logs.log("corpus.int的文件行数也就是sentence个数为%d"%numlines)
    with open(filename, 'r') as fid:    
        corpus = [str2int(l) for l in fid.readlines()]
    if training:
        logs.log('设置用于从语料库中筛选出训练集的max sentence length=%d'%MAX_SENTENCE_LENGTH)
       
        #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
        g=lambda x: (len(x) < MAX_SENTENCE_LENGTH) and (len(x)>1)
        training_data=[element for element in corpus if g(element)]
        logs.log('小于MAX_SENTENCE_LENGTH的training corpus has %d sentences'%(len(training_data)))
        #logs.log(''%((len)training_data))
        sizes = list(map(len,training_data))
        logs.log('training corpus最长的sentencde的长度为%d'%(max(sizes)))
        pass
        return training_data,max(sizes)
    else:
        g=lambda x:(len(x)>1) #对于整个语料库而言，如果不是用于训练，就不需要限制上限，即句子最大长度。因为训练需要对网络的规模进行限制，而句子长度就决定了网络的规模。
        all_data=[element for element in corpus if g(element)]
        sizes = list(map(len,all_data))
        logs.log('full corpus最长的sentencde的长度为%d'%(max(sizes)))
        pass
        return all_data,max(sizes)
def str2int(string):
    numbers = list(map(int, string.strip().split()))
    return numbers

def str2double(string):
    numbers = list(map(float, string.strip().split()))
    return numbers

