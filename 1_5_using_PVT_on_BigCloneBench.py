#!/usr/bin/python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import shutil
import tensorflow as tf
from xiaojie_log import xiaojie_log_class
from processWord2vecOutData import preprocess_withAST,preprocess_withAST_experimentID_1,preprocess_withAST_experimentID_10
from scipy.spatial.distance import pdist, squareform
from random import shuffle
import random
import pickle
global logs
logs=xiaojie_log_class()
class Config(object):

    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    def __init__(self,configuration_dict):
        
        logs.log("(1)>>>>  before training RvNN,配置超参数")
        
        self.label_size = configuration_dict['label_size']
        self.early_stopping = configuration_dict['early_stopping'] #如果在评估集上训练了10次，都没有小于之前在评估集上计算的损失值，那么就终止训练
#        self.max_epochs = 20
        self.max_epochs = configuration_dict['max_epochs']
#        self.max_epochs = 1
        self.anneal_threshold = configuration_dict['anneal_threshold']#损失值一直不怎么降低的时候，我们就降低学习率
        self.anneal_by =configuration_dict['anneal_by']
        #max_epochs = 30
        
        self.lr = configuration_dict['lr']
        self.l2 = configuration_dict['l2']
        self.embed_size = configuration_dict['embed_size'] #这个要根据word2vec的配置来。
        self.model_name = configuration_dict['model_name']
        logs.log('模型名称为%s'%self.model_name)
        self.IDIR=configuration_dict['IDIR']
        self.ODIR=configuration_dict['ODIR']
        self.corpus_fixed_tree_constructionorder_file=configuration_dict['corpus_fixed_tree_constructionorder_file']
#        self.corpus_fixed_tree_constructionorder_file='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/xiaojie.AstConstruction.txt'###测试用，并且要将xiaojie.int改为corpus.int.
        #self.MAX_SENTENCE_LENGTH=10000
        self.MAX_SENTENCE_LENGTH=100000
#        self.batch_size=10
        self.batch_size=configuration_dict['batch_size']###用单个句子检验网络结构的正确性。 #训练用的batch大小
        self.batch_size_using_model_notTrain=configuration_dict['batch_size_using_model_notTrain']###使用模型用的batch大小，not_train
#        self.batch_size=9###利用批样本检验网络结构的正确性。
        self.MAX_SENTENCE_LENGTH_for_Bigclonebench=configuration_dict['MAX_SENTENCE_LENGTH_for_Bigclonebench']
        self.corpus_fixed_tree_construction_parentType_weight_file=configuration_dict['corpus_fixed_tree_construction_parentType_weight_file']
        logs.log("(1)<<<<  before training RvNN,配置超参数完毕")
class costTree(object):
    def __init__(self,sentence_length=0):
        self.sl=sentence_length
        self.nodeScores=np.zeros((2*self.sl-1,1),dtype=np.double)#存储每两个节点建立重构后的重构误差。有sl个叶子节点，也有sl-1个新节点(就是两两合并后得到的节点)。
        self.collapsed_sentence = (list)(range(0,self.sl)) #这列表里面存放的是，每一次进行循环计算交叉熵时，即for j in range(0,sl-1)中的循环操作，每个位置的真正的节点编号。
        #初始化是全部叶子节点的变化，即从0到sl-1，但是随着循环的进行，叶子节点会被删除，新的父节点会加入新的循环，此时collapse就会记录真正参与循环的节点编号
        #如此以来，我们就能够知道如何从一群叶子，成长为一颗树。
        self.pp=np.zeros((2*self.sl-1,1),dtype=np.int) #记录每个子节点编号的父节点编号
class RNN_Model():
#
    def load_data(self):
        """Loads train/dev/test data and builds vocabulary."""
        logs.log("(2)>>>>  加载词向量数据和语料库")
        (self.trainCorpus,self.fullCorpus,self.trainCorpus_sentence_length,self.fullCorpus_sentence_length,self.vocabulary,self.We,self.config.max_sentence_length_train_Corpus,self.config.max_sentence_length_full_Corpus,self.train_corpus_fixed_tree_constructionorder,self.full_corpus_fixed_tree_constructionorder)=preprocess_withAST(self.config.IDIR,self.config.ODIR,self.config.corpus_fixed_tree_constructionorder_file,self.config.MAX_SENTENCE_LENGTH)
        ####注意：递归自编码器是一种无监督学习方式，其必然要在整个语料库上进行训练，这也是论文的一贯方式，但是由于网络的问题，要求我们必须筛选长度小于一定值的句子作为训练集，即train_corpus
        ####最后，使用模型计算每个句子的向量，肯定是在整个语料库上，即fullCorpus
        ####但是，我们如何判定模型是否收敛呢？所以，我们需要一个验证集。假如我们使用fullCorpus中的一部分数据集作为评估，而不是trainCorpus中的一部分数据集作为评估，那么不符合递归自编码器的思想。因为其是无监督，要求在所训练语料库上的重构误差越来越小。
        ####所以，必须使用trainCorpus作为评估。但是，trainCorpus中可能有几十万个句子，我们每次评估用这么多，不太合适。所以，我们从中划分出一部分作为验证集
        ####用在这一部分验证集的重构损失是否减小，来度量整个trainCorpus语料上的无监督递归自编码网络的重构损失是否减小。
        ####那么选取多大呢？我们根据经验，假使trainCorpus中有超过4000个句子，我们就随机选取4000个句子，这个还能处理。假使，小于4000个句子，我们就使用trainCorpus作为训练集。
        if len(self.trainCorpus)>4000:
            sentence_num=len(self.trainCorpus)
            data_idxs=list(range(sentence_num))
            shuffle(data_idxs) #打乱索引
            evalution_idxs=data_idxs[0:4000]
            self.evalutionCorpus=[self.trainCorpus[index] for index in evalution_idxs]
            self.config.max_sentence_length_evalution_Corpus=self.config.max_sentence_length_train_Corpus #仍然保持一致
            self.evalution_corpus_fixed_tree_constructionorder=[self.train_corpus_fixed_tree_constructionorder[index] for index in evalution_idxs]
        else:
            self.evalutionCorpus=self.trainCorpus
            self.config.max_sentence_length_evalution_Corpus=self.config.max_sentence_length_train_Corpus
            self.evalution_corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder
        ####
        logs.log("(2)>>>>  加载词向量数据和语料库完毕")
    def load_data_experimentID_1(self):
        ###加权
        """Loads train/dev/test data and builds vocabulary."""
        logs.log("(2)>>>>  加载词向量数据和语料库")
        
        
        (self.trainCorpus,self.trainCorpus_sentence_length,self.vocabulary,self.We,self.config.max_sentence_length_train_Corpus,self.train_corpus_fixed_tree_constructionorder,self.train_corpus_fixed_tree_parentType_weight)=preprocess_withAST_experimentID_10(self.config.IDIR,self.config.ODIR,self.config.corpus_fixed_tree_constructionorder_file,self.config.MAX_SENTENCE_LENGTH,self.config.corpus_fixed_tree_construction_parentType_weight_file)
        
        
        ####注意：递归自编码器是一种无监督学习方式，其必然要在整个语料库上进行训练，这也是论文的一贯方式，但是由于网络的问题，要求我们必须筛选长度小于一定值的句子作为训练集，即train_corpus
        ####最后，使用模型计算每个句子的向量，肯定是在整个语料库上，即fullCorpus
        ####但是，我们如何判定模型是否收敛呢？所以，我们需要一个验证集。假如我们使用fullCorpus中的一部分数据集作为评估，而不是trainCorpus中的一部分数据集作为评估，那么不符合递归自编码器的思想。因为其是无监督，要求在所训练语料库上的重构误差越来越小。
        ####所以，必须使用trainCorpus作为评估。但是，trainCorpus中可能有几十万个句子，我们每次评估用这么多，不太合适。所以，我们从中划分出一部分作为验证集
        ####用在这一部分验证集的重构损失是否减小，来度量整个trainCorpus语料上的无监督递归自编码网络的重构损失是否减小。
        ####那么选取多大呢？我们根据经验，假使trainCorpus中有超过4000个句子，我们就随机选取4000个句子，这个还能处理。假使，小于4000个句子，我们就使用trainCorpus作为训练集。
        
        
        ########第一步：根据BigCloneBench的所有ID编号对应我们语料库中的位置，取出训练用的语料库集合。
        logs.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")
        all_idMapline_pkl = './SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
        lines_for_trainCorpus=[]
        with open(all_idMapline_pkl, 'rb') as f:
            id_line_dict = pickle.load(f)
            for id in id_line_dict.keys():
                line = id_line_dict[id]
                if(line==-1):
                    continue
                lines_for_trainCorpus.append(line)
        self.lines_for_bigcloneBench=lines_for_trainCorpus
        num=len(lines_for_trainCorpus)
        logs.log('BigCloneBench中有效函数ID多少个，对应的取出我们语料库中的语料多少个.{}个'.format(num))
        ########第二步：在BigCloneBench标注的函数上进行训练。从语料库中取出训练集
        
        
        
        self.trainCorpus=[self.trainCorpus[index] for index in lines_for_trainCorpus]
        #还得配套计算训练集的其它值。
        self.train_corpus_fixed_tree_constructionorder=[self.train_corpus_fixed_tree_constructionorder[index] for index in lines_for_trainCorpus]
        self.trainCorpus_sentence_length=[self.trainCorpus_sentence_length[index] for index in lines_for_trainCorpus]
        self.train_corpus_fixed_tree_parentType_weight=[self.train_corpus_fixed_tree_parentType_weight[index] for index in lines_for_trainCorpus]
        sizes = list(map(len,self.trainCorpus))
        self.config.max_sentence_length_train_Corpus=max(sizes)
        
        self.bigCloneBench_Corpus=self.trainCorpus #根据BigCloneBench标记的函数从整个Corpus中筛选出语料库。
        self.bigCloneBench_Corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder
        self.bigCloneBench_Corpus_sentence_length=self.trainCorpus_sentence_length
        self.bigCloneBench_Corpus_max_sentence_length=self.config.max_sentence_length_train_Corpus
        self.bigCloneBench_Corpus_fixed_tree_parentType_weight=self.train_corpus_fixed_tree_parentType_weight
        logs.log('(2)>>>>  对照BigCloneBench中标注的函数,从我们的语料库中抽取语料{}个'.format(len(lines_for_trainCorpus)))      

    def load_data_experimentID_2(self):
        """Loads train/dev/test data and builds vocabulary."""
        ########加载需要计算向量树的BigCloneBench的Id编号
        
        need_vectorTree_id='./vectorTree/valid_dataset_lst.pkl'
        need_vectorTree_id_dict=self.read_from_pkl(need_vectorTree_id)
        
        logs.log("(2)>>>>  加载词向量数据和语料库")
        ###
        (self.trainCorpus,self.trainCorpus_sentence_length,self.vocabulary,self.We,self.config.max_sentence_length_train_Corpus,self.train_corpus_fixed_tree_constructionorder)=preprocess_withAST_experimentID_1(self.config.IDIR,self.config.ODIR,self.config.corpus_fixed_tree_constructionorder_file,self.config.MAX_SENTENCE_LENGTH)
        
        
        ########第一步：根据BigCloneBench的所有ID编号对应我们语料库中的位置，取出训练用的语料库集合。
        logs.log("------------------------------\n对照BigCloneBench中标注的函数，找出在我们的语料库中的编号位置")

        all_idMapline_pkl = './SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
        need_vectorTree_lines_for_trainCorpus=[]
        need_vectorTree_ids_for_trainCorpus=[]
        with open(all_idMapline_pkl, 'rb') as f:
            self.id_line_dict = pickle.load(f)
            index=0
            for id in need_vectorTree_id_dict:
                line =self.id_line_dict[id]
                if(line==-1):
                    continue
                #测试用
#                if(index==25): #取出25个作为测试用。
#                    break
                #测试用
                need_vectorTree_ids_for_trainCorpus.append(id)
                need_vectorTree_lines_for_trainCorpus.append(line)
                index+=1
        self.need_vectorTree_lines_for_trainCorpus=need_vectorTree_lines_for_trainCorpus
        self.need_vectorTree_ids_for_trainCorpus=need_vectorTree_ids_for_trainCorpus
        num=len(need_vectorTree_ids_for_trainCorpus)
        logs.log('路哥需要取出{}个ID对应的向量树'.format(num))
        ########第二步：在BigCloneBench标注的函数上进行训练。从语料库中取出训练集
        
        self.trainCorpus=[self.trainCorpus[index] for index in need_vectorTree_lines_for_trainCorpus]
        #还得配套计算训练集的其它值。
        self.train_corpus_fixed_tree_constructionorder=[self.train_corpus_fixed_tree_constructionorder[index] for index in need_vectorTree_lines_for_trainCorpus]
        self.trainCorpus_sentence_length=[self.trainCorpus_sentence_length[index] for index in need_vectorTree_lines_for_trainCorpus]
        sizes = list(map(len,self.trainCorpus))
        self.config.max_sentence_length_train_Corpus=max(sizes)
        
        self.need_vectorTree_Corpus=self.trainCorpus #根据BigCloneBench标记的函数从整个Corpus中筛选出语料库。
        self.need_vectorTree_Corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder
        self.need_vectorTree_Corpus_sentence_length=self.trainCorpus_sentence_length
        self.need_vectorTree_Corpus_max_sentence_length=self.config.max_sentence_length_train_Corpus
        
    def load_data_experimentID_3(self):
        """Loads train/dev/test data and builds vocabulary."""
        ########加载需要计算向量树的BigCloneBench的Id编号
        ###这个实验，需要盲训练。也就是不知道标记数据集。所以，我们需要从full_corpus中随机选取训练集。
        logs.log("(2)>>>>  加载词向量数据和语料库")
        ###
        (self.fullCorpus,self.fullCorpus_sentence_length,self.vocabulary,self.We,self.config.max_sentence_length_full_Corpus,self.full_corpus_fixed_tree_constructionorder,self.full_corpus_fixed_tree_parentType_weight)=preprocess_withAST_experimentID_10(self.config.IDIR,self.config.ODIR,self.config.corpus_fixed_tree_constructionorder_file,self.config.MAX_SENTENCE_LENGTH,self.config.corpus_fixed_tree_construction_parentType_weight_file)
        ###("加载了全部语料")
        ###我们暂时不考虑模型怎么训练的事情。而是先用之前在标记函数上训练的模型。
        ########接下来就是计算全部语料的所有平均向量。我们称之为full_trained_mean和faull_trained_root。然后检测大规模语料库上是否能够有效应用。
        ########第一步：根据BigCloneBench的所有ID编号对应我们语料库中的位置，取出训练用的语料库集合。
        
        
    def xiaojie_RvNN_fixed_tree(self): 
        self.add_placeholders_fixed_tree()
        self.add_loss_fixed_tree()
        
    def xiaojie_RvNN_fixed_tree_for_usingmodel(self): 
        self.add_placeholders_fixed_tree()
        self.add_loss_and_batchSentenceNodesVector_fixed_tree()
    def buqi_2DmatrixTensor(self,_2DmatrixTensor,lines,columns,targetlines,targetcolumns):
        
#        #首先在列上补齐
#    buqi_column=tf.zeros([lines,targetcolumns-columns],dtype=tf.float64)
#    _2DmatrixTensor=tf.concat([_2DmatrixTensor,buqi_column],axis=1)
#    buqi_line=tf.zeros(shape=[targetlines-lines,targetcolumns],dtype=tf.float64)
#    _2DmatrixTensor=tf.concat([_2DmatrixTensor,buqi_line],axis=0)
        
#        用tf.padding补齐
        _2DmatrixTensor=tf.pad(_2DmatrixTensor,[[0,targetlines-lines],[0,targetcolumns-columns]])

        return _2DmatrixTensor
    def modify_one_profile(self,tensor,_2DmatrixTensor,index_firstDimension,size_firstDimension,size_secondDimension,size_thirdDimension):
    ##tensor为三维矩阵
    ##首先，我们用index_firstDimenion取出整个tensor在第一维度取值index_firstDimenion的剖面，然后分为剖面左侧部分，剖面右侧部分，然后将取出的剖面替换成二维矩阵
        _2DmatrixTensor=tf.expand_dims(_2DmatrixTensor,axis=0) #扩展成为三维
        new_tensor_left=tf.slice(tensor, [0,0,0], [index_firstDimension,size_secondDimension,size_thirdDimension]) #剖面左侧部分
        new_tensor_right=tf.slice(tensor, [index_firstDimension+1,0,0], [size_firstDimension-index_firstDimension-1,size_secondDimension,size_thirdDimension]) #剖面右侧部分
        new_tensor=tf.concat([new_tensor_left,_2DmatrixTensor,new_tensor_right],0)
        return new_tensor_left,new_tensor_right,new_tensor
    def delete_one_column(self,tensor,index,numlines,numcolunms):#index也是tensor
        #tensor为二维矩阵
        #columnTensor的维度就是tensor中的一列
        new_tensor_left=tf.slice(tensor, [0, 0], [numlines, index])
        new_tensor_right=tf.slice(tensor, [0, index+1], [numlines, numcolunms-(index+1)])
        new_tensor=tf.concat([new_tensor_left,new_tensor_right],1)
        return new_tensor
    def modify_one_column(self,tensor,columnTensor,index,numlines,numcolunms):#index也是tensor
        #tensor为二维矩阵
        #columnTensor的维度就是tensor中的一列，是修改后
        #index是待修改列的列数索引
        #numlines是行数
        #numcolumns是列数
        #所有参数都是tensor
        new_tensor_left=tf.slice(tensor, [0, 0], [numlines, index])
        new_tensor_right=tf.slice(tensor, [0, index+1], [numlines, numcolunms-(index+1)])
        new_tensor=tf.concat([new_tensor_left,columnTensor,new_tensor_right],1)
        return new_tensor
    def computeloss_withAST(self,sentence,treeConstructionOrders):#计算网络权值的基础上的loss，按照指定的抽象语法树的方式去构建
        """Adds loss ops to the computational graph.
        """
        with tf.variable_scope('Composition', reuse=True):
            W1=tf.get_variable("W1",dtype=tf.float64)
            b1=tf.get_variable("b1",dtype=tf.float64)
        with tf.variable_scope('Projection',reuse=True):
            U=tf.get_variable("U",dtype=tf.float64)
            bs=tf.get_variable("bs",dtype=tf.float64)
        ones=np.ones_like(sentence)
        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
        wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
        L = wordEmbedding[:,words_indexed]#wordEmbedding的每一列对应一个单词的词向量，L的每一列对应的就是sentence的每一个单词的词向量
#        print (L)
        sl=L.shape[1]#句子的长度。
        nodes_tensor = dict()#这一个数据结构很重要，是保存每个节点的词向量的结构。包括叶子节点和非叶子节点（对于非叶子节点而言，是权值矩阵的因变量）
        for i in range(0,sl):
            nodes_tensor[i]=np.expand_dims(L[:,i],1) #前面的0到sl-1的下标，存储的就是最原始的词向量，但是我们也要将其转变为Tensor
        node_tensors_cost_tensor = dict()#每合并一个非叶子节点，得到的损失值，用tensor表示。
        
        if (sl > 1):#don't process one word sentences
            for j in range(0,sl-1):#j的范围是0到sl-2。注意，不包括sl-1。这是因为range比较特殊。一共循环sl-1次。
                #print (j)
                ###将神经网络的参数具体化
                W1_python=W1.eval()
                b1_python=b1.eval()
                U_python=U.eval()
                bs_python=bs.eval()
                treeConstructionOrder=treeConstructionOrders[:,j]
                #取出左节点编号，
                leftChild_index=treeConstructionOrder[0]-1 #注意，一定要减去1.因为文本中存储的抽象语法树的叶子节点编号是从1开始的。
                left_child=nodes_tensor[leftChild_index]
                #取出右节点编号，
                rightChild_index=treeConstructionOrder[1]-1
                right_child=nodes_tensor[rightChild_index]                    
                #取出父亲节点编号。
                parent_index=treeConstructionOrder[2]-1
                #编码出父亲节点的向量。
                child=np.concatenate((left_child,right_child), axis=0)
                p=np.tanh(np.dot(W1_python,child)+b1_python)
                p_normalization=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
                #存储父节点的词向量
                nodes_tensor[parent_index]=p_normalization
                #######
                y=np.tanh(np.dot(U_python,p_normalization)+bs_python)#根据Matlab中的源码来的，即重构后，也有一个激活的过程。
                #将Y矩阵拆分成上下部分之后，再分别进行标准化。
                [y1,y2]=np.split(y, 2, axis = 0)
                y1_normalization=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
                y2_normalization=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))	
                #论文中提出一种计算重构误差时要考虑的权重信息。具体见论文，这里暂时不实现。
                #这个权重是可以修改的。
                alpha_cat=1 
                bcat=1
                #计算重构误差矩阵
                y1c1, y2c2 = alpha_cat*(y1_normalization-left_child), bcat*(y2_normalization-right_child)
                constructionError = np.sum((y1c1*(y1_normalization-left_child)+y2c2*(y2_normalization-right_child)),axis=0)*0.5 #如果是直接使用sum函数的话，就不需要axis=0这个参数选项
                node_tensors_cost_tensor[j]=constructionError
#                print('{}+{}->{}'.format(leftChild_index,rightChild_index,parent_index))
                pass
            pass
        #计算整颗树的实际损失和用权值矩阵表示的Tensor变量损失
        treeloss=0
        for (key,value) in node_tensors_cost_tensor.items():
            treeloss=treeloss+value
        #####由于每颗树的节点数目不一样，我们需要计算平均的重构误差。
        treeloss=treeloss/(sl-1)
        return treeloss    
    def add_loss_fixed_tree(self):#计算一个batch样本的所有重构误差，固定的树结构，提供了treeConstructionOrders


        batch_trees_total_cost=tf.zeros(self.batch_len,tf.float64)
        batch_trees_total_cost=tf.expand_dims(batch_trees_total_cost,0)#扩展成为二维
        #由于batch_len是传入参数的时候才确定，故而是(1,?)类型。此时，用它来生成一个tf.Variable会提示shape必须被指定。所以下面这句代码是错误的。
#        batch_trees_total_cost=tf.Variable(batch_trees_total_cost,trainable=False)#batch_trees_total_cost不能在一个不确定的shape上构建variable
        self.numlines_tensor3=tf.constant(1,dtype=tf.int32)
        self.numcolunms_tensor3=self.batch_len[0]
        #Tensorflow方式的循环
        i=tf.constant(0,dtype=tf.int32)
        batch_len=self.batch_len[0]#batch_len的shape是(1,)
        loop_cond = lambda a,b,c: tf.less(a,batch_len)
        tfPrint=tf.constant(0)#专门为了tf.Print输出准备的。
        loop_vars=[i,batch_trees_total_cost,tfPrint]
        def _recurrence(i,batch_trees_total_cost,tfPrint):#循环的目的是处理每一个样本
            #处理每一个样本，即处理placeholder中的数据。
            self.sentence_embeddings=tf.gather(self.input,i,axis=0) #取出第i个样本
            self.sentence_length=self.batch_real_sentence_length[i] #取出第i个样本的句子长度
            self.treeConstructionOrders=self.batch_treeConstructionOrders[i] #取出第i个句子，用于构建树的顺序，顺序的列表。
            self.sentence_parentTypes_weight=self.batch_sentence_parentTypes_weight[i] #取出第i个句子，所有非叶子节点的权重信息。
           ####权重模型才会用到，下面是所有非叶子节点的TF-IDF权重，第一种权重模式。
            self.sentence_parentTypes_weight=self.batch_sentence_parentTypes_weight[i] #取出第i个句子，所有非叶子节点的权重信息。
            ####权重模型才会用到，这是第二种权重模式
            self.treechildparentweight=self.batch_childparentweight[i] #取出第i个句子，用于构建树的顺序中各个节点的权重。

##########################################################################################            
            #贪心方式，没有提供treeConstructionOrders。需要根据传入的数据和权值矩阵，自动化构建出treeConstructionOrders
            #treeConstructionOrders=self.treeConstructionOrders[i] #取出第i个句子，用于构建树的顺序。
##########################################################################################            
            ###############
            #I通过下面的方式得到的nodes_tensor的维度是(self.config.embed_size,?)。因为sentence_length是一个tensor
            #nodes_tensor用于存储句子构建的整颗树的每一个节点的词向量。
            tree_node_size=2*self.sentence_length-1
            nodes_tensor=tf.zeros(tree_node_size,tf.float64)
            nodes_tensor=tf.expand_dims(nodes_tensor,0)#扩展成为二维
            nodes_tensor=tf.tile(nodes_tensor,(self.config.embed_size,1))#张量在维度上复制。得到[embed_size,tree_node_size]矩阵
            ###############
            #II先将叶子节点的词向量全部放入nodes_tensor中。需要使用tensorflow的一个循环体。
            self.numlines_tensor=tf.constant(self.config.embed_size,dtype=tf.int32)
            self.numcolunms_tensor=tree_node_size
            ii=tf.constant(0,dtype=tf.int32)
            loop__cond=lambda a,b: tf.less(a,self.sentence_length)
            loop__vars=[ii,nodes_tensor]
            def __recurrence(ii,nodes_tensor):
                #前面的0到sentence_length-1的下标，存储的就是最原始的词向量，但是我们也要将其转变为Tensor
                new_column_tensor=tf.expand_dims(self.sentence_embeddings[:,ii],1)
                nodes_tensor=self.modify_one_column(nodes_tensor,new_column_tensor,ii,self.numlines_tensor,self.numcolunms_tensor)
                ii=tf.add(ii,1)
                return ii,nodes_tensor
            ii,nodes_tensor=tf.while_loop(loop__cond,__recurrence,loop__vars,parallel_iterations=1)
#            self.nodes_tensor=tf.identity/(nodes_tensor)
            #为什么要这样做，是因为同一个tensor不能进入两个while循环之中。
            ###############
            #将每一次构建父亲节点的重构误差存入node_tensors_cost_tensor
            node_tensors_cost_tensor=tf.zeros(self.sentence_length-1,tf.float64)
            node_tensors_cost_tensor=tf.expand_dims(node_tensors_cost_tensor,0)#扩展成为二维
            self.numlines_tensor2=tf.constant(1,dtype=tf.int32)
            self.numcolunms_tensor2=self.sentence_length-1
            
            #IIII通过计算将非叶子节点的词向量也放入nodes_tensor中。
            iiii=tf.constant(0,dtype=tf.int32)
            loop____cond = lambda a,b,c,d: tf.less(a,self.sentence_length-1)#iiii的范围是0到sl-2。注意，不包括sl-1。这是因为只需要计算sentence_length-1次，就能构建出一颗树
            loop____vars=[iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint]
            def ____recurrence(iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint):#循环的目的是直接根据确定的树结构来构建树，不同于Greedy算法。
                ###
                #根据固定的树结构来搭建树               
                ###     
                treeConstructionOrder=self.treeConstructionOrders[:,iiii] #取出第iii次构建树的方式
                #取出左节点编号，
                leftChild_index=treeConstructionOrder[0]-1 #注意，一定要减去1.因为文本中存储的抽象语法树的叶子节点编号是从1开始的。
                left_child_tensor=nodes_tensor[:,leftChild_index]
                #取出右节点编号，
                rightChild_index=treeConstructionOrder[1]-1
                right_child_tensor=nodes_tensor[:,rightChild_index]                    
                #取出父亲节点编号。
                parent_index=treeConstructionOrder[2]-1
                #编码出父亲节点的向量。





########################################################    
#                childparentweight=self.treechildparentweight[:,iiii]
#                leftchild_weight=childparentweight[0]
#                rightchild_weight=childparentweight[1]
#                
#                ####选择编和解码的矩阵参数
#                ####十层以内，我们用一个。每十层用一个。
#                encode_decode_index=tf.to_int32(iiii/10) ###0到9都会用0。
#                encode_decode_index=tf.cond(tf.less(encode_decode_index,29),lambda:encode_decode_index,lambda:29)
#                #但是最多是30个，这是我们设置的。所以，encode_decode_index最多为29。从0开始编号
#                ####选择编和解码的矩阵参数
                
                
#                leftchild_weight=tf.cond(tf.equal(x,y),lambda:tf.constant(0.5,tf.float64),lambda:x/(x+y))
#                rightchild_weight=tf.cond(tf.equal(x,y),lambda:tf.constant(0.5,tf.float64),lambda:y/(x+y))
#                left_child_tensor2=tf.multiply(left_child_tensor,(leftchild_weight)/(leftchild_weight+rightchild_weight))
#                right_child_tensor2=tf.multiply(right_child_tensor,(rightchild_weight)/(leftchild_weight+rightchild_weight))
#                
#                
#                
                
                new_parent_tensor=tf.multiply(tf.add(left_child_tensor,right_child_tensor),0.5) #直接求中心点
                new_parent_tensor=tf.expand_dims(new_parent_tensor,1)#变成二维的

                nodes_tensor=self.modify_one_column(nodes_tensor,new_parent_tensor,parent_index,self.numlines_tensor,self.numcolunms_tensor)
#                xishu=self.sentence_parentTypes_weight[iiii] #取出第iii次构建的非叶子节点类型的权重。

                
#################################################################################






                
#################################################################################
                print_info=tf.Print(iiii,[iiii],"\niiii:")#专门为了调试用，输出相关信息。
                tfPrint=print_info+tfPrint
                tfPrint=print_info+tfPrint
                print_info=tf.Print(leftChild_index,[leftChild_index],"\nleftChild_index:",summarize=100)#专门为了调试用，输出相关信息。
                tfPrint=tf.to_int32(print_info)+tfPrint#一种不断输出tf.Print的方式，注意tf.Print的返回值。
                print_info=tf.Print(rightChild_index,[rightChild_index],"\nrightChild_index:",summarize=100)#专门为了调试用，输出相关信息。
                tfPrint=tf.to_int32(print_info)+tfPrint#一种不断输出tf.Print的方式，注意tf.Print的返回值。
                print_info=tf.Print(parent_index,[parent_index],"\nparent_index:",summarize=100)#专门为了调试用，输出相关信息。
                tfPrint=tf.to_int32(print_info)+tfPrint#一种不断输出tf.Print的方式，注意tf.Print的返回值。
################################################################################
                ####进入下一次循环
                iiii=tf.add(iiii,1)
#                columnLinesOfL=tf.subtract(columnLinesOfL,1) #在上面的循环体中已经执行了，没有必要再执行。
                return iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint
            iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint=tf.while_loop(loop____cond,____recurrence,loop____vars,parallel_iterations=1)
            pass
            self.node_tensors_cost_tensor=tf.identity(node_tensors_cost_tensor)
            self.nodes_tensor=tf.identity(nodes_tensor)
#            one_tree_total_cost=tf.reduce_sum(self.node_tensors_cost_tensor)#对于每一行而言，每一列就是节点的重构误差。reduce_sum之后，第一是降维，第二是每一行的所有元素值相加。
            ##########由于前面已经乘上了权重系数。而且权重系数的和为1。是所有非叶子节点出权重的和作为分母。所以，乘上以后，就相当于加权平均了。
            one_tree_total_cost=tf.reduce_mean(self.node_tensors_cost_tensor) 
#            one_tree_total_cost=tf.reduce_sum(self.node_tensors_cost_tensor) 
            ################由于前面已经乘上了权重系数。而且是所有非叶子节点出权重的和作为分母。所以，乘上以后，就相当于加权平均了。
            
            one_tree_total_cost=tf.expand_dims(tf.expand_dims(one_tree_total_cost,0),1)
            batch_trees_total_cost=self.modify_one_column(batch_trees_total_cost,one_tree_total_cost,i,self.numlines_tensor3,self.numcolunms_tensor3)
            i=tf.add(i,1)
            return i,batch_trees_total_cost,tfPrint
        i,batch_trees_total_cost,tfPrint=tf.while_loop(loop_cond,_recurrence,loop_vars,parallel_iterations=10)#可以批处理
        self.tfPrint=tfPrint
        with tf.name_scope('loss'):
            #ce_loss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))#如果用有监督，这里也是要计算的。
            
            ###在多棵树之间仍然是求平均。            
            self.batch_constructionError=tf.reduce_mean(batch_trees_total_cost)#应该是求平均值。
            self.tensorLoss_fixed_tree=self.batch_constructionError
        return self.tensorLoss_fixed_tree
    def add_loss_and_batchSentenceNodesVector_fixed_tree(self):#计算一个batch样本的所有重构误差，固定的树结构，提供了treeConstructionOrders

#        total_cost=tf.constant(0,dtype=tf.float64)
        batch_trees_total_cost=tf.zeros(self.batch_len,tf.float64)
        batch_trees_total_cost=tf.expand_dims(batch_trees_total_cost,0)#扩展成为二维
        #由于batch_len是传入参数的时候才确定，故而是(1,?)类型。此时，用它来生成一个tf.Variable会提示shape必须被指定。所以下面这句代码是错误的。
#        batch_trees_total_cost=tf.Variable(batch_trees_total_cost,trainable=False)#batch_trees_total_cost不能在一个不确定的shape上构建variable
##############################
        ####这个不是为了训练来的，而是为了快速使用模型去计算几十个句子对应的递归自编码器编码出的向量。
        ####我们为每一批句子保存每个句子对应的最终vector。
        #####
        #根据self.batch_real_sentence_length中计算batch中最长的句子
        
        max_sentence_length_in_batch=self.batch_real_sentence_length[tf.argmax(self.batch_real_sentence_length)]
        max_sentence_nodesize_in_batch=2*max_sentence_length_in_batch-1
        matrix_element_num=self.batch_len[0]*self.config.embed_size*max_sentence_nodesize_in_batch
        batch_sentence_vectors=tf.zeros(matrix_element_num,tf.float64)
        batch_sentence_vectors=tf.reshape(batch_sentence_vectors,[self.batch_len[0],self.config.embed_size,max_sentence_nodesize_in_batch])
        self.size_firstDimension=self.batch_len[0]
        self.size_secondDimension=tf.constant(self.config.embed_size,dtype=tf.int32) #转换成tensor
        self.size_thirdDimension=max_sentence_nodesize_in_batch
        
        #在下面的循环中，对batch_sentence_nodes_vectors进行修改，我们要求修改的是特定剖面的信息（对应一个batch的一个样本）。即将下面循环中的nodes_tensor通过零补齐填写到batch_sentence_nodes_vectors中去。
##############################
        self.numlines_tensor3=tf.constant(1,dtype=tf.int32)#用于操作batch_trees_total_cost
        self.numcolunms_tensor3=self.batch_len[0] #用于操作batch_trees_total_cost
        
        self.numlines_tensor4=tf.constant(self.config.embed_size,dtype=tf.int32)#用于操作batch_sentence_vectors
        self.numcolunms_tensor4=self.batch_len[0] #用于操作batch_sentence_vectors
        
        #Tensorflow方式的循环
        i=tf.constant(0,dtype=tf.int32)
        batch_len=self.batch_len[0]#batch_len的shape是(1,)
        loop_cond = lambda a,b,c,d: tf.less(a,batch_len)
        tfPrint=tf.constant(0)#专门为了tf.Print输出准备的。
        loop_vars=[i,batch_trees_total_cost,tfPrint,batch_sentence_vectors]
        def _recurrence(i,batch_trees_total_cost,tfPrint,batch_sentence_vectors):#循环的目的是处理每一个样本
            #处理每一个样本，即处理placeholder中的数据。
            self.sentence_embeddings=tf.gather(self.input,i,axis=0) #取出第i个样本
            self.sentence_length=self.batch_real_sentence_length[i] #取出第i个样本的句子长度
            self.treeConstructionOrders=self.batch_treeConstructionOrders[i] #取出第i个句子，用于构建树的顺序，顺序的列表。
           ####权重模型才会用到，下面是所有非叶子节点的TF-IDF权重，第一种权重模式。
            self.sentence_parentTypes_weight=self.batch_sentence_parentTypes_weight[i] #取出第i个句子，所有非叶子节点的权重信息。
            ####权重模型才会用到，这是第二种权重模式
            self.treechildparentweight=self.batch_childparentweight[i] #取出第i个句子，用于构建树的顺序中各个节点的权重。

##########################################################################################            
            #贪心方式，没有提供treeConstructionOrders。需要根据传入的数据和权值矩阵，自动化构建出treeConstructionOrders
            #treeConstructionOrders=self.treeConstructionOrders[i] #取出第i个句子，用于构建树的顺序。
##########################################################################################            
            ###############
            #I通过下面的方式得到的nodes_tensor的维度是(self.config.embed_size,?)。因为sentence_length是一个tensor
            #nodes_tensor用于存储句子构建的整颗树的每一个节点的词向量。
            tree_node_size=2*self.sentence_length-1
            nodes_tensor=tf.zeros(tree_node_size,tf.float64)
            nodes_tensor=tf.expand_dims(nodes_tensor,0)#扩展成为二维
            nodes_tensor=tf.tile(nodes_tensor,(self.config.embed_size,1))#张量在维度上复制。得到[embed_size,tree_node_size]矩阵
            ###############
            #II先将叶子节点的词向量全部放入nodes_tensor中。需要使用tensorflow的一个循环体。
            self.numlines_tensor=tf.constant(self.config.embed_size,dtype=tf.int32)
            self.numcolunms_tensor=tree_node_size
            ii=tf.constant(0,dtype=tf.int32)
            loop__cond=lambda a,b: tf.less(a,self.sentence_length)
            loop__vars=[ii,nodes_tensor]
            def __recurrence(ii,nodes_tensor):
                #前面的0到sentence_length-1的下标，存储的就是最原始的词向量，但是我们也要将其转变为Tensor
                new_column_tensor=tf.expand_dims(self.sentence_embeddings[:,ii],1)
                
#                ####我们在这里对其进行标准化
#                ####相当于对叶子节点的词向量进行模长为1的变化。
#                new_column_tensor=self.normalization(new_column_tensor) #存储标准化后的向量。即模长为1.
#                ####
                
                nodes_tensor=self.modify_one_column(nodes_tensor,new_column_tensor,ii,self.numlines_tensor,self.numcolunms_tensor)
                ii=tf.add(ii,1)
                return ii,nodes_tensor
            ii,nodes_tensor=tf.while_loop(loop__cond,__recurrence,loop__vars,parallel_iterations=1)
#            self.nodes_tensor=tf.identity/(nodes_tensor)
            #为什么要这样做，是因为同一个tensor不能进入两个while循环之中。
            ###############
            #将每一次构建父亲节点的重构误差存入node_tensors_cost_tensor
            node_tensors_cost_tensor=tf.zeros(self.sentence_length-1,tf.float64)
            node_tensors_cost_tensor=tf.expand_dims(node_tensors_cost_tensor,0)#扩展成为二维
            self.numlines_tensor2=tf.constant(1,dtype=tf.int32)
            self.numcolunms_tensor2=self.sentence_length-1
            
            #IIII通过计算将非叶子节点的词向量也放入nodes_tensor中。
            iiii=tf.constant(0,dtype=tf.int32)
            loop____cond = lambda a,b,c,d: tf.less(a,self.sentence_length-1)#iiii的范围是0到sl-2。注意，不包括sl-1。这是因为只需要计算sentence_length-1次，就能构建出一颗树
            loop____vars=[iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint]
            def ____recurrence(iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint):#循环的目的是直接根据确定的树结构来构建树，不同于Greedy算法。
                ###
                #根据固定的树结构来搭建树               
                ###     
                treeConstructionOrder=self.treeConstructionOrders[:,iiii] #取出第iii次构建树的方式
                #取出左节点编号，
                leftChild_index=treeConstructionOrder[0]-1 #注意，一定要减去1.因为文本中存储的抽象语法树的叶子节点编号是从1开始的。
                left_child_tensor=nodes_tensor[:,leftChild_index]
                #取出右节点编号，
                rightChild_index=treeConstructionOrder[1]-1
                right_child_tensor=nodes_tensor[:,rightChild_index]                    
                #取出父亲节点编号。
                parent_index=treeConstructionOrder[2]-1
                #编码出父亲节点的向量。
############################老式的编码方式############################
#                child_tensor=tf.concat([left_child_tensor,right_child_tensor],axis=0)#注意，1.0版本之后的concat是数字在后，调整一下位置即可。
#                child_tensor=tf.expand_dims(child_tensor,1)#变成二维的
#                p_tensor=tf.tanh(tf.add(tf.matmul(self.W1,child_tensor),self.b1)) #此处不像之前进行Greedy的时候，将W和b1都用eval()得到具体值，这里直接都是Tensor的计算
#                new_parent_tensor=self.normalization(p_tensor) #存储标准化后的词向量,这里是Tensor
#                #第一步：将新的非叶子节点的词向量存入nodes_tensor
#                nodes_tensor=self.modify_one_column(nodes_tensor,new_parent_tensor,parent_index,self.numlines_tensor,self.numcolunms_tensor)
#                #第二步：记录重构的损失值
#                y=tf.matmul(self.U,new_parent_tensor)+self.bs#根据Matlab中的源码来的，即重构后，也有一个激活的过程。
#                #将Y矩阵拆分成上下部分之后，再分别进行标准化。
#                columnlines_y=y.shape[1].value
#                (y1,y2)=self.split_by_row(y,columnlines_y)
#                #论文中提出一种计算重构误差时要考虑的权重信息。具体见论文，这里暂时不实现。
#                #这个权重是可以修改的。
#                alpha_cat=1 
#                bcat=1
#                left_child_tensor=tf.expand_dims(left_child_tensor,1)#变成二维的
#                right_child_tensor=tf.expand_dims(right_child_tensor,1)#变成二维的
#                y1c1=tf.subtract(y1,left_child_tensor)
#                y2c2=tf.subtract(y2,right_child_tensor)                
#                constructionError=self.constructionError(y1c1,y2c2,alpha_cat,bcat)
#                constructionError=tf.expand_dims(constructionError,1)
#                node_tensors_cost_tensor=self.modify_one_column(node_tensors_cost_tensor,constructionError,iiii,self.numlines_tensor2,self.numcolunms_tensor2)
############################老式的编码方式############################
########################################################    
                childparentweight=self.treechildparentweight[:,iiii]
                leftchild_weight=childparentweight[0]
                rightchild_weight=childparentweight[1]
#                
#                ####选择编和解码的矩阵参数
#                ####十层以内，我们用一个。每十层用一个。
#                encode_decode_index=tf.to_int32(iiii/10) ###0到9都会用0。
#                encode_decode_index=tf.cond(tf.less(encode_decode_index,29),lambda:encode_decode_index,lambda:29)
#                #但是最多是30个，这是我们设置的。所以，encode_decode_index最多为29。从0开始编号
#                ####选择编和解码的矩阵参数
                
                
#                leftchild_weight=tf.cond(tf.equal(x,y),lambda:tf.constant(0.5,tf.float64),lambda:x/(x+y))
#                rightchild_weight=tf.cond(tf.equal(x,y),lambda:tf.constant(0.5,tf.float64),lambda:y/(x+y))
                
                #修改时间2019年9月9日20:59:06
                ####错误代码，删除
                #我们在这里进行模为1的标准化。即长度变为1.
#                left_child_tensor=self.normalization_2(left_child_tensor) ###一维的用self.normalization_2
#                right_child_tensor=self.normalization_2(right_child_tensor)
                ####错误代码，删除
                
                #修改时间2019年9月9日20:59:06
                
                left_child_tensor2=tf.multiply(left_child_tensor,(leftchild_weight)/(leftchild_weight+rightchild_weight))
                right_child_tensor2=tf.multiply(right_child_tensor,(rightchild_weight)/(leftchild_weight+rightchild_weight))
  
                new_parent_tensor=tf.add(left_child_tensor2,right_child_tensor2)
#                new_parent_tensor=tf.multiply(tf.add(left_child_tensor,right_child_tensor),0.5) #直接求中心点
                new_parent_tensor=tf.expand_dims(new_parent_tensor,1)#变成二维的
                ###修改
                #虽然叶子节点对应的词向量不是模为1。我们这里将非叶子节点的向量全部变为模为1.
                new_parent_tensor=self.normalization(new_parent_tensor) #存储标准化后的向量。即模长为1.
                ####
                nodes_tensor=self.modify_one_column(nodes_tensor,new_parent_tensor,parent_index,self.numlines_tensor,self.numcolunms_tensor)
#                xishu=self.sentence_parentTypes_weight[iiii] #取出第iii次构建的非叶子节点类型的权重。

                
#################################################################################


               
                
#################################################################################
                print_info=tf.Print(iiii,[iiii],"\niiii:")#专门为了调试用，输出相关信息。
                tfPrint=print_info+tfPrint
                tfPrint=print_info+tfPrint
                print_info=tf.Print(leftChild_index,[leftChild_index],"\nleftChild_index:",summarize=100)#专门为了调试用，输出相关信息。
                tfPrint=tf.to_int32(print_info)+tfPrint#一种不断输出tf.Print的方式，注意tf.Print的返回值。
                print_info=tf.Print(rightChild_index,[rightChild_index],"\nrightChild_index:",summarize=100)#专门为了调试用，输出相关信息。
                tfPrint=tf.to_int32(print_info)+tfPrint#一种不断输出tf.Print的方式，注意tf.Print的返回值。
                print_info=tf.Print(parent_index,[parent_index],"\nparent_index:",summarize=100)#专门为了调试用，输出相关信息。
                tfPrint=tf.to_int32(print_info)+tfPrint#一种不断输出tf.Print的方式，注意tf.Print的返回值。
################################################################################
                ####进入下一次循环
                iiii=tf.add(iiii,1)
#                columnLinesOfL=tf.subtract(columnLinesOfL,1) #在上面的循环体中已经执行了，没有必要再执行。
                return iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint
            iiii,node_tensors_cost_tensor,nodes_tensor,tfPrint=tf.while_loop(loop____cond,____recurrence,loop____vars,parallel_iterations=1)
            pass
            self.node_tensors_cost_tensor=tf.identity(node_tensors_cost_tensor)
            self.nodes_tensor=tf.identity(nodes_tensor)
#            one_tree_total_cost=tf.reduce_sum(self.node_tensors_cost_tensor)#对于每一行而言，每一列就是节点的重构误差。reduce_sum之后，第一是降维，第二是每一行的所有元素值相加。
            
################计算批的损失
            one_tree_total_cost=tf.reduce_mean(self.node_tensors_cost_tensor) 
            one_tree_total_cost=tf.expand_dims(tf.expand_dims(one_tree_total_cost,0),1)
            batch_trees_total_cost=self.modify_one_column(batch_trees_total_cost,one_tree_total_cost,i,self.numlines_tensor3,self.numcolunms_tensor3)
################计算批的损失       
################计算批的每个样本的最终向量表示
            #####这里计算的是向量树。也就是说，一个样本，对应一个句子，一个句子对应一颗树，一颗树有2*n-1个节点（n为句子的长度），那么就要保存2*n-1个向量。在nodes_tensor中。
            #####首先将nodes_tensor补齐
            lines=self.numlines_tensor #self.numlines_tensor=tf.constant(self.config.embed_size,dtype=tf.int32)
            columns=self.numcolunms_tensor 
            targetlines=self.size_secondDimension # self.size_secondDimension=tf.constant(self.config.embed_size,tf.int32) #转换成tensor
            targetcolumns=self.size_thirdDimension
            
####################测试.一个解决不了的困惑。暂时先不管了
#            a_tensor=tf.zeros([lines,1000],dtype=tf.int32)
#            b_tensor=tf.zeros([targetlines,1000],dtype=tf.int32)
#            #####其实，lines和targetlines是相同的，但是测试的结果是，a_tensor是[300,1000]维度，而b_tensor是[?,1000]维度
#            #####
####################测试.一个解决不了的困惑。暂时先不管了
            nodes_tensor=self.buqi_2DmatrixTensor(nodes_tensor,lines,columns,targetlines,targetcolumns)
            ####由于buqi_2DmatrixTensor以后，nodes_tensor的第一维度（确定值）变成（？）所以，我们reshape一下
            nodes_tensor=tf.reshape(nodes_tensor,[self.config.embed_size,targetcolumns])
            ####
            
            #####紧接着将nodes_tensor作为一个剖面，放入三维矩阵batch_sentence_vectors中去
            index_firstDimension=i #该批中第i个样本
            size_firstDimension=self.size_firstDimension
            size_secondDimension=self.size_secondDimension
            size_thirdDimension=self.size_thirdDimension
            _,_,batch_sentence_vectors=self.modify_one_profile(batch_sentence_vectors,nodes_tensor,index_firstDimension,size_firstDimension,size_secondDimension,size_thirdDimension)
            
            ####修改完毕
################计算批的每个样本的最终向量表示
            i=tf.add(i,1)
            return i,batch_trees_total_cost,tfPrint,batch_sentence_vectors
        i,batch_trees_total_cost,tfPrint,batch_sentence_vectors=tf.while_loop(loop_cond,_recurrence,loop_vars,parallel_iterations=10)#可以批处理
        self.tfPrint=tfPrint
        self.batch_sentence_vectors=tf.identity(batch_sentence_vectors)
        with tf.name_scope('loss'):
            #ce_loss=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels))#如果用有监督，这里也是要计算的。
            self.batch_constructionError=tf.reduce_mean(batch_trees_total_cost)#应该是求平均值。
            self.tensorLoss_fixed_tree=self.batch_constructionError
        return self.tensorLoss_fixed_tree,self.batch_sentence_vectors
    def add_placeholders_fixed_tree(self):
#        dim1=self.config.batch_size
        dim2=self.config.embed_size
        self.input = tf.placeholder(tf.float64,[None,dim2,None],name='input') #存放输入的句子的词向量矩阵，第三维度是不确定的，因为句子的长度不确定。
        #第一种情况，input是整棵树的节点数目，假如句子长度50，那么有50个叶子节点和49个非叶子节点，然后None就应该实际传入99。后面49个非叶子节点的词向量是在权值矩阵的基础上计算出来的，在输入过程中无法确定，需要在输入时补0.
        #第二种情况，input就是叶子节点的数目
        #这里，input是叶子节点的数目。
############################################################################################## 
        self.batch_treeConstructionOrders=tf.placeholder(tf.int32,[None,3,None],name='treeConstructionOrders')
        #treeConstructionOrder保留的是依据单词编号构建树的顺序。每一行的三个元素 ，比如长度为50的句子，那么第一行假如是0,1,50.就是第0个单词和第1个单词，先合并出一个非叶子节点，编号为50
        #之前一开始的时候，是想将每一个句子的Greedy的方式传入的。但是这种适合树已经固定的方式。而Greedy方式必须是每一次根据具体的W权重，去确定最佳的结合方式，所以，就需要在网络中得出，而不是用feed_dict传入。
##############################################################################################        
        #指定None其实就是numpy的shape(?,)其实就是一维列表。比如[1,2]的shape就是(2,)。
        self.batch_real_sentence_length=tf.placeholder(tf.int32,[None],name='batch_real_sentence_length')#存入一批样本，每个样本的真实句子长度。
        self.batch_len=tf.placeholder(tf.int32,shape=(1,),name='batch_len')
        ##存入批的权重信息。
        self.batch_sentence_parentTypes_weight=tf.placeholder(tf.float64,[None,None],name='batch_sentence_parentType_weights')
        #为什么将第一个维度都是定义为None，就是为了处理零散数据。我们的数据是以self.config.batch_size对训练数据进行划分的，但是训练数据的数目不一定能够被batch_size整除，因此，我们要求传入的batch数据，可以是不定长度的。但是大部分批都是batch_size大小，少量小于这个。
        #########################
        ##########下面是第二种权重。就是树所有节点的权重
        self.batch_childparentweight=tf.placeholder(tf.float64,[None,3,None],name='treeChildParentWeights')
        #########################    
    
    def normalization(self,tensor):#对每一列的向量进行标准化。
        
        numlines=tensor.shape[0].value
        tensor_pow_2=tf.pow(tensor,2)
        reduce_sum=tf.reduce_sum(tensor_pow_2,0)
        reduce_sum_2=tf.expand_dims(reduce_sum, 0)#必须扩展到二维，才能使用tf.tile
#        print (reduce_sum_2.get_shape())
        tensor_divide_by=tf.tile(tf.sqrt(reduce_sum_2),(numlines,1))
        normalization_tensor=tf.divide(tensor,tensor_divide_by)
#        print (tensor_divide_by.get_shape())
#        print (tensor_divide_by.eval())
#        print (reduce_sum.eval())
#        print (reduce_sum.get_shape())
#        print (normalization_tensor.get_shape())
#        print (normalization_tensor.eval())
        return normalization_tensor
    def normalization_2(self,tensor):#将向量变为模为1
        b=tf.pow(tensor,2)
        b2=tf.reduce_sum(b)
        divide_by=tf.sqrt(b2)
        c=tf.divide(tensor,divide_by)
        return c
        
        
    def split_by_row(self,tensor,numcolunms):#将矩阵拆分为上下两个矩阵，矩阵的行数必须是2的整数倍
        numlines=tensor.shape[0].value
#        numcolunms=tensor.shape[1].value #传入的可能是一个?，这个时候计算的就是None
        split_by_row_tensor=tf.slice(tensor, [0, 0], [(int)(numlines/2), numcolunms])
        split_by_row_tensor_2=tf.slice(tensor, [(int)(numlines/2), 0], [(int)(numlines/2), numcolunms])
#        print (split_by_row_tensor.eval())
#        print (split_by_row_tensor_2.eval())
        pass
        return (split_by_row_tensor,split_by_row_tensor_2)
    
    def constructionError(self,tensor1,tensor2,alpha_cat,bcat):#这个函数仅仅适用于这个程序和RAE的这个算法
        tensor1_sum=tf.multiply(tf.reduce_sum(tf.pow(tensor1,2),0),alpha_cat)
        tensor2_sum=tf.multiply(tf.reduce_sum(tf.pow(tensor2,2),0),bcat)
#        print (tensor1_sum.eval())
#        print (tensor2_sum.eval())
        constructionErrorMatrix=tf.multiply(tf.add(tensor1_sum,tensor2_sum),0.5)
#        print (constructionErrorMatrix.eval())
        return constructionErrorMatrix
    def training(self, loss):
        train_op = None
        opt=tf.train.GradientDescentOptimizer(self.config.lr)
        train_op=opt.minimize(loss)
        return train_op

    def __init__(self, config, experimentID=None):
        ####################在这里可以快速调试网络，正常执行时要注释掉
#        with tf.Graph().as_default():       
#            self.xiaojie_RvNN_fixed_tree_for_usingmodel()
####################在这里可以快速调试网络，正常执行时要注释掉
        if(experimentID==None):
            self.config = config
            self.load_data()
        elif(experimentID==1):#实验1 计算所有树的平均/root
            self.config = config
            self.load_data_experimentID_1()
        elif(experimentID==2):#实验2 配合后续树卷积，拿到向量树
            self.config = config
            self.load_data_experimentID_2()
        elif(experimentID==3):#实验2 配合后续树卷积，拿到向量树
            self.config = config
            self.load_data_experimentID_3()
    #def computelossAndVector_no_tensor_all_corpus(self,corpus):#对于计算整个语料库每个句子的值时，如果每单个句子再一个一个的算，会非常的慢。
    def computelossAndVector_no_tensor_withAST(self, sentence,treeConstructionOrders):#待修改//计算单个句子的。并且无法使用GPU加速。
        """学习出的模型，权值矩阵是固定的。我们用固定树结构计算出总损失，同时计算出每个节点的Vector
        """
        ones=np.ones_like(sentence)
        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
        wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
        L = wordEmbedding[:,words_indexed]#wordEmbedding的每一列对应一个单词的词向量，L的每一列对应的就是sentence的每一个单词的词向量
        sl=L.shape[1]#句子的长度。
        
        node_vectors = dict()#这一个数据结构很重要，是保存每个节点的词向量的结构。包括叶子节点和非叶子节点（对于非叶子节点而言，是权值矩阵的因变量）
        for i in range(0,sl):
            node_vectors[i]=np.expand_dims(L[:,i],1) #前面的0到sl-1的下标，存储的就是最原始的词向量
        node_tensors_cost_tensor = dict()#每合并一个非叶子节点，得到的损失值，用tensor表示。
        the_last_node_vector=None
        if (sl > 1):#don't process one word sentences
            for j in range(0,sl-1):#j的范围是0到sl-2。注意，不包括sl-1。这是因为range比较特殊。一共循环sl-1次。
                 #print (j)
                ###将神经网络的参数具体化

                treeConstructionOrder=treeConstructionOrders[:,j]
                #取出左节点编号，
                leftChild_index=treeConstructionOrder[0]-1 #注意，一定要减去1.因为文本中存储的抽象语法树的叶子节点编号是从1开始的。
                left_child=node_vectors[leftChild_index]
                #取出右节点编号，
                rightChild_index=treeConstructionOrder[1]-1
                right_child=node_vectors[rightChild_index]                    
                #取出父亲节点编号。
                parent_index=treeConstructionOrder[2]-1
                #编码出父亲节点的向量。
                ########################################################    
                
                new_parent=np.multiply(np.add(left_child,right_child),0.5) #直接求中心点
                node_vectors[parent_index]=new_parent
                ########################################################    


                
                
                
#                child=np.concatenate((left_child,right_child), axis=0)
#                p=np.tanh(np.dot(W1_python,child)+b1_python)
#                p_normalization=p/(np.linalg.norm(p,axis=0)*np.ones(p.shape))
#                #存储父节点的词向量
#                node_vectors[parent_index]=p_normalization
#                the_last_node_vector=p_normalization
#                #######
#                y=np.tanh(np.dot(U_python,p_normalization)+bs_python)#根据Matlab中的源码来的，即重构后，也有一个激活的过程。
#                #将Y矩阵拆分成上下部分之后，再分别进行标准化。
#                [y1,y2]=np.split(y, 2, axis = 0)
#                y1_normalization=y1/(np.linalg.norm(y1,axis=0)*np.ones(y1.shape))
#                y2_normalization=y2/(np.linalg.norm(y2,axis=0)*np.ones(y2.shape))	
#                #论文中提出一种计算重构误差时要考虑的权重信息。具体见论文，这里暂时不实现。
#                #这个权重是可以修改的。
#                alpha_cat=1 
#                bcat=1
#                #计算重构误差矩阵
#                y1c1, y2c2 = alpha_cat*(y1_normalization-left_child), bcat*(y2_normalization-right_child)
#                constructionError = np.sum((y1c1*(y1_normalization-left_child)+y2c2*(y2_normalization-right_child)),axis=0)*0.5 #如果是直接使用sum函数的话，就不需要axis=0这个参数选项
#                node_tensors_cost_tensor[j]=constructionError
#                print('{}+{}->{}'.format(leftChild_index,rightChild_index,parent_index))
                pass
            root_vector=node_vectors[2*sl-2]
            pass
        treeloss=0
        for (key,value) in node_tensors_cost_tensor.items():
            treeloss=treeloss+value
        return (treeloss,node_vectors,root_vector,the_last_node_vector)#第一个是重构误差 第2个是返回最后计算的每个节点，包括非叶子节点的vector。第3个是树的最顶层节点的vector
#    def run_epoch(self,sess,epoch,corpusData,corpus_real_sentence_length,corpus_fixed_tree_constructionorder,verbose=True,training=True):
    def run_epoch_train(self,sess,epoch):
    ###run_epoch既要处理train_corpus又要处理evalution_corpus，取决于training是否为true，所以下面的代码中，来回判断traing。
        
    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
    #第三个维度长度尤其要处理成最大的句子长度。因为numpy和我写的placeholder都只能接受这样的输入。
    #返回mean_construcitonError_evalutionCorpus,loss_history
    #loss_history记录这个epoch中每个batch的所有样本的loss的平均值，是一个列表
    #loss_history是训练过程中，W权值矩阵在变化的时候，分别计算的每个batch所有样本的的平均损失。W是变化的。并且当training为true的时候，这个误差包括参数L2正则化误差。
        loss_history = []
        
        
        corpus=self.trainCorpus
        corpus_sentence_length=self.trainCorpus_sentence_length
        corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder
        max_sentence_length_Corpus=self.config.max_sentence_length_train_Corpus
        train_corpus_num=len(self.trainCorpus)
####################################################################################################################################################################################
####################################################################################################################################################################################
        ###由于训练时要强调对齐策略，因此，如果一批样本中有一个样本的长度特别长，那么对齐时，就会补齐太多零，甚至耗尽内存。所以，我们对于过长的样本，则进行单独训练，对于较短的样本，则用在一个batch中进行训练。
        ##所以，我们预先将corpus中的某些大样本抽取出来。单独进行计算。最后再进行整合。但是我们最后整合的时候，还要保证其在原语料库中的顺序。
        #对corpus进行处理，抽取句子长度特别长的语料。经过试验观察发现，句子长度1000以下的都能处理。但是超过1000的，最好单独计算。
#        filter_length=1000;           
        filter_length=self.config.MAX_SENTENCE_LENGTH_for_Bigclonebench
#                filter_length=100;           
        logs.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(filter_length))
        #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
        short_sentence_indexincorpus_list=[]
        long_sentence_indexincorpus_list=[]
        for index,length in enumerate(corpus_sentence_length):
            if length<filter_length:
                short_sentence_indexincorpus_list.append(index)
            else:
                long_sentence_indexincorpus_list.append(index)
        logs.log("训练集的句子{}个".format(train_corpus_num))
        logs.log("较长的句子{}个".format(len(long_sentence_indexincorpus_list)))
        logs.log("较短的句子{}个".format(len(short_sentence_indexincorpus_list)))
        short_sentence_corpus=[corpus[index] for index in short_sentence_indexincorpus_list]
        short_sentence_corpus_length=[corpus_sentence_length[index] for index in short_sentence_indexincorpus_list]
        short_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in short_sentence_indexincorpus_list]
                
        long_sentence_corpus=[corpus[index] for index in long_sentence_indexincorpus_list]
        ####由于单独处理了，用的不是网络，而是另外一个计算过程，因此，就不需要这个长度信息了。
        long_sentence_corpus_length=[corpus_sentence_length[index] for index in long_sentence_indexincorpus_list]
        long_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in long_sentence_indexincorpus_list]
                
        logs.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
        logs.log("先处理较短的句子的语料，批处理开始")
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
        short_sentence_num=len(short_sentence_indexincorpus_list)
        #######读取词向量矩阵
        wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
        wordEmbedding_size=wordEmbedding.shape[0]
        #######读取词向量矩阵
        data_idxs=list(range(short_sentence_num)) 
        i_set_batch_size=self.config.batch_size

        
        batch_index=0
        for i_range in range(0,short_sentence_num,i_set_batch_size):
###################调试用，当处理某个batch出现错误的时候，可以用。
#            if (batch_index!=1503): #实际实验的过程中，这个batch的数据送入网络总是出错，我们单独拿出来进行调试。
#                batch_index=batch_index+1
#                continue
###################调试
#            if(batch_index!=661):#这个批次的总是出错
#                batch_index=batch_index+1
#                continue
            #####调试对较长句子的处理,正确执行时，要注释掉。
#            batch_index=batch_index+1
#            continue
            #####调试对较长句子的处理
            
            
            
            
            real_batch_size = min(i_range+i_set_batch_size,short_sentence_num)-i_range
            batch_idxs=data_idxs[i_range:i_range+real_batch_size]
            batchCorpus=[short_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
            batch_real_sentence_length=[short_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
            batch_max_sentence_length=max(batch_real_sentence_length)
            x=[]
            for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                sentence_len=batch_real_sentence_length[i]
                ones=np.ones_like(sentence)
                words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                L1 = wordEmbedding[:,words_indexed]
                #######
                ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                #######
                buqi=batch_max_sentence_length-sentence_len
                L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                L=np.concatenate((L1,L2),axis=1)
                x.append(L)
            wordEmbedding_batchCorpus=np.array(x)
            batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
            ###
            #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
            ###
            #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
            x=[]
            batchCorpus_fixed_tree_constructionorder=[short_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
            
            for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                L2=np.zeros([3,buqi],np.int32)
                L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                x.append(L)
            batch_fixed_tree_constructionorder=np.array(x)
#####################           
            feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
#                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
            #####用run_options选项可以查看内存崩溃的情况。
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            loss,_=sess.run([self.tensorLoss_fixed_tree,self.train_op],feed_dict=feed,options=run_options)
            loss_history.append(loss)
            
###########################################################################################################################
###########################################################################################################################
#            #验证我的全符号网络搭建正确。
#            loss=sess.run([self.tensorLoss_fixed_tree],feed_dict=feed)
#            batch_constructionError=sess.run([self.batch_constructionError],feed_dict=feed)
#            loss2=0.0
#            for j in range(len(batchCorpus)):
#                loss2=loss2+self.computeloss_withAST(batchCorpus[j],batchCorpus_fixed_tree_constructionorder[j])
#            mean_loss2=loss2/real_batch_size
#            print(loss,batch_constructionError,mean_loss2)
#            ##                sess.run(self.tfPrint,feed_dict=feed)#打印tensorflow网络节点中的信息。
###########################################################################################################################
###########################################################################################################################            
            logs.log('\repoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(epoch,batch_index, real_batch_size,loss))             
            logs.log('\repoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(epoch,len(loss_history), np.mean(loss_history)))           
            batch_index=batch_index+1
            pass            

####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
#        logs.log("再处理较长的句子的语料，每个句子单独处理，开始")
#        logs.log("改变策略，对于较长的句子，我们不再用于训练递归自编码器网络，因为内存很快就崩溃了。一个句子就能搞崩溃")   
#        long_sentence_num=len(long_sentence_indexincorpus_list)
#        data_idxs=list(range(long_sentence_num)) 
#        for i_range,sentence in enumerate(long_sentence_corpus):
#            real_batch_size = 1
#            batch_idxs=data_idxs[i_range:i_range+real_batch_size]
#            batchCorpus=[long_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
#            batch_real_sentence_length=[long_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
##                    batch_real_sentence_length = list(map(len,batchCorpus))
##                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]            
#            batch_max_sentence_length=max(batch_real_sentence_length)
#            
#            x=[]
#            for i,sentence in enumerate(batchCorpus):
##                        sentence_len=len(sentence) #句子长度
#                sentence_len=batch_real_sentence_length[i]
#                ones=np.ones_like(sentence)
#                words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
#                L1 = wordEmbedding[:,words_indexed]
#                #######
#                ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
#                #######
#                buqi=batch_max_sentence_length-sentence_len
#                L2=np.zeros([wordEmbedding_size,buqi],np.float64)
#                L=np.concatenate((L1,L2),axis=1)
#                x.append(L)
#            wordEmbedding_batchCorpus=np.array(x)
#            batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
#            ###
#            #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
#            ###
#            #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
#            x=[]
#            batchCorpus_fixed_tree_constructionorder=[long_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
#            
#            for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
#                sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
##                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
#                buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
#                L2=np.zeros([3,buqi],np.int32)
#                L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
#                x.append(L)
#            batch_fixed_tree_constructionorder=np.array(x)
######################           
#            feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
##                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
#            #####用run_options选项可以查看内存崩溃的情况。
##                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
##                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
##            loss,_=sess.run([self.tensorLoss_fixed_tree,self.train_op],feed_dict=feed)
#            run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#            loss,_=sess.run([self.tensorLoss_fixed_tree,self.train_op],feed_dict=feed,options=run_options)
#            loss_history.append(loss)
#            logs.log('\repoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(epoch,batch_index, real_batch_size,loss))             
#            logs.log('\repoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(epoch,len(loss_history), np.mean(loss_history)))           
#            batch_index=batch_index+1
#            pass
####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
        logs.log("保存模型到temp")
        saver = tf.train.Saver()
        if not os.path.exists("./weights"):
            os.makedirs("./weights")
        saver.save(sess, './weights/%s.temp'%self.config.model_name)
        return loss_history 
    def run_epoch_evaluation(self,sess,epoch):
    ###run_epoch既要处理train_corpus又要处理evalution_corpus，取决于training是否为true，所以下面的代码中，来回判断traing。
        
    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
    #第三个维度长度尤其要处理成最大的句子长度。因为numpy和我写的placeholder都只能接受这样的输入。
    #返回mean_construcitonError_evalutionCorpus,loss_history
    #loss_history记录这个epoch中每个batch的所有样本的loss的平均值，是一个列表
    #loss_history是训练过程中，W权值矩阵在变化的时候，分别计算的每个batch所有样本的的平均损失。W是变化的。并且当training为true的时候，这个误差包括参数L2正则化误差。
        loss_history = []
        
        corpus=self.evalutionCorpus
        corpus_sentence_length=self.evalution_corpus_sentence_length
        corpus_fixed_tree_constructionorder=self.evalution_corpus_fixed_tree_constructionorder
        max_sentence_length_Corpus=self.config.max_sentence_length_evalution_Corpus
        evalution_corpus_num=len(self.evalutionCorpus)
####################################################################################################################################################################################
####################################################################################################################################################################################
        ###由于训练时要强调对齐策略，因此，如果一批样本中有一个样本的长度特别长，那么对齐时，就会补齐太多零，甚至耗尽内存。所以，我们对于过长的样本，则进行单独训练，对于较短的样本，则用在一个batch中进行训练。
        ##所以，我们预先将corpus中的某些大样本抽取出来。单独进行计算。最后再进行整合。但是我们最后整合的时候，还要保证其在原语料库中的顺序。
        #对corpus进行处理，抽取句子长度特别长的语料。经过试验观察发现，句子长度1000以下的都能处理。但是超过1000的，最好单独计算。
        filter_length=1000;           
#                filter_length=100;           
        logs.log('训练过程设置长短的衡量标准是{}，长的单独成batch，短的集合成batch'.format(filter_length))
        #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
        short_sentence_indexincorpus_list=[]
        long_sentence_indexincorpus_list=[]
        for index,length in enumerate(corpus_sentence_length):
            if length<filter_length:
                short_sentence_indexincorpus_list.append(index)
            else:
                long_sentence_indexincorpus_list.append(index)
        logs.log("训练集的句子{}个".format(train_corpus_num))
        logs.log("较长的句子{}个".format(len(long_sentence_indexincorpus_list)))
        logs.log("较短的句子{}个".format(len(short_sentence_indexincorpus_list)))
        short_sentence_corpus=[corpus[index] for index in short_sentence_indexincorpus_list]
        short_sentence_corpus_length=[corpus_sentence_length[index] for index in short_sentence_indexincorpus_list]
        short_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in short_sentence_indexincorpus_list]
                
        long_sentence_corpus=[corpus[index] for index in long_sentence_indexincorpus_list]
        ####由于单独处理了，用的不是网络，而是另外一个计算过程，因此，就不需要这个长度信息了。
        long_sentence_corpus_length=[corpus_sentence_length[index] for index in long_sentence_indexincorpus_list]
        long_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in long_sentence_indexincorpus_list]
                
        logs.log("较短的句子，我们走批处理训练网络。长句子，我们单独计算训练网络")
        logs.log("先处理较短的句子的语料，批处理开始")
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
        short_sentence_num=len(short_sentence_indexincorpus_list)
        #######读取词向量矩阵
        wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
        wordEmbedding_size=wordEmbedding.shape[0]
        #######读取词向量矩阵
        data_idxs=list(range(short_sentence_num)) 
#        i_set_batch_size=self.config.batch_size
        i_set_batch_size=self.config.batch_size_using_model_notTrain #evalution的时候，就不需要用小批量了。
        
        batch_index=0
        for i_range in range(0,short_sentence_num,i_set_batch_size):
###################调试用，当处理某个batch出现错误的时候，可以用。
#            if (batch_index!=1503): #实际实验的过程中，这个batch的数据送入网络总是出错，我们单独拿出来进行调试。
#                batch_index=batch_index+1
#                continue
###################调试    
            real_batch_size = min(i_range+i_set_batch_size,short_sentence_num)-i_range
            batch_idxs=data_idxs[i_range:i_range+real_batch_size]
            batchCorpus=[short_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
            batch_real_sentence_length=[short_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
            batch_max_sentence_length=max(batch_real_sentence_length)
            x=[]
            for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                sentence_len=batch_real_sentence_length[i]
                ones=np.ones_like(sentence)
                words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                L1 = wordEmbedding[:,words_indexed]
                #######
                ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                #######
                buqi=batch_max_sentence_length-sentence_len
                L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                L=np.concatenate((L1,L2),axis=1)
                x.append(L)
            wordEmbedding_batchCorpus=np.array(x)
            batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
            ###
            #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
            ###
            #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
            x=[]
            batchCorpus_fixed_tree_constructionorder=[short_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
            
            for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                L2=np.zeros([3,buqi],np.int32)
                L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                x.append(L)
            batch_fixed_tree_constructionorder=np.array(x)
#####################           
            feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
#                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
            #####用run_options选项可以查看内存崩溃的情况。
#                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
#            loss=sess.run([self.batch_constructionError],feed_dict=feed)#在评估语料库上评估模型时，用重构误差。
            loss=sess.run([self.tensorLoss_fixed_tree],feed_dict=feed)####考虑一下评估模型是仅仅用重构误差还是包含参数正则化损失。
            loss_history.append(loss)
            
###########################################################################################################################
###########################################################################################################################
#            #验证我的全符号网络搭建正确。
#            loss=sess.run([self.tensorLoss_fixed_tree],feed_dict=feed)
#            batch_constructionError=sess.run([self.batch_constructionError],feed_dict=feed)
#            loss2=0.0
#            for j in range(len(batchCorpus)):
#                loss2=loss2+self.computeloss_withAST(batchCorpus[j],batchCorpus_fixed_tree_constructionorder[j])
#            mean_loss2=loss2/real_batch_size
#            print(loss,batch_constructionError,mean_loss2)
#            ##                sess.run(self.tfPrint,feed_dict=feed)#打印tensorflow网络节点中的信息。
###########################################################################################################################
###########################################################################################################################            
            logs.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(epoch,batch_index, real_batch_size,loss))             
            logs.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(epoch,len(loss_history), np.mean(loss_history)))           
            batch_index=batch_index+1
            pass            

####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
        logs.log("再处理较长的句子的语料，每个句子单独处理，开始")   
        long_sentence_num=len(long_sentence_indexincorpus_list)
        data_idxs=list(range(long_sentence_num)) 
        for i_range,sentence in enumerate(long_sentence_corpus):
            real_batch_size = 1
            batch_idxs=data_idxs[i_range:i_range+real_batch_size]
            batchCorpus=[long_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
            batch_real_sentence_length=[long_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]            
            batch_max_sentence_length=max(batch_real_sentence_length)
            
            x=[]
            for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                sentence_len=batch_real_sentence_length[i]
                ones=np.ones_like(sentence)
                words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                L1 = wordEmbedding[:,words_indexed]
                #######
                ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                #######
                buqi=batch_max_sentence_length-sentence_len
                L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                L=np.concatenate((L1,L2),axis=1)
                x.append(L)
            wordEmbedding_batchCorpus=np.array(x)
            batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
            ###
            #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
            ###
            #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
            x=[]
            batchCorpus_fixed_tree_constructionorder=[long_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
            
            for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                L2=np.zeros([3,buqi],np.int32)
                L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                x.append(L)
            batch_fixed_tree_constructionorder=np.array(x)
#####################           
            feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
#                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
            #####用run_options选项可以查看内存崩溃的情况。
#                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
#            loss=sess.run([self.batch_constructionError],feed_dict=feed)#在评估语料库上评估模型时，用重构误差。
            loss=sess.run([self.tensorLoss_fixed_tree],feed_dict=feed) ####考虑一下评估模型是仅仅用重构误差还是包含参数正则化损失。
            loss_history.append(loss)
            logs.log('\r验证过程epoch:{},第{}个batch(), 包括{}个sentence，这个batch的mean_loss = {}'.format(epoch,batch_index, real_batch_size,loss))             
            logs.log('\r验证过程epoch:{},到目前，一共训练{}个batch, 所有batch的mean_loss = {}'.format(epoch,len(loss_history), np.mean(loss_history)))           
            batch_index=batch_index+1
            pass
        return loss_history 
   
    def train(self,restore=False):

######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
#下面的代码一次性将所有的句子都处理成了向量表示，前期处理成train_corpus的时候，就已经耗了很多的内存，如果再这样将每个句子还处理成向量表示，就会更加耗尽内存。
####我们的改进方式是：修改run_epoch函数，然后将里面的batch生成的时候，再执行下面的将句子转变为词向量表示的操作。
###########################################################################
        #处理数据最好放在训练的前面。
        #一、处理数据，将trainCorpus从二维矩阵，行是句子的数目，列是句子的单词树，变成三维矩阵。添加中间维度。中间维度就是词向量信息。
        #比如两个句子6 7 8，我们需要将其变为300*3。然后多个句子。所以，是三个维度。
        #但是numpy有一个特点[[[1]],[[1]]]是三个维度，但是[[[1]],[[1,2]]]就是两个维度。也就是说，其每个维度内部，必须对齐。
        #所以，我们用trainCorpus中最长的句子长度作为第三维度对齐全部矩阵。然后另外使用real_sentence_length保存每个句子的真实长度。
#        print (self.trainCorpus)#self.trainCorpus是一共list。而且每个元素的长度又不相同，因此不能转为numpy数组。
#####################
#        wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
#        wordEmbedding_size=wordEmbedding.shape[0]
#        x=[]
#        for sentence in self.trainCorpus:
#            sentence_len=len(sentence) #句子长度
#            ones=np.ones_like(sentence)
#            words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
#            L1 = wordEmbedding[:,words_indexed]
#            buqi=self.config.max_sentence_length_train_Corpus-sentence_len
#            L2=np.zeros([wordEmbedding_size,buqi],np.float64)
#            L=np.concatenate((L1,L2),axis=1)
#            x.append(L)
#        self.wordEmbedding_train_corpus=np.array(x)
#        self.train_corpus_real_sentence_length=[len(sentence) for sentence in self.trainCorpus]
#        
#        #处理train_corpus的抽象语法树构建顺序的数据，同样需要补齐
#        x=[]
#        for sentence_fixed_tree_constructionorder in self.train_corpus_fixed_tree_constructionorder:
#            sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
#            buqi=(self.config.max_sentence_length_train_Corpus-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
#            L2=np.zeros([3,buqi],np.int32)
#            L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
#            x.append(L)
#        self.train_corpus_fixed_tree_constructionorder_for_tensorflow_network=np.array(x)
#####################
######################################################################################################################################################
######################################################################################################################################################
######################################################################################################################################################
        ##########二、处理评估语料库 我们这个版本中将评估语料库和训练语料库等同，具体原因是为了处理函数句子有的长达1000多个，造成补齐操作耗尽内存。
#        wordEmbedding_size=wordEmbedding.shape[0]
#        y=[]
#        for sentence in self.fullCorpus:
#            sentence_len=len(sentence) #句子长度
#            ones=np.ones_like(sentence)
#            words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
#            L1 = wordEmbedding[:,words_indexed]
#            buqi=self.config.max_sentence_length_full_Corpus-sentence_len
#            L2=np.zeros([wordEmbedding_size,buqi],np.float64)
#            L=np.concatenate((L1,L2),axis=1)
#            y.append(L)
#        self.wordEmbedding_full_corpus=np.array(y)
#        self.full_corpus_real_sentence_length=[len(sentence) for sentence in self.fullCorpus]       
        ##########二、处理评估语料库
#        print(self.trainCorpus[0])#注意单词编号是在文档中是从1开始。但是我们做根据编号去取词向量的时候，要减去1.
#        print(self.wordEmbedding_train_corpus.shape)
#        print(self.wordEmbedding_train_corpus[0].shape)
#        print(self.wordEmbedding_train_corpus[0])
###########################################################################
        #二、创建计算图
        with tf.Graph().as_default():
            self.xiaojie_RvNN_fixed_tree()#构建网络结构图
#将计算图保存下来，用tensorboard查看
#            writer = tf.summary.FileWriter('./graphs/RvNN_grap_new_model2', sess.graph)
#            writer.close()
#        """输出所有可训练的变量名称，也就是神经网络的参数"""
#        with tf.Graph().as_default(), tf.Session(config=config) as sess:
#            self.add_model_vars()
#            saver = tf.train.Saver()
#            saver.restore(sess, './weights/%s.temp'%self.config.model_name)
#            variable_list_name = [c.name for c in tf.trainable_variables()]
#            variable_list = sess.run(variable_list_name)
#            for k,v in zip(variable_list_name,variable_list):
#                print("输出计算图的参数变量信息")
#                print("variable name:",k)
#                print("shape:",v.shape)
#                print(v) 
            init=tf.initialize_all_variables()
            saver = tf.train.Saver()

            #三、开始训练，记录相关训练信息
            complete_loss_history = []
            mean_construcitonError_evalutionCorpus_history=[]
            prev_epoch_loss = float('inf')#float("inf"), float("-inf")表示正负无穷
            best_val_loss = float('inf')
            best_val_epoch = 0 #要区分评估集，训练集和测试集。
            stopped = -1
            epoch=0
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),allow_soft_placement=True)
#            config.gpu_options.allocator_type = 'BFC'
            config.gpu_options.per_process_gpu_memory_fraction = 0.95
            with tf.Session(config=config) as sess:
                sess.run(init)
                start_time=time.time()
                #如果从模型中重新加载
                if restore:saver.restore(sess, './weights/%s'%self.config.model_name)
                #开始训练
                while epoch<self.config.max_epochs:
                    logs.log('epoch %d'%epoch)
#                    loss_history = self.run_epoch(sess,epoch,corpusData=self.wordEmbedding_train_corpus,corpus_real_sentence_length=self.train_corpus_real_sentence_length,corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder_for_tensorflow_network,training=True) #这个loss_history是包括参数正则化误差的，并且每个元素对应一个batch样本集的平均损失。
                    loss_history = self.run_epoch_train(sess,epoch) #这个loss_history是包括参数正则化误差的，并且每个元素对应一个batch样本集的平均损失。
                    complete_loss_history.extend(loss_history)
####################################################################################################
                    ##以前是用训练集来修改学习率
                    #lr annealing
#                    epoch_loss = np.mean(loss_history)
#                    ###修改学习率
#                    if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:#当损失值一直不怎么变化的时候，这个时候可能就已经出现震荡和波动了，这个时候就应该减少学习率。
#                        self.config.lr/=self.config.anneal_by
#                        logs.log ('annealed lr to %f'%self.config.lr)
#                    prev_epoch_loss = epoch_loss #
                    #save if model has improved on val
                    #现在在评估语料库上评估模型
####################################################################################################
                    #loss_history_evalution= self.run_epoch(sess,corpusData=self.wordEmbedding_full_corpus,corpus_real_sentence_length=self.full_corpus_real_sentence_length,training=False)#这个loss_history是不包括参数正则化误差的，因为是评估。我们只取重构误差。并且每个元素对应一个batch样本集的平均损失。评估时用批只是为了计算更快。
                    #loss_history_evalution= self.run_epoch(sess,epoch,corpusData=self.wordEmbedding_train_corpus,corpus_real_sentence_length=self.train_corpus_real_sentence_length,corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder_for_tensorflow_network,training=False)#这个loss_history是不包括参数正则化误差的，因为是评估。我们只取重构误差。并且每个元素对应一个batch样本集的平均损失。评估时用批只是为了计算更快。
                    loss_history_evalution= self.run_epoch_evaluation(sess,epoch)#这个loss_history是不包括参数正则化误差的，因为是评估。我们只取重构误差。并且每个元素对应一个batch样本集的平均损失。评估时用批只是为了计算更快。
                    mean_construcitonError_evalutionCorpus=np.mean(loss_history_evalution) ##将所有batch的损失的均值作为整个评估集上的误差。
                    mean_construcitonError_evalutionCorpus_history.append(mean_construcitonError_evalutionCorpus)
                    logs.log("time per epoch is {} s".format(time.time()-start_time))
####################################################################################################
                    ##现在要用验证集来修改学习率
                    #lr annealing
                    epoch_loss = mean_construcitonError_evalutionCorpus
                    ###修改学习率
                    if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:#当损失值一直不怎么变化的时候，这个时候可能就已经出现震荡和波动了，这个时候就应该减少学习率。
                        self.config.lr/=self.config.anneal_by
                        logs.log ('annealed lr to %f'%self.config.lr)
                    prev_epoch_loss = epoch_loss #
####################################################################################################                    
                    ##保存最优损失的模型
                    if mean_construcitonError_evalutionCorpus < best_val_loss:
                        #保存一个模型不仅仅就这一个文件 
                        #shutil.copyfile('./weights/%s.temp'%self.config.model_name, './weights/%s'%self.config.model_name)
                         
                        shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%self.config.model_name, './weights/%s.data-00000-of-00001'%self.config.model_name)
                        shutil.copyfile('./weights/%s.temp.index'%self.config.model_name, './weights/%s.index'%self.config.model_name)
                        shutil.copyfile('./weights/%s.temp.meta'%self.config.model_name, './weights/%s.meta'%self.config.model_name) 
                         
                        best_val_loss = mean_construcitonError_evalutionCorpus
                        best_val_epoch = epoch
                    elif epoch - best_val_epoch >= self.config.early_stopping:#又经过early_stopping=2轮训练，发现没有什么改进的时候，就会提出终止训练。
                    # if model has not imprvoved for a while stop
                        stopped=epoch
                        break
                        #break #break应该打出来。
                    epoch+=1
                    start_time=time.time()
                    pass
                if(epoch<(self.config.max_epochs-1)):#没有训练全部轮。
                    logs.log ('预定训练{}个epoch,一共训练{}个epoch，在评估集上最优的是第{}个epoch(从0开始计数),最优评估loss是{}'.format(self.config.max_epochs,stopped+1,best_val_epoch,best_val_loss))
                elif (epoch==(self.config.max_epochs-1)):#这里面就是stop的时候，是最后一轮执行完毕，但是效果没有提升，恰好相较于最好效果的所在轮数为self.config.early_stopping，也会终止。即此时也是达成了全部轮数。
                    logs.log ('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(self.config.max_epochs,best_val_epoch,best_val_loss))
                    #        print ('\n\nstopped at %d\n'%stopped)
                else: #这种epoch就直接等于self.config.max_epochs，说明全部训练epoch达成。
                    logs.log ('预定训练{}个epoch,全部轮数达成，在评估集上最优的是第{}个epoch,最优评估loss是{}'.format(self.config.max_epochs,best_val_epoch,best_val_loss))
            return {
                'complete_loss_history': complete_loss_history,#返回的是全部Epoch中对每个句子进行训练的过程中计算的loss，并且W权重是变化的。还包括权重的正则化损失，不仅仅是重构误差。每个元素是一个batch样本集的损失平均值。
                'evalution_loss_history': mean_construcitonError_evalutionCorpus_history,#返回的是每次epoch训练结束后，在全部evalution集上进行计算得到的平均损失（这里evalution集就是所有的sentences）。只包括重构误差。每个元素是一个batch样本集的损失平均值。
            }
            
    def using_model_for_BigCloneBench_experimentID_1(self):
#        self.lines_for_bigcloneBench  ####这个就是一个列表。假设为[100,199.....]则，BigCloneBench中的某个ID编号对应语料库的编号为100，并且在self.bigCloneBench_Corpus的第0个位置
#        self.bigCloneBench_Corpus
#        self.bigCloneBench_Corpus_fixed_tree_constructionorder
#        self.bigCloneBench_Corpus_sentence_length
#        self.bigCloneBench_Corpus_max_sentence_length
        #所以，我们首先计算self.bigCloneBench_Corpus的每个语料的向量表示。
        #然后对于BigCloneBench中的任何一个函数ID，则首先找到其在语料库中的编号line
        #接着，我们将lines_for_bigcloneBench处理成为字典，即100:0,199:1这种形式。那么我们就能根据line，找到其在bigCloneBench_Corpus中的位置。
        #最后，给定BigCloneBench中的任何一个函数ID，我们都能知道其向量表示了。
        
        ##################
        #######第一步：读取BigCloneBench的所有ID编号
        ##################
        logs.log("------------------------------\n读取BigCloneBench的所有ID编号")
        all_idMapline_pkl = './SplitDataSet/data/all_idMapline_XIAOJIE.pkl'
        ids_in_BigCloneBench=[]
        with open(all_idMapline_pkl, 'rb') as f:
            id_line_dict = pickle.load(f)
            for id_num in id_line_dict.keys():
                line = id_line_dict[id_num]
                if(line==-1):
                    continue ##对于出错的函数，我们不给于处理
                ids_in_BigCloneBench.append(id_num) 
        ##################
        #######第二步：计算self.bigCloneBench_Corpus中每个语料的向量表示
        ##################
        random_str='ZHIXIN_'+''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
        path_pkl_root='./vector/diffweight/'+random_str+'_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        path_pkl_root='./vector/'+random_str+'_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_root):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_root)  
        else:
            print('no such file:%s'%path_pkl_root) # 则返回文件不存在
        

        path_pkl_mean='./vector/diffweight/'+random_str+'_using_PVT_BigCloneBenchFunction_ID_Map_Vector_mean.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        path_pkl_mean='./vector/'+random_str+'_using_PVT_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_mean):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_mean)  
        else:
            print('no such file:%s'%path_pkl_mean) # 则返回文件不存在


        
        path_pkl_weighted_TF_IDF='./vector/diffweight/'+random_str+'_using_PVT_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        path_pkl_mean='./vector/'+random_str+'_using_PVT_BigCloneBenchFunction_ID_Map_Vector_weighted.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_mean):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_mean)  
        else:
            print('no such file:%s'%path_pkl_mean) # 则返回文件不存在

        
        
        
        
        path_pkl_meanAndRootMost='./vector/diffweight/'+random_str+'_using_PVT_BigCloneBenchFunctionID_Map_Vector_meanAndRootMost.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        path_pkl_meanAndRootMost='./vector/'+random_str+'_using_PVT_BigCloneBenchFunctionID_Map_Vector_meanAndRootMost.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_root):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_root)  
        else:
            print('no such file:%s'%path_pkl_root) # 则返回文件不存在
            
        
        
        path_pkl_weightedAndCengci='./vector/diffweight/'+random_str+'_using_PVT_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        path_pkl_weightedAndCengci='./vector/'+random_str+'_using_PVT_BigCloneBenchFunctionID_Map_Vector_weightedAndCengci.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_root):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_root)  
        else:
            print('no such file:%s'%path_pkl_root) # 则返回文件不存在
        ##################################################################################################
        corpus=self.bigCloneBench_Corpus
        corpus_fixed_tree_constructionorder=self.bigCloneBench_Corpus_fixed_tree_constructionorder
        corpus_sentence_length=self.bigCloneBench_Corpus_sentence_length
        corpus_fixed_tree_parentType_weight=self.bigCloneBench_Corpus_fixed_tree_parentType_weight
        
        
        
#        corpus_sentence_nodes_vectors={}
        corpus_sentence_vector_root={}
        corpus_sentence_vector_mean={}
        corpus_sentence_vector_weighted_TF_IDF={}
        corpus_sentence_vector_meanAndRootMost={}
        corpus_sentence_vector_weightedAndCengci={}
        ###为避免内存过载，释放一些内存
        del(self.trainCorpus)
        del(self.trainCorpus_sentence_length)
        del(self.train_corpus_fixed_tree_constructionorder)
        del(self.vocabulary)
        ##################################################################################################
        
        with tf.Graph().as_default():
#            self.xiaojie_RvNN_fixed_tree()#构建网络结构图 
            self.xiaojie_RvNN_fixed_tree_for_usingmodel()  ##这个模型是用来计算向量树的。我们这里只需要计算一个根节点的向量，所以，要将返回结果进行处理，避免开销过大。
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            config.gpu_options.allocator_type = 'BFC'
            with tf.Session(config=config) as sess:
############################################################################################
#                直接用随机化权重，计算一下，看看效果。
#                init=tf.initialize_all_variables()
#                sess.run(init)
#                
#                #不用训练后模型
#                if random_flag==True:
#                    init=tf.initialize_all_variables()
#                    sess.run(init)
#                else:
#                    weights_path='./weights/epoch_%s_%s'%(epoch,self.config.model_name)
#                    saver.restore(sess, weights_path)#导入权重
                            
############################################################################################
############################################################################################
                #下面这个过程计算时间耗费太久。因为无法利用GPU。并没有用tensorflow的网络进行计算。
#                step=0
#                while step < len(corpus):
#                    print('step{}/{}'.format(step,len(self.bigCloneBench_Corpus)))
#                    sentence = corpus[step]#取出的是单词的索引编号，而且下标是从1开始计数的。
#                    treeConstructionOrders=corpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                    (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                    features[:,step,0]=root_vector[:,0]
#                    sum_vector=np.zeros((self.config.embed_size,1),dtype=np.float64)
#                    num_nodes=0
#                    for (_,node_vector) in node_vectors.items():
#                        sum_vector=node_vector+sum_vector
#                        num_nodes+=1
#                        pass
#                    mean_node_vector=sum_vector/num_nodes
#                    features[:,step,1]=mean_node_vector[:,0]
#                    step+=1
#                    pass
#                pass
                
############################################################################################
                filter_length=300;           
#                filter_length=100;           
                logs.log('设置长短的衡量标准是{}'.format(filter_length))
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                short_sentence_indexincorpus_list=[]
                long_sentence_indexincorpus_list=[]
                for index,length in enumerate(corpus_sentence_length):
                    if length<filter_length:
                        short_sentence_indexincorpus_list.append(index)
                    else:
                        long_sentence_indexincorpus_list.append(index)
                
                logs.log("较长的句子{}个".format(len(long_sentence_indexincorpus_list)))
                logs.log("较短的句子{}个".format(len(short_sentence_indexincorpus_list)))
                short_sentence_corpus=[corpus[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_length=[corpus_sentence_length[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_fixed_tree_parentType_weight=[corpus_fixed_tree_parentType_weight[index] for index in short_sentence_indexincorpus_list]
                long_sentence_corpus=[corpus[index] for index in long_sentence_indexincorpus_list]
                ####由于单独处理了，用的不是网络，而是另外一个计算过程，因此，就不需要这个长度信息了。
                long_sentence_corpus_length=[corpus_sentence_length[index] for index in long_sentence_indexincorpus_list]
                long_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in long_sentence_indexincorpus_list]
                long_sentence_corpus_fixed_tree_parentType_weight=[corpus_fixed_tree_parentType_weight[index] for index in long_sentence_indexincorpus_list]
                
                logs.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
                logs.log("先处理较短的句子的语料，批处理开始")
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
                sentence_num=len(short_sentence_indexincorpus_list)
                #######读取词向量矩阵
                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
                ####时间2019年9月9日20:45:15，我们添加代码对词向量全部进行标准化。具体表现在对wordEmbedding矩阵的每一列的内容，都标准化为模为1。
                ###对整个矩阵的每一列变为模为1，耗时会比较长。我们在下面读取这个词向量的地方，在那里改为模为1.就是直接对向量进行模为1计算。
                
                
                ####
#                del(self.We)
                wordEmbedding_size=wordEmbedding.shape[0]
                #######读取词向量矩阵
                data_idxs=list(range(sentence_num)) 
                i_set_batch_size=self.config.batch_size_using_model_notTrain
                
                #保存整个语料库的所有句子的最终向量表示
                xiaojie_batch_num=(sentence_num-1)/i_set_batch_size #从0开始编号的batch数目
                batch_index=0
                
####################################################################################测试 取出前10个
                
#                #下面这个过程计算时间耗费太久。因为无法利用GPU。并没有用tensorflow的网络进行计算。
#                step=0
#                while step < 10:
#                    print('step{}/{}'.format(step,len(short_sentence_corpus)))
#                    sentence = short_sentence_corpus[step]#取出的是单词的索引编号，而且下标是从1开始计数的。
#                    treeConstructionOrders=short_sentence_corpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                    (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                    step+=1
#                    pass
#                pass
                
####################################################################################测试 取出前10个
            
                for i_range in range(0,sentence_num,i_set_batch_size):
                    #测试用#测试用#测试用#测试用#测试用
#                    batch_index=batch_index+1
#                    continue #测试用
                    #测试用#测试用#测试用#测试用#测试用
#                    if(batch_index==2):
#                        break
                    
                    logs.log("batch_index:{}/{}".format(batch_index,xiaojie_batch_num))
                    
                    real_batch_size = min(i_range+i_set_batch_size,sentence_num)-i_range
                    batch_idxs=data_idxs[i_range:i_range+real_batch_size]
                    batchCorpus=[short_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[short_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    del(wordEmbedding_batchCorpus)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[short_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
                    
                                        #####权重信息：这一次用所包含的叶子节点的数目
                    batch_childweight_parentweight=[]
                    for i in range(real_batch_size):
                        sentence_constructionorder=batchCorpus_fixed_tree_constructionorder[i]
                        sentence_len=batch_real_sentence_length[i]
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1 #比句子长度小1
                        nodeslength=sentence_len+sentence_fixed_tree_constructionorder_len
                        cengci={}#键值是节点在句子中的编号。然后映射的值就是其所在的层次
                        sum=0
                        for i in range(sentence_fixed_tree_constructionorder_len+1):#先将叶子节点标注出来
                            key=i+1#节点从1开始编号
                            cengci[key]=1
                            sum+=1
                        for i in range(sentence_fixed_tree_constructionorder_len):
                            left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
                            right_index=sentence_constructionorder[1,i]
                            parent_index=sentence_constructionorder[2,i]
                            cegnci_left=cengci[left_index]
                            cegnci_right=cengci[right_index]
                            cengci[parent_index]=cegnci_left+cegnci_right #把左右子树的叶子节点数目加起来即可。
                            sum+=cengci[parent_index]
                        ###现在将cengci中的系数与进行归一化。
        #                for i in range(1,nodeslength+1):#节点编号从1开始。到nodeslength结束
        #                    cengci[i]=cengci[i]/(sum) #cengci中的节点编号仍然是从1开始。
                        ###结构参照sentence_fixed_tree_constructionorder
                        sentence_childweight_parentweight=[]
                        for i in range(sentence_fixed_tree_constructionorder_len):
                            left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
                            right_index=sentence_constructionorder[1,i]
                            parent_index=sentence_constructionorder[2,i]
                            leftchild_weight=cengci[left_index]
                            rightchild_weight=cengci[right_index]
                            parentchild_weight=cengci[parent_index]
                            sentence_childweight_parentweight.append([leftchild_weight,rightchild_weight,parentchild_weight])
                        b=np.array(sentence_childweight_parentweight)
                        b=b.transpose()
                        batch_childweight_parentweight.append(b) #列表转变成了numpy矩阵
                        pass
                    #补齐操作
                    x=[]
                    for i,sentence_childweight_parentweight in enumerate(batch_childweight_parentweight):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
        #                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.float64)
                        L=np.concatenate((sentence_childweight_parentweight,L2),axis=1)
                        x.append(L)
                    batch_childweight_parentweight=np.array(x)
                    
                    ############处理权重信息
                                ####权重信息
                    x=[]
                    batchCorpus_fixed_tree_parentType_weight=[short_sentence_corpus_fixed_tree_parentType_weight[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_parentType_weight in enumerate(batchCorpus_fixed_tree_parentType_weight):
                        #########################
                        ###我们要计算非叶子节点在整棵树中的权重信息。读入的tf-idf值要进一步转换
                        sumsum=np.sum(sentence_fixed_tree_parentType_weight)
                        sentence_fixed_tree_parentType_weight=sentence_fixed_tree_parentType_weight/sumsum
        #                print(np.sum(sentence_fixed_tree_parentType_weight)) #看看是不是1
                        #########################
                        
                        sentence_fixed_tree_parentType_weight_len=batch_real_sentence_length[i]-1
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_parentType_weight_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_parentType_weight,L2)) #一维信息直接连接，不用axis
                        x.append(L)
                    batch_fixed_tree_parentType_weight=np.array(x)                    
                    
                    
                    
                    
                    
        #####################
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder,self.batch_childparentweight:batch_childweight_parentweight,self.batch_sentence_parentTypes_weight:batch_fixed_tree_parentType_weight}
#                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
        #                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
                    #####用run_options选项可以查看内存崩溃的情况。
#                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
                    if batch_index==0:
                        print(loss)
#                    print('loss:')
#                    print(loss)
                    ###这个batch_sentence_vectors中保存的是三维矩阵。第一维表示样本在批中的编号，第二维度的长度是词向量的长度，第三维是按照batch中最长节点数目的长度来的（补齐数据）
                    ###所以，我们要取出这个batch_sentence_vectors每个样本的向量树的话，就需要按照batch_real_sentence_length从补齐数据中再取回来。
                    
                    #####测试上面的过程是否正确。
                    #####紧接着，我们不用tensorflow网络的批计算，而是用一个句子一个句子计算的方式进行计算。进行测试
#                    step=0
#                    for sentence in batchCorpus:
#                        treeConstructionOrders=batchCorpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                        (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                        step=step+1;
                        ###接下来，比较root_vector在每一次循环中是否同前面的batch_setence_vectors中对应位置的向量相同即可判定，那个xiaojie_RvNN_fixed_tree_for_usingmodel函数中的add_loss_and_batch...有没有写正确。
                    ####将批句子的最终表示矩阵添加到整个预料的句子的最终表示矩阵
                    
                    
                    

                    
                    
                    in_batch_number=0
                    for index in batch_idxs:
                        sentence_vectors_buqi=batch_sentence_vectors[in_batch_number,:,:] #取出补齐后的数据
                        #取出权重系数
                        sentence_fixed_tree_parentType_weight=batch_fixed_tree_parentType_weight[in_batch_number]
                        #取出权重系数
                        sentence_length=batch_real_sentence_length[in_batch_number]
                        nodeslength=2*sentence_length-1
                        sentence_nodes_vectors=sentence_vectors_buqi[0:self.config.embed_size,0:nodeslength] #取出补齐前的数据
                        sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                        
                        
                        sentence_constructionorder=batch_fixed_tree_constructionorder[in_batch_number,:,:]
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[in_batch_number]-1#构建顺序的元祖数目
                        #########
                        #########
                        #########
                        in_batch_number=in_batch_number+1
                        #########
                        #########
                        #########
                        ###将整个句子的所有向量保存下来
                        
                        sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                        sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
            ####################用根节点的向量
                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
#                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
                        #我们可以保存列表，也可以保存一个单独的向量。
                        #我们先仅仅保存根节点的向量
                        corpus_sentence_vector_root[corpus_index]=sentence_nodes_vectors_list[nodeslength-1]
            ####################用根节点的向量
            ####################用所有节点的平均向量
#                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
##                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
#                        #我们可以保存列表，也可以保存一个单独的向量。
#                        #我们先仅仅保存根节点的向量
#                        mean_vector=np.mean(sentence_nodes_vectors,axis=0)
#                        corpus_sentence_vector_mean[corpus_index]=mean_vector
#            ####################用所有节点的平均向量
#                        ##########q用加权平均
#                        corpus_index=short_sentence_indexincorpus_list[index]
#                        vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
#                        for i in range(sentence_length,nodeslength):
#                            vector=sentence_nodes_vectors_list[i]
#                            j=i-sentence_length
#                            xishu=sentence_fixed_tree_parentType_weight[j]
#                            vector=np.multiply(vector,xishu)
#                            vector_weighted=np.add(vector_weighted,vector)
#                        corpus_sentence_vector_weighted_TF_IDF[corpus_index]=vector_weighted
#                        
##                    ##############用加权，但是更能突出根节点
#                        vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
##                        for i in range(sentence_length,nodeslength-1):
#                        for i in range(sentence_length,nodeslength):
#                            vector=sentence_nodes_vectors_list[i]
#                            j=i-sentence_length
#                            xishu=sentence_fixed_tree_parentType_weight[j]
#                            vector=np.multiply(vector,xishu)
#                            vector_weighted=np.add(vector_weighted,vector)
#                        vector_root=sentence_nodes_vectors_list[nodeslength-1]
#                        ##凸显出根节点的重要性，权重不要用在根节点。因为根节点methodDeclaration在所有的函数中均存在，那么其重要性就会变的很低。
#                        vector_weighted=np.add(np.multiply(vector_weighted,0.5),np.multiply(vector_root,0.5))
#                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
#                        corpus_sentence_vector_meanAndRootMost[corpus_index]=vector_weighted
#                    
                    
                    
                    
                    
#                    ######用加权，并且乘以每个非叶子节点的层次
#                        cengci={}#键值是节点在句子中的编号。然后映射的值就是其所在的层次
#                        
#                        for i in range(sentence_fixed_tree_constructionorder_len+1):#先将叶子节点标注出来
#                            key=i+1#节点从1开始编号
#                            cengci[key]=0
#                        for i in range(sentence_fixed_tree_constructionorder_len):
    #                            left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
#                            right_index=sentence_constructionorder[1,i]
#                            parent_index=sentence_constructionorder[2,i]
#                            cegnci_left=cengci[left_index]
#                            cegnci_right=cengci[right_index]
#                            if (cegnci_left>cegnci_right):
#                                cengci[parent_index]=cegnci_left+1
#                            else:
#                                cengci[parent_index]=cegnci_right+1
#                                                #改为用叶子节点的数目
#                        cengci={}#键值是节点在句子中的编号。然后映射的值就是其所在的层次
#                        for i in range(sentence_fixed_tree_constructionorder_len+1):#先将叶子节点标注出来
#                            key=i+1#节点从1开始编号
#                            cengci[key]=1
#                        sum=0
#                        for i in range(sentence_fixed_tree_constructionorder_len):
#                            left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
#                            right_index=sentence_constructionorder[1,i]
#                            parent_index=sentence_constructionorder[2,i]
#                            cegnci_left=cengci[left_index]
#                            cegnci_right=cengci[right_index]
#                            cengci[parent_index]=cegnci_left+cegnci_right #把左右子树的叶子节点数目加起来即可。
#                            sum+=cengci[parent_index]
#                        ###现在将cengci中的系数与进行归一化。
#                        for i in range(sentence_length+1,nodeslength+1):#节点编号从1开始。
#                            cengci[i]=cengci[i]/(sum) 
#                            
#                            
#                        vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
#                        for i in range(sentence_length,nodeslength):
#    #                    for i in range(sentence_length,nodeslength-1):
#                            vector=sentence_nodes_vectors_list[i]
#                            j=i-sentence_length
#                            xishu=sentence_fixed_tree_parentType_weight[j]
#                            ####
#                            parent_index=sentence_constructionorder[2,j]#取出父亲节点的编号
#                            ####
#                            xishu=xishu*cengci[parent_index] #乘以所在的层次
##                            xishu=cengci[parent_index]
#                            ###我们也可以不乘试试
#                            vector=np.multiply(vector,xishu)
#                            vector_weighted=np.add(vector_weighted,vector)
#                        corpus_index=short_sentence_indexincorpus_list[index]
#                        
#                        ####将向量归一化
#                        vector_weighted=vector_weighted/(np.linalg.norm(vector_weighted))
#                        
#                        corpus_sentence_vector_weightedAndCengci[corpus_index]=vector_weighted 
#                    
                    
                    
                    
                    ######结束循环，我们将batch_index加上1
#                    del(batch_sentence_vectors)
#                    del(sentence_vectors_buqi)
#                    del(sentence_nodes_vectors)
#                    del(sentence_nodes_vectors_list)
#                    del(batch_fixed_tree_constructionorder)
#                    del(batch_data_numpy)
#                    del(batchCorpus)
#                    del(batch_real_sentence_length)
#                    del(batch_idxs)
                    batch_index=batch_index+1
                    ######结束循环，我们将batch_index加上1
####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
                logs.log("再处理较长的句子的语料，每个句子单独处理，开始")   
                long_sentence_num=len(long_sentence_indexincorpus_list)
                data_idxs=list(range(long_sentence_num)) 
                for i_range,sentence in enumerate(long_sentence_corpus):
                    
#            ################调试用
#                    if(i_range==2):
#                        break
#            ################
                    
                    
                    logs.log("long_setence_index:{}/{}".format(i_range,long_sentence_num))
                    real_batch_size = 1
                    batch_idxs=data_idxs[i_range:i_range+real_batch_size]
                    batchCorpus=[long_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[long_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
        #                    batch_real_sentence_length = list(map(len,batchCorpus))
        #                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]            
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
        #                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[long_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
        #                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
                    
                    
                                ###权重信息：这是用的TF-IDF模型

                    
                    
                                #####权重信息：这一次用所包含的叶子节点的数目
                    batch_childweight_parentweight=[]
                    for i in range(real_batch_size):
                        sentence_constructionorder=batchCorpus_fixed_tree_constructionorder[i]
                        sentence_len=batch_real_sentence_length[i]
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1 #比句子长度小1
                        nodeslength=sentence_len+sentence_fixed_tree_constructionorder_len
                        cengci={}#键值是节点在句子中的编号。然后映射的值就是其所在的层次
                        sum=0
                        for i in range(sentence_fixed_tree_constructionorder_len+1):#先将叶子节点标注出来
                            key=i+1#节点从1开始编号
                            cengci[key]=1
                            sum+=1
                        for i in range(sentence_fixed_tree_constructionorder_len):
                            left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
                            right_index=sentence_constructionorder[1,i]
                            parent_index=sentence_constructionorder[2,i]
                            cegnci_left=cengci[left_index]
                            cegnci_right=cengci[right_index]
                            cengci[parent_index]=cegnci_left+cegnci_right #把左右子树的叶子节点数目加起来即可。
                            sum+=cengci[parent_index]
                        ###现在将cengci中的系数与进行归一化。
        #                for i in range(1,nodeslength+1):#节点编号从1开始。到nodeslength结束
        #                    cengci[i]=cengci[i]/(sum) #cengci中的节点编号仍然是从1开始。
                        ###结构参照sentence_fixed_tree_constructionorder
                        sentence_childweight_parentweight=[]
                        for i in range(sentence_fixed_tree_constructionorder_len):
                            left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
                            right_index=sentence_constructionorder[1,i]
                            parent_index=sentence_constructionorder[2,i]
                            leftchild_weight=cengci[left_index]
                            rightchild_weight=cengci[right_index]
                            parentchild_weight=cengci[parent_index]
                            sentence_childweight_parentweight.append([leftchild_weight,rightchild_weight,parentchild_weight])
                        b=np.array(sentence_childweight_parentweight)
                        b=b.transpose()
                        batch_childweight_parentweight.append(b) #列表转变成了numpy矩阵
                        pass
                    #补齐操作
                    x=[]
                    for i,sentence_childweight_parentweight in enumerate(batch_childweight_parentweight):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
        #                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.float64)
                        L=np.concatenate((sentence_childweight_parentweight,L2),axis=1)
                        x.append(L)
                    batch_childweight_parentweight=np.array(x)
                    
                    
                    x=[]
                    batchCorpus_fixed_tree_parentType_weight=[long_sentence_corpus_fixed_tree_parentType_weight[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_parentType_weight in enumerate(batchCorpus_fixed_tree_parentType_weight):
                        #########################
                        ###我们要计算非叶子节点在整棵树中的权重信息。读入的tf-idf值要进一步转换
                        sumsum=np.sum(sentence_fixed_tree_parentType_weight)
                        sentence_fixed_tree_parentType_weight=sentence_fixed_tree_parentType_weight/sumsum
        #                print(np.sum(sentence_fixed_tree_parentType_weight)) #看看是不是1
                        #########################
                        
                        sentence_fixed_tree_parentType_weight_len=batch_real_sentence_length[i]-1
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_parentType_weight_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_parentType_weight,L2)) #一维信息直接连接，不用axis
                        x.append(L)
                    batch_fixed_tree_parentType_weight=np.array(x)
                    
        ##################### 
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder,self.batch_childparentweight:batch_childweight_parentweight,self.batch_sentence_parentTypes_weight:batch_fixed_tree_parentType_weight}
#                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder,self.batch_childparentweight:batch_childweight_parentweight,self.batch_sentence_parentTypes_weight:batch_fixed_tree_parentType_weight}          
#                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
                    
                    

                    
                    
                    
                    
                    sentence_nodes_vectors=batch_sentence_vectors[0,:,:] #batch中只有一个样本。
                    #取出权重系数
                    sentence_fixed_tree_parentType_weight=batch_fixed_tree_parentType_weight[0]
                    #取出权重系数
                    sentence_length=batch_real_sentence_length[0]
                    nodeslength=2*sentence_length-1
                    sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                    ###将整个句子的所有向量保存下来
                    sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                    sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
         ####################用根节点的向量
                    corpus_index=long_sentence_indexincorpus_list[i_range] #找出其在corpus中的序号
#                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
#                        我们可以保存列表，也可以保存一个单独的向量。
#                        我们先仅仅保存根节点的向量
                    corpus_sentence_vector_root[corpus_index]=sentence_nodes_vectors_list[nodeslength-1]
        ####################用根节点的向量
                    ###################用所有节点的平均向量
#                    corpus_index=long_sentence_indexincorpus_list[i_range] #找出其在corpus中的序号                    
#                    mean_vector=np.mean(sentence_nodes_vectors,axis=0)
#                    corpus_sentence_vector_mean[corpus_index]=mean_vector  
                    
                    ####################用所有节点的平均向量
##                     ###用加权
#                    corpus_index=long_sentence_indexincorpus_list[i_range]
#                    vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
#                    for i in range(sentence_length,nodeslength):
#                        vector=sentence_nodes_vectors_list[i]
#                        j=i-sentence_length
#                        xishu=sentence_fixed_tree_parentType_weight[j]
#                        vector=np.multiply(vector,xishu)
#                        vector_weighted=np.add(vector_weighted,vector)
#                    corpus_sentence_vector_weighted_TF_IDF[corpus_index]=vector_weighted
                    
                    ######用加权，但是更加突出根节点
#                    
#                    vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
##                    for i in range(sentence_length,nodeslength-1):
#                    for i in range(sentence_length,nodeslength):
#                        vector=sentence_nodes_vectors_list[i]
#                        j=i-sentence_length
#                        xishu=sentence_fixed_tree_parentType_weight[j]
#                        vector=np.multiply(vector,xishu)
#                        vector_weighted=np.add(vector_weighted,vector)
#                    vector_root=sentence_nodes_vectors_list[nodeslength-1]
#                    ##凸显出根节点的重要性，权重不要用在根节点。因为根节点methodDeclaration在所有的函数中均存在，那么其重要性就会变的很低。
#                    vector_weighted=np.add(np.multiply(vector_weighted,0.5),np.multiply(vector_root,0.5))
#                    corpus_index=long_sentence_indexincorpus_list[i_range]
#                    corpus_sentence_vector_meanAndRootMost[corpus_index]=vector_weighted
    
                    #####用加权的同时，考虑所有非叶子节点所在的层次。
#                    cengci={}#键值是节点在句子中的编号。然后映射的值就是其所在的层次
#                    sentence_constructionorder=batch_fixed_tree_constructionorder[0,:,:]#只有一个样本
#                    sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[0]-1#构建顺序的元祖数目
#                    for i in range(sentence_fixed_tree_constructionorder_len+1):#先将叶子节点标注出来
#                        key=i+1#节点从1开始编号
#                        cengci[key]=0
#                    for i in range(sentence_fixed_tree_constructionorder_len):
#                        left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
#                        right_index=sentence_constructionorder[1,i]
#                        parent_index=sentence_constructionorder[2,i]
#                        cegnci_left=cengci[left_index]
#                        cegnci_right=cengci[right_index]
#                        if (cegnci_left>cegnci_right):
#                            cengci[parent_index]=cegnci_left+1
#                        else:
#                            cengci[parent_index]=cegnci_right+1
                    #改为用叶子节点的数目
#                    cengci={}#键值是节点在句子中的编号。然后映射的值就是其所在的层次
#                    sentence_constructionorder=batch_fixed_tree_constructionorder[0,:,:]#只有一个样本
#                    sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[0]-1#构建顺序的元祖数目
#                    for i in range(sentence_fixed_tree_constructionorder_len+1):#先将叶子节点标注出来
#                        key=i+1#节点从1开始编号
#                        cengci[key]=1
#                    sum=0
#                    for i in range(sentence_fixed_tree_constructionorder_len):
#                        left_index=sentence_constructionorder[0,i] #节点编号都是从1开始
#                        right_index=sentence_constructionorder[1,i]
#                        parent_index=sentence_constructionorder[2,i]
#                        cegnci_left=cengci[left_index]
#                        cegnci_right=cengci[right_index]
#                        cengci[parent_index]=cegnci_left+cegnci_right #把左右子树的叶子节点数目加起来即可。
#                        sum+=cengci[parent_index]
#                    ###现在将cengci中的系数与进行归一化。
#                    for i in range(sentence_length+1,nodeslength+1):#节点编号从1开始。
#                        cengci[i]=cengci[i]/(sum) 
#
#                        
#                    vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
#                    for i in range(sentence_length,nodeslength):
##                    for i in range(sentence_length,nodeslength-1):
#                        vector=sentence_nodes_vectors_list[i]
#                        j=i-sentence_length
#                        xishu=sentence_fixed_tree_parentType_weight[j]
#                        ####
#                        parent_index=sentence_constructionorder[2,j]#取出父亲节点的编号
#                            ####
#                        xishu=xishu*cengci[parent_index] #乘以所在的层次
##                        xishu=cengci[parent_index]
#                            ###我们也可以不乘试试
#                        vector=np.multiply(vector,xishu)
#                        vector_weighted=np.add(vector_weighted,vector)
#                    corpus_index=long_sentence_indexincorpus_list[i_range]
#                    
#                    ####将向量归一化
#                    vector_weighted=vector_weighted/(np.linalg.norm(vector_weighted))
#                    
#                    corpus_sentence_vector_weightedAndCengci[corpus_index]=vector_weighted        
#                    
#                    del(batch_sentence_vectors)
#                    del(sentence_nodes_vectors)
#                    del(sentence_nodes_vectors_list)
#                    del(batch_fixed_tree_constructionorder)
#                    del(batch_data_numpy)
#                    del(batchCorpus)
#                    del(batch_real_sentence_length)
#                    del(batch_idxs)
                    pass
        
        ##################
        #######第三步：将BigCloneBench的id和向量表示写入到pkl文件中去。
        ##################
        bigCloneBenchId_Map_vector_root={}
        bigCloneBenchId_Map_vector_mean={}
        bigCloneBenchId_Map_vector_weighted_TF_IDF={}
        bigCloneBenchId_Map_vector_meanAndRootMost={}
        bigCloneBenchId_Map_vector_weightedAndCengci={}
        
        lineMapNoinbigCloneBench_Corpus={}
        for i, line in enumerate(self.lines_for_bigcloneBench):
            lineMapNoinbigCloneBench_Corpus[line]=i
        for id_no in ids_in_BigCloneBench:
            line=id_line_dict[id_no] #先转换为fullCorpus中的line
            no_inbigCloneBench_Corpus=lineMapNoinbigCloneBench_Corpus[line] #再找到在bigCloneBench_Corpus中的序号。
            
            vector_root=corpus_sentence_vector_root[no_inbigCloneBench_Corpus] #取出向量。
#            vector_mean=corpus_sentence_vector_mean[no_inbigCloneBench_Corpus] #取出向量。
            
            bigCloneBenchId_Map_vector_root[id_no]=vector_root
#            bigCloneBenchId_Map_vector_mean[id_no]=vector_mean
#            
#            vector_weighted_TF_IDF=corpus_sentence_vector_weighted_TF_IDF[no_inbigCloneBench_Corpus]
#            bigCloneBenchId_Map_vector_weighted_TF_IDF[id_no]=vector_weighted_TF_IDF
#            
#            vector_meanAndRootMost=corpus_sentence_vector_meanAndRootMost[no_inbigCloneBench_Corpus] #取出向量。
#            bigCloneBenchId_Map_vector_meanAndRootMost[id_no]=vector_meanAndRootMost
#            
#            vector_weightedAndCengci=corpus_sentence_vector_weightedAndCengci[no_inbigCloneBench_Corpus] #取出向量。
#            bigCloneBenchId_Map_vector_weightedAndCengci[id_no]=vector_weightedAndCengci
#            
            
        self.save_to_pkl(bigCloneBenchId_Map_vector_root,path_pkl_root)
        self.save_to_pkl(bigCloneBenchId_Map_vector_mean,path_pkl_mean)
        self.save_to_pkl(bigCloneBenchId_Map_vector_weighted_TF_IDF,path_pkl_weighted_TF_IDF)
        self.save_to_pkl(bigCloneBenchId_Map_vector_meanAndRootMost,path_pkl_meanAndRootMost)
        self.save_to_pkl(bigCloneBenchId_Map_vector_weightedAndCengci,path_pkl_weightedAndCengci)
#        dict_id_map_vector=self.read_from_pkl(path)
        print('random:')
        print(random)
        print('\n')
        print(path_pkl_root)
        print(path_pkl_mean)
        print(path_pkl_weighted_TF_IDF)
        print(path_pkl_meanAndRootMost)
        print(path_pkl_weightedAndCengci)
        pass
        return 


    def using_model_for_BigCloneBench_experimentID_2(self):
 
#        self.need_vectorTree_lines_for_trainCorpus####这个就是一个列表。假设为[100,199.....]则，BigCloneBench中的某个ID编号对应语料库的编号为100，并且在self.bigCloneBench_Corpus的第0个位置
#        self.need_vectorTree_ids_for_trainCorpus
#        self.need_vectorTree_Corpus
#        self.need_vectorTree_Corpus_fixed_tree_constructionorder
#        self.need_vectorTree_Corpus_sentence_length
#        self.need_vectorTree_Corpus_max_sentence_length
        
        ##################
        #######第二步：计算self.bigCloneBench_Corpus中每个语料的向量表示
        ##################
        random_str=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
        path_pkl='./vectorTree/'+random_str+'_vectorTree.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_root)  
        else:
            print('no such file:%s'%path_pkl) # 则返回文件不存在
            
        
        ##################################################################################################
        corpus=self.need_vectorTree_Corpus
        corpus_fixed_tree_constructionorder=self.need_vectorTree_Corpus_fixed_tree_constructionorder
        corpus_sentence_length=self.need_vectorTree_Corpus_sentence_length
        corpus_sentence_nodes_vectors={}
        ##################################################################################################
        
        with tf.Graph().as_default():
#            self.xiaojie_RvNN_fixed_tree()#构建网络结构图 
            self.xiaojie_RvNN_fixed_tree_for_usingmodel()  ##这个模型是用来计算向量树的。我们这里只需要计算一个根节点的向量，所以，要将返回结果进行处理，避免开销过大。
            saver = tf.train.Saver()
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            config.gpu_options.allocator_type = 'BFC'
            with tf.Session(config=config) as sess:
############################################################################################
#                直接用随机化权重，计算一下，看看效果。
#                init=tf.initialize_all_variables()
#                sess.run(init)
                
                #用训练后模型
                weights_path='./weights/%s'%self.config.model_name
                saver.restore(sess, weights_path)#导入权重
                            
                filter_length=500;           
#                filter_length=100;           
                logs.log('设置长短的衡量标准是{}'.format(filter_length))
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                short_sentence_indexincorpus_list=[]
                long_sentence_indexincorpus_list=[]
                for index,length in enumerate(corpus_sentence_length):
                    if length<filter_length:
                        short_sentence_indexincorpus_list.append(index)
                    else:
                        long_sentence_indexincorpus_list.append(index)
                
                logs.log("较长的句子{}个".format(len(long_sentence_indexincorpus_list)))
                logs.log("较短的句子{}个".format(len(short_sentence_indexincorpus_list)))
                short_sentence_corpus=[corpus[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_length=[corpus_sentence_length[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in short_sentence_indexincorpus_list]
                
                long_sentence_corpus=[corpus[index] for index in long_sentence_indexincorpus_list]
                ####由于单独处理了，用的不是网络，而是另外一个计算过程，因此，就不需要这个长度信息了。
                long_sentence_corpus_length=[corpus_sentence_length[index] for index in long_sentence_indexincorpus_list]
                long_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in long_sentence_indexincorpus_list]
                
                logs.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
                logs.log("先处理较短的句子的语料，批处理开始")
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
                sentence_num=len(short_sentence_indexincorpus_list)
                #######读取词向量矩阵
                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
                wordEmbedding_size=wordEmbedding.shape[0]
                #######读取词向量矩阵
                data_idxs=list(range(sentence_num)) 
                i_set_batch_size=self.config.batch_size_using_model_notTrain
                
                #保存整个语料库的所有句子的最终向量表示
                xiaojie_batch_num=(sentence_num-1)/i_set_batch_size #从0开始编号的batch数目
                batch_index=0
                for i_range in range(0,sentence_num,i_set_batch_size):
                    #测试用#测试用#测试用#测试用#测试用
#                    batch_index=batch_index+1
#                    continue #测试用
                    #测试用#测试用#测试用#测试用#测试用
                    
                    
                    logs.log("batch_index:{}/{}".format(batch_index,xiaojie_batch_num))
                    
                    real_batch_size = min(i_range+i_set_batch_size,sentence_num)-i_range
                    batch_idxs=data_idxs[i_range:i_range+real_batch_size]
                    batchCorpus=[short_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[short_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[short_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
        #####################           
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
        #                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
                    #####用run_options选项可以查看内存崩溃的情况。
#                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
#                    print('loss:')
#                    print(loss)
                    ###这个batch_sentence_vectors中保存的是三维矩阵。第一维表示样本在批中的编号，第二维度的长度是词向量的长度，第三维是按照batch中最长节点数目的长度来的（补齐数据）
                    ###所以，我们要取出这个batch_sentence_vectors每个样本的向量树的话，就需要按照batch_real_sentence_length从补齐数据中再取回来。
                    
                    #####测试上面的过程是否正确。
                    #####紧接着，我们不用tensorflow网络的批计算，而是用一个句子一个句子计算的方式进行计算。进行测试
#                    step=0
#                    for sentence in batchCorpus:
#                        treeConstructionOrders=batchCorpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                        (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                        step=step+1;
                        ###接下来，比较root_vector在每一次循环中是否同前面的batch_setence_vectors中对应位置的向量相同即可判定，那个xiaojie_RvNN_fixed_tree_for_usingmodel函数中的add_loss_and_batch...有没有写正确。
                    ####将批句子的最终表示矩阵添加到整个预料的句子的最终表示矩阵
                    in_batch_number=0
                    for index in batch_idxs:
                        sentence_vectors_buqi=batch_sentence_vectors[in_batch_number,:,:] #取出补齐后的数据
                        sentence_length=batch_real_sentence_length[in_batch_number]
                        nodeslength=2*sentence_length-1
                        sentence_nodes_vectors=sentence_vectors_buqi[0:self.config.embed_size,0:nodeslength] #取出补齐前的数据
                        sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                        in_batch_number=in_batch_number+1
                    
                        ###将整个句子的所有向量保存下来
                        
                        sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                        sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
            
                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list
            
                    ######结束循环，我们将batch_index加上1
                    batch_index=batch_index+1
                    ######结束循环，我们将batch_index加上1
####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
                logs.log("再处理较长的句子的语料，每个句子单独处理，开始")   
                long_sentence_num=len(long_sentence_indexincorpus_list)
                data_idxs=list(range(long_sentence_num)) 
                for i_range,sentence in enumerate(long_sentence_corpus):
                    logs.log("long_setence_index:{}/{}".format(i_range,long_sentence_num))
                    real_batch_size = 1
                    batch_idxs=data_idxs[i_range:i_range+real_batch_size]
                    batchCorpus=[long_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[long_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
        #                    batch_real_sentence_length = list(map(len,batchCorpus))
        #                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]            
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
        #                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[long_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
        #                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
        #####################           
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
                    
                    sentence_nodes_vectors=batch_sentence_vectors[0,:,:] #batch中只有一个样本。
                    sentence_length=batch_real_sentence_length[0]
                    nodeslength=2*sentence_length-1
                    sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                    ###将整个句子的所有向量保存下来
                    sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                    sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
                    corpus_index=long_sentence_indexincorpus_list[i_range] #找出其在corpus中的序号
                    corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list
        
        ##################
        #######第三步：将need_vectorTree_ids_for_trainCorpus的id和向量表示写入到pkl文件中去。
        ##################
        need_vectorTree_id_Map_vectorTree={}
        lineMapNoinneed_vectorTree_Corpus={}
        for i, line in enumerate(self.need_vectorTree_lines_for_trainCorpus):
            lineMapNoinneed_vectorTree_Corpus[line]=i
        for id_no in self.need_vectorTree_ids_for_trainCorpus:
            line=self.id_line_dict[id_no] #先转换为fullCorpus中的line
            no_inneed_vectorTree_Corpus=lineMapNoinneed_vectorTree_Corpus[line] #再找到在bigCloneBench_Corpus中的序号。
            vectorTree=corpus_sentence_nodes_vectors[no_inneed_vectorTree_Corpus]#取出向量树
            need_vectorTree_id_Map_vectorTree[id_no]=vectorTree
            
        self.save_to_pkl(need_vectorTree_id_Map_vectorTree,path_pkl)
#        dict_id_map_vector=self.read_from_pkl(path)
        print(path_pkl)
        pass
        return    
    def using_model_for_BigCloneBench_experimentID_3(self):
        #(self.fullCorpus,self.fullCorpus_sentence_length,self.vocabulary,self.We,self.config.max_sentence_length_full_Corpus,self.full_corpus_fixed_tree_constructionorder)
        ##################
        #######计算fullCorpus中每个语料的向量表示
        ##################
#        random_str=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
#        path_pkl_root='./vector/'+random_str+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        if os.path.exists(path_pkl_root):  # 如果文件存在
#        # 删除文件，可使用以下两种方法。
#            os.remove(path_pkl_root)  
#        else:
#            print('no such file:%s'%path_pkl_root) # 则返回文件不存在
#        
#        xiaojiepkl_index_1=0;
#        random_str=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
#        path_pkl_mean='./vector/'+random_str+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
#        if os.path.exists(path_pkl_mean):  # 如果文件存在
#        # 删除文件，可使用以下两种方法。
#            os.remove(path_pkl_mean)  
#        else:
#            print('no such file:%s'%path_pkl_mean) # 则返回文件不存在
        xiaojiepkl_index=0 #由于占用内存过大，我们分开存储。然后后续再执行merge操作。
        path_pkl_root='./vector/'+str(xiaojiepkl_index)+'_using_Weigthed_RAE_fullCorpusLine_Map_Vector_root.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_root):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_root)  
        else:
            print('no such file:%s'%path_pkl_root) # 则返回文件不存在
        path_pkl_mean='./vector/'+str(xiaojiepkl_index)+'_using_Weigthed_RAE_fullCorpusLine_Map_Vector_mean.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path_pkl_mean):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path_pkl_mean)  
        else:
            print('no such file:%s'%path_pkl_mean) # 则返回文件不存在
        store_size=0
        ##################################################################################################
        ###测试用
#        corpus=[self.fullCorpus[0]]
#        corpus_fixed_tree_constructionorder=[self.full_corpus_fixed_tree_constructionorder[0]]
#        corpus_sentence_length=[self.fullCorpus_sentence_length[0]]
        ###测试用
        corpus=self.fullCorpus
        corpus_fixed_tree_constructionorder=self.full_corpus_fixed_tree_constructionorder
        corpus_sentence_length=self.fullCorpus_sentence_length
        corpus_fixed_tree_parentType_weight=self.full_corpus_fixed_tree_parentType_weight
        corpus_sentence_vector_root={}
        corpus_sentence_vector_mean={}
        del(self.fullCorpus)
        del(self.fullCorpus_sentence_length)
        del(self.full_corpus_fixed_tree_constructionorder)
        ##################################################################################################
        
        with tf.Graph().as_default():
#            self.xiaojie_RvNN_fixed_tree()#构建网络结构图 
            self.xiaojie_RvNN_fixed_tree_for_usingmodel()  ##这个模型是用来计算向量树的。我们这里只需要计算一个根节点的向量，所以，要将返回结果进行处理，避免开销过大。
            saver = tf.train.Saver()
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            config.gpu_options.allocator_type = 'BFC'
            with tf.Session(config=config) as sess:
############################################################################################
#                直接用随机化权重，计算一下，看看效果。
#                init=tf.initialize_all_variables()
#                sess.run(init)
                
                #用训练后模型
                weights_path='./weights/%s'%self.config.model_name
                saver.restore(sess, weights_path)#导入权重
                            
                filter_length=300;     #主要不能因为对齐而耗尽内存。可以增大批的规模，降低对齐要对齐的长度。       
#                filter_length=100;           
                logs.log('设置长短的衡量标准是{}'.format(filter_length))
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                short_sentence_indexincorpus_list=[]
                long_sentence_indexincorpus_list=[]
                for index,length in enumerate(corpus_sentence_length):
                    if length<filter_length:
                        short_sentence_indexincorpus_list.append(index)
                    else:
                        long_sentence_indexincorpus_list.append(index)
                
                logs.log("较长的句子{}个".format(len(long_sentence_indexincorpus_list)))
                logs.log("较短的句子{}个".format(len(short_sentence_indexincorpus_list)))
                short_sentence_corpus=[corpus[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_length=[corpus_sentence_length[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_fixed_tree_parentType_weight=[corpus_fixed_tree_parentType_weight[index] for index in short_sentence_indexincorpus_list]
                long_sentence_corpus=[corpus[index] for index in long_sentence_indexincorpus_list]
                ####由于单独处理了，用的不是网络，而是另外一个计算过程，因此，就不需要这个长度信息了。
                long_sentence_corpus_length=[corpus_sentence_length[index] for index in long_sentence_indexincorpus_list]
                long_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in long_sentence_indexincorpus_list]
                long_sentence_corpus_fixed_tree_parentType_weight=[corpus_fixed_tree_parentType_weight[index] for index in long_sentence_indexincorpus_list]
                logs.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
                logs.log("先处理较短的句子的语料，批处理开始")
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
                sentence_num=len(short_sentence_indexincorpus_list)
                #######读取词向量矩阵
                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
                del(self.We)
                wordEmbedding_size=wordEmbedding.shape[0]
                #######读取词向量矩阵
                data_idxs=list(range(sentence_num)) 
                i_set_batch_size=self.config.batch_size_using_model_notTrain
                
                #保存整个语料库的所有句子的最终向量表示
                xiaojie_batch_num=(sentence_num-1)/i_set_batch_size #从0开始编号的batch数目
                batch_index=0
                for i_range in range(0,sentence_num,i_set_batch_size):
                    #测试用#测试用#测试用#测试用#测试用
#                    batch_index=batch_index+1
#                    continue #测试用
                    #测试用#测试用#测试用#测试用#测试用
                    
                    
                    logs.log("batch_index:{}/{}".format(batch_index,xiaojie_batch_num))
                    
                    real_batch_size = min(i_range+i_set_batch_size,sentence_num)-i_range
                    batch_idxs=data_idxs[i_range:i_range+real_batch_size]
                    batchCorpus=[short_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[short_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[short_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
        #####################           
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
        #                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
                    #####用run_options选项可以查看内存崩溃的情况。
#                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
#                    print('loss:')
#                    print(loss)
                    ###这个batch_sentence_vectors中保存的是三维矩阵。第一维表示样本在批中的编号，第二维度的长度是词向量的长度，第三维是按照batch中最长节点数目的长度来的（补齐数据）
                    ###所以，我们要取出这个batch_sentence_vectors每个样本的向量树的话，就需要按照batch_real_sentence_length从补齐数据中再取回来。
                    
                    #####测试上面的过程是否正确。
                    #####紧接着，我们不用tensorflow网络的批计算，而是用一个句子一个句子计算的方式进行计算。进行测试
#                    step=0
#                    for sentence in batchCorpus:
#                        treeConstructionOrders=batchCorpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                        (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                        step=step+1;
                        ###接下来，比较root_vector在每一次循环中是否同前面的batch_setence_vectors中对应位置的向量相同即可判定，那个xiaojie_RvNN_fixed_tree_for_usingmodel函数中的add_loss_and_batch...有没有写正确。
                    ####将批句子的最终表示矩阵添加到整个预料的句子的最终表示矩阵
                    
                    x=[]
                    batchCorpus_fixed_tree_parentType_weight=[short_sentence_corpus_fixed_tree_parentType_weight[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_parentType_weight in enumerate(batchCorpus_fixed_tree_parentType_weight):
                        #########################
                        ###我们要计算非叶子节点在整棵树中的权重信息。读入的tf-idf值要进一步转换
                        sumsum=np.sum(sentence_fixed_tree_parentType_weight)
                        sentence_fixed_tree_parentType_weight=sentence_fixed_tree_parentType_weight/sumsum
        #                print(np.sum(sentence_fixed_tree_parentType_weight)) #看看是不是1
                        #########################
                        
                        sentence_fixed_tree_parentType_weight_len=batch_real_sentence_length[i]-1
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_parentType_weight_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_parentType_weight,L2)) #一维信息直接连接，不用axis
                        x.append(L)
                    batch_fixed_tree_parentType_weight=np.array(x)
                    
                    
                    in_batch_number=0
                    for index in batch_idxs:
                        sentence_vectors_buqi=batch_sentence_vectors[in_batch_number,:,:] #取出补齐后的数据
                        #取出权重系数
                        sentence_fixed_tree_parentType_weight=batch_fixed_tree_parentType_weight[in_batch_number]
                        #取出权重系数
                        sentence_length=batch_real_sentence_length[in_batch_number]
                        nodeslength=2*sentence_length-1
                        sentence_nodes_vectors=sentence_vectors_buqi[0:self.config.embed_size,0:nodeslength] #取出补齐前的数据
                        sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                        in_batch_number=in_batch_number+1
                    
                        ###将整个句子的所有向量保存下来
                        
                        sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                        sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
            
                        ####################用根节点的向量
                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
        #                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
                        #我们可以保存列表，也可以保存一个单独的向量。
                        #我们先仅仅保存根节点的向量
                        corpus_sentence_vector_root[corpus_index]=sentence_nodes_vectors_list[nodeslength-1]
            ####################用根节点的向量
            ####################用所有节点的平均向量
        #                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
        ##                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
        #                        #我们可以保存列表，也可以保存一个单独的向量。
        #                        #我们先仅仅保存根节点的向量
#                        mean_vector=np.mean(sentence_nodes_vectors,axis=0)
#                        corpus_sentence_vector_mean[corpus_index]=mean_vector
                        corpus_index=short_sentence_indexincorpus_list[index]
                        vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
                        for i in range(sentence_length,nodeslength):
                            vector=sentence_nodes_vectors_list[i]
                            j=i-sentence_length
                            xishu=sentence_fixed_tree_parentType_weight[j]
                            vector=np.multiply(vector,xishu)
                            vector_weighted=np.add(vector_weighted,vector)
                        corpus_sentence_vector_mean[corpus_index]=vector_weighted
            ####################用所有节点的平均向量
                    ######结束循环，我们将batch_index加上1
                    ##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储
                    if(store_size>70000):
                        self.save_to_pkl(corpus_sentence_vector_root,path_pkl_root)
                        self.save_to_pkl(corpus_sentence_vector_mean,path_pkl_mean)
                        store_size=0
                        del(corpus_sentence_vector_root)
                        del(corpus_sentence_vector_mean)
                        corpus_sentence_vector_root={}
                        corpus_sentence_vector_mean={}
                        xiaojiepkl_index+=1 #由于占用内存过大，我们分开存储。然后后续再执行merge操作。
                        path_pkl_root='./vector/'+str(xiaojiepkl_index)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
                        if os.path.exists(path_pkl_root):  # 如果文件存在
                            # 删除文件，可使用以下两种方法。
                            os.remove(path_pkl_root)  
                        else:
                            print('no such file:%s'%path_pkl_root) # 则返回文件不存在
        
                        path_pkl_mean='./vector/'+str(xiaojiepkl_index)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
                        if os.path.exists(path_pkl_mean):  # 如果文件存在
                        # 删除文件，可使用以下两种方法。
                            os.remove(path_pkl_mean)  
                        else:
                            print('no such file:%s'%path_pkl_mean) # 则返回文件不存在
                    ##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储
                    batch_index=batch_index+1
                    store_size=store_size+real_batch_size
                    ######结束循环，我们将batch_index加上1
####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
                    ##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储                    
                self.save_to_pkl(corpus_sentence_vector_root,path_pkl_root)
                self.save_to_pkl(corpus_sentence_vector_mean,path_pkl_mean)
                del(corpus_sentence_vector_root)
                del(corpus_sentence_vector_mean)
                corpus_sentence_vector_root={}
                corpus_sentence_vector_mean={}
                xiaojiepkl_index+=1 #由于占用内存过大，我们分开存储。然后后续再执行merge操作。
                path_pkl_root='./vector/'+str(xiaojiepkl_index)+'_fullCorpusLine_Map_Vector_root.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
                if os.path.exists(path_pkl_root):  # 如果文件存在
                    # 删除文件，可使用以下两种方法。
                    os.remove(path_pkl_root)  
                else:
                    print('no such file:%s'%path_pkl_root) # 则返回文件不存在
                    
                path_pkl_mean='./vector/'+str(xiaojiepkl_index)+'_fullCorpusLine_Map_Vector_mean.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
                if os.path.exists(path_pkl_mean):  # 如果文件存在
                # 删除文件，可使用以下两种方法。
                    os.remove(path_pkl_mean)  
                else:
                    print('no such file:%s'%path_pkl_mean) # 则返回文件不存在
                ##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储##分批存储
                
                    
                    
                logs.log("再处理较长的句子的语料，每个句子单独处理，开始")   
                long_sentence_num=len(long_sentence_indexincorpus_list)
                data_idxs=list(range(long_sentence_num)) 
                for i_range,sentence in enumerate(long_sentence_corpus):
                    logs.log("long_setence_index:{}/{}".format(i_range,long_sentence_num))
                    real_batch_size = 1
                    batch_idxs=data_idxs[i_range:i_range+real_batch_size]
                    batchCorpus=[long_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[long_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
        #                    batch_real_sentence_length = list(map(len,batchCorpus))
        #                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]            
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
        #                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[long_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    
                    x=[]
                    batchCorpus_fixed_tree_parentType_weight=[long_sentence_corpus_fixed_tree_parentType_weight[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_parentType_weight in enumerate(batchCorpus_fixed_tree_parentType_weight):
                        #########################
                        ###我们要计算非叶子节点在整棵树中的权重信息。读入的tf-idf值要进一步转换
                        sumsum=np.sum(sentence_fixed_tree_parentType_weight)
                        sentence_fixed_tree_parentType_weight=sentence_fixed_tree_parentType_weight/sumsum
        #                print(np.sum(sentence_fixed_tree_parentType_weight)) #看看是不是1
                        #########################
                        
                        sentence_fixed_tree_parentType_weight_len=batch_real_sentence_length[i]-1
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_parentType_weight_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_parentType_weight,L2)) #一维信息直接连接，不用axis
                        x.append(L)
                    batch_fixed_tree_parentType_weight=np.array(x)
                    
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
        #                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
        #####################           
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
                    
                    sentence_nodes_vectors=batch_sentence_vectors[0,:,:] #batch中只有一个样本。
                    #取出权重系数
                    sentence_fixed_tree_parentType_weight=batch_fixed_tree_parentType_weight[0]
                    #取出权重系数
                    sentence_length=batch_real_sentence_length[0]
                    nodeslength=2*sentence_length-1
                    sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                    ###将整个句子的所有向量保存下来
                    sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                    sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
#                    corpus_index=long_sentence_indexincorpus_list[i_range] #找出其在corpus中的序号
#                    corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list
                    
         ####################用根节点的向量
                    corpus_index=long_sentence_indexincorpus_list[i_range] #找出其在corpus中的序号
#                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
                        #我们可以保存列表，也可以保存一个单独的向量。
                        #我们先仅仅保存根节点的向量
                    corpus_sentence_vector_root[corpus_index]=sentence_nodes_vectors_list[nodeslength-1]
        ####################用根节点的向量
                    ####################用所有节点的平均向量
#                    corpus_index=long_sentence_indexincorpus_list[i_range] #找出其在corpus中的序号                    
#                    mean_vector=np.mean(sentence_nodes_vectors,axis=0)
#                    corpus_sentence_vector_mean[corpus_index]=mean_vector  
                    
                    ####################用所有节点的平均向量
                    #用加权向量
                    corpus_index=short_sentence_indexincorpus_list[index]
                    vector_weighted=np.zeros_like(sentence_nodes_vectors_list[0],np.float32)
                    for i in range(sentence_length,nodeslength):
                        vector=sentence_nodes_vectors_list[i]
                        j=i-sentence_length
                        xishu=sentence_fixed_tree_parentType_weight[j]
                        vector=np.multiply(vector,xishu)
                        vector_weighted=np.add(vector_weighted,vector)
                    corpus_sentence_vector_mean[corpus_index]=vector_weighted
        
        self.save_to_pkl(corpus_sentence_vector_root,path_pkl_root)
        self.save_to_pkl(corpus_sentence_vector_mean,path_pkl_mean)
        print(path_pkl_root)
        print(path_pkl_mean)
        del(corpus_sentence_vector_root)
        del(corpus_sentence_vector_mean)
        pass
        return     
    def save_to_pkl(self,python_content, pickle_name):
        with open(pickle_name, 'wb') as pickle_f:
            pickle.dump(python_content, pickle_f)
    def read_from_pkl(self,pickle_name):
        with open(pickle_name, 'rb') as pickle_f:
            python_content = pickle.load(pickle_f)
        return python_content      
    def similarities(self,corpus,corpus_sentence_length,corpus_fixed_tree_constructionorder,weights_path):#在语料库计算句子与句子之间的相似性，使用的是模型训练好的权重，即weight_path
        #由于对模型进行评估的时候，不需要像训练那样递归搭建网络，因此不需要上面的for循环和ReSET_AFTER控制网络规模，可以和之前的run_epoch进行对比。
        logs.log('对语料库计算句与句的相似性')             
        logs.log('被相似计算的语料库一共{}个sentence'.format(len(corpus)))
        #建立一个numpy矩阵，存储矩阵。用numpy矩阵而不是字典的原因是，对于矩阵，有计算向量之间距离的快捷方法。
        #对于每一个句子，我们可以选择用树的顶点去度量，也可以用整颗树的所有节点的向量的平均值进行度量。所以我们要建立一个三维矩阵
#        features=np.zeros((self.config.embed_size,len(corpus),2),dtype=np.float64)

        #为了避免这么庞大的一个程序因为这个矩阵存储的文件位置，而导致程序执行失败。我们随机生成一个字符串作为目标矩阵存储的文件名。
#        path = 'xiaojie_corpus_setence_vector.npy'  # 文件路径     
        random_str=''.join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
        path=random_str+'.xiaojiepkl'#为了便于找到这个随机文件，我们用xiaojienpy作为格式后缀
        if os.path.exists(path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
            os.remove(path)  
        else:
            print('no such file:%s'%path) # 则返回文件不存在
        corpus_sentence_nodes_vectors={}
        
        
        with tf.Graph().as_default():
#            self.xiaojie_RvNN_fixed_tree()#构建网络结构图
            self.xiaojie_RvNN_fixed_tree_for_usingmodel()
            saver = tf.train.Saver()
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            config.gpu_options.allocator_type = 'BFC'
            with tf.Session(config=config) as sess:
                saver.restore(sess, weights_path)#导入权重
#                直接用随机化权重，计算一下，看看效果。
#                init=tf.initialize_all_variables()
#                sess.run(init)
#测试随机化权重和模型的交叉熵。比较一下模型的训练是否影响最后的检测结果。
###################################################################################################################
####################################################################################################################                
#                loss2=self.computeloss_withAST(corpus[0],corpus_fixed_tree_constructionorder[0])
#                loss3,_,root_vector1,root_vector2=self.computelossAndVector_no_tensor_withAST(corpus[0],corpus_fixed_tree_constructionorder[0])
#                print(loss2,loss3)
#                epoch=0
#                
#                
#                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
#                wordEmbedding_size=wordEmbedding.shape[0]
#                x=[]
#                for sentence in self.trainCorpus:
#                    sentence_len=len(sentence) #句子长度
#                    ones=np.ones_like(sentence)
#                    words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
#                    L1 = wordEmbedding[:,words_indexed]
#                    buqi=self.config.max_sentence_length_train_Corpus-sentence_len
#                    L2=np.zeros([wordEmbedding_size,buqi],np.float64)
#                    L=np.concatenate((L1,L2),axis=1)
#                    x.append(L)
#                self.wordEmbedding_train_corpus=np.array(x)
#                self.train_corpus_real_sentence_length=[len(sentence) for sentence in self.trainCorpus]
#                
#                #处理train_corpus的抽象语法树构建顺序的数据，同样需要补齐
#                x=[]
#                for sentence_fixed_tree_constructionorder in self.train_corpus_fixed_tree_constructionorder:
#                    sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
#                    buqi=(self.config.max_sentence_length_train_Corpus-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
#                    L2=np.zeros([3,buqi],np.int32)
#                    L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
#                    x.append(L)
#                self.train_corpus_fixed_tree_constructionorder_for_tensorflow_network=np.array(x)        
#                loss_history_evalution= self.run_epoch(sess,epoch,corpusData=self.wordEmbedding_train_corpus,corpus_real_sentence_length=self.train_corpus_real_sentence_length,corpus_fixed_tree_constructionorder=self.train_corpus_fixed_tree_constructionorder_for_tensorflow_network,training=False)#这个loss_history是不包括参数正则化误差的，因为是评估。我们只取重构误差。并且每个元素对应一个batch样本集的平均损失。评估时用批只是为了计算更快。
#                print(np.mean(loss_history_evalution))
#                
#                input('input:')#在命令行执行时可以暂停程序，从而观察程序执行结果。
###################################################################################################################
###################################################################################################################                


###################################################################################################################
###################################################################################################################      
###测试add_loss_and_batchSentenceNodesVector_fixed_tree中计算的每批样本中每个样本的最终向量表示和我们不用该函数计算的是否一样
###
#
#                sentence_num=len(corpus)
#                #######读取词向量矩阵
#                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
#                wordEmbedding_size=wordEmbedding.shape[0]
#                #######读取词向量矩阵
#                data_idxs=list(range(sentence_num)) 
#                i_set_batch_size=self.config.batch_size
#                batch_index=0
##                for i in range(0,sentence_num,i_set_batch_size):
#                for i in range(0,1):#只取第一batch做测试用
#                    real_batch_size = min(i+i_set_batch_size,sentence_num)-i
#                    batch_idxs=data_idxs[i:i+real_batch_size]
#                    batchCorpus=[corpus[index] for index in batch_idxs]#[i:i+batch_size]
#                    sizes = list(map(len,batchCorpus))
#                    batch_max_sentence_length=max(sizes)
#                    x=[]
#                    for sentence in batchCorpus:
#                        sentence_len=len(sentence) #句子长度
#                        ones=np.ones_like(sentence)
#                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
#                        L1 = wordEmbedding[:,words_indexed]
#                        #######
#                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
#                        #######
#                        buqi=batch_max_sentence_length-sentence_len
#                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
#                        L=np.concatenate((L1,L2),axis=1)
#                        x.append(L)
#                    wordEmbedding_batchCorpus=np.array(x)
#                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
#                    ###
#                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
#                    ###
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
#                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
#                    x=[]
#                    batchCorpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
#                    
#                    for sentence_fixed_tree_constructionorder in batchCorpus_fixed_tree_constructionorder:
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
#                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
#                        L2=np.zeros([3,buqi],np.int32)
#                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
#                        x.append(L)
#                    batch_fixed_tree_constructionorder=np.array(x)
#        #####################           
#                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
#        #                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
#                    loss,batch_sentece_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
#                
#                #####紧接着，我们不用tensorflow网络的批计算，而是用一个句子一个句子计算的方式进行计算。
#                    step=0
#                    for sentence in batchCorpus:
#                        treeConstructionOrders=batchCorpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                        (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                        step=step+1;
#                        ###接下来，比较root_vector在每一次循环中是否同前面的batch_setence_vectors中对应位置的向量相同即可判定，那个xiaojie_RvNN_fixed_tree_for_usingmodel函数中的add_loss_and_batch...有没有写正确。
###################################################################################################################
###################################################################################################################    
###################################################################################################################
###################################################################################################################    

#####################
#####################
##由于corpus中有一些句子特别长的样本，甚至超出了5000个单词，那么对应的句子长度就是10000，如果300维度的话，然后又用批次的话，就会导致出现特别大的tensor。              
##所以，我们预先将corpus中的某些大样本抽取出来。单独进行计算。最后再进行整合。但是我们最后整合的时候，还要保证其在原语料库中的顺序。
                #对corpus进行处理，抽取句子长度特别长的语料。经过试验观察发现，句子长度1000以下的都能处理。但是超过1000的，最好单独计算。
                filter_length=1000;           
#                filter_length=100;           
                logs.log('设置长短的衡量标准是{}'.format(filter_length))
                #训练数据，从小于max sentence length的句子作为训练集合 #并且至少两个单词以上。
                short_sentence_indexincorpus_list=[]
                long_sentence_indexincorpus_list=[]
                for index,length in enumerate(corpus_sentence_length):
                    if length<filter_length:
                        short_sentence_indexincorpus_list.append(index)
                    else:
                        long_sentence_indexincorpus_list.append(index)
                
                logs.log("较长的句子{}个".format(len(long_sentence_indexincorpus_list)))
                logs.log("较短的句子{}个".format(len(short_sentence_indexincorpus_list)))
                short_sentence_corpus=[corpus[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_length=[corpus_sentence_length[index] for index in short_sentence_indexincorpus_list]
                short_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in short_sentence_indexincorpus_list]
                
                long_sentence_corpus=[corpus[index] for index in long_sentence_indexincorpus_list]
                ####由于单独处理了，用的不是网络，而是另外一个计算过程，因此，就不需要这个长度信息了。
                long_sentence_corpus_length=[corpus_sentence_length[index] for index in long_sentence_indexincorpus_list]
                long_sentence_corpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in long_sentence_indexincorpus_list]
                
                logs.log("计算较短的句子的全部语料，我们走批处理。长句子，我们每个句子单独计算")
                logs.log("先处理较短的句子的语料，批处理开始")
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
####################较短句子的语料集合，走批处理           
####################较短句子的语料集合，走批处理
                sentence_num=len(short_sentence_indexincorpus_list)
                #######读取词向量矩阵
                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
                wordEmbedding_size=wordEmbedding.shape[0]
                #######读取词向量矩阵
                data_idxs=list(range(sentence_num)) 
                i_set_batch_size=self.config.batch_size_using_model_notTrain
                
                #保存整个语料库的所有句子的最终向量表示
                xiaojie_batch_num=(sentence_num-1)/i_set_batch_size #从0开始编号的batch数目
                batch_index=0
                for i in range(0,sentence_num,i_set_batch_size):
                    
                    logs.log("batch_index:{}/{}".format(batch_index,xiaojie_batch_num))
                    
                    real_batch_size = min(i+i_set_batch_size,sentence_num)-i
                    batch_idxs=data_idxs[i:i+real_batch_size]
                    batchCorpus=[short_sentence_corpus[index] for index in batch_idxs]#[i:i+batch_size]
                    batch_real_sentence_length=[short_sentence_corpus_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
#                    batch_real_sentence_length = list(map(len,batchCorpus))
#                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
                    batch_max_sentence_length=max(batch_real_sentence_length)
                    x=[]
                    for i,sentence in enumerate(batchCorpus):
#                        sentence_len=len(sentence) #句子长度
                        sentence_len=batch_real_sentence_length[i]
                        ones=np.ones_like(sentence)
                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
                        L1 = wordEmbedding[:,words_indexed]
                        #######
                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
                        #######
                        buqi=batch_max_sentence_length-sentence_len
                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
                        L=np.concatenate((L1,L2),axis=1)
                        x.append(L)
                    wordEmbedding_batchCorpus=np.array(x)
                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
                    ###
                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
                    ###
                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
                    x=[]
                    batchCorpus_fixed_tree_constructionorder=[short_sentence_corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
                    
                    for i,sentence_fixed_tree_constructionorder in enumerate(batchCorpus_fixed_tree_constructionorder):
                        sentence_fixed_tree_constructionorder_len=batch_real_sentence_length[i]-1
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
                        L2=np.zeros([3,buqi],np.int32)
                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
                        x.append(L)
                    batch_fixed_tree_constructionorder=np.array(x)
        #####################           
                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
        #                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
                    #####用run_options选项可以查看内存崩溃的情况。
#                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
                    
                    ###这个batch_sentence_vectors中保存的是三维矩阵。第一维表示样本在批中的编号，第二维度的长度是词向量的长度，第三维是按照batch中最长节点数目的长度来的（补齐数据）
                    ###所以，我们要取出这个batch_sentence_vectors每个样本的向量树的话，就需要按照batch_real_sentence_length从补齐数据中再取回来。
                    
                    #####测试上面的过程是否正确。
                    #####紧接着，我们不用tensorflow网络的批计算，而是用一个句子一个句子计算的方式进行计算。进行测试
#                    step=0
#                    for sentence in batchCorpus:
#                        treeConstructionOrders=batchCorpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                        (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                        step=step+1;
                        ###接下来，比较root_vector在每一次循环中是否同前面的batch_setence_vectors中对应位置的向量相同即可判定，那个xiaojie_RvNN_fixed_tree_for_usingmodel函数中的add_loss_and_batch...有没有写正确。
                    ####将批句子的最终表示矩阵添加到整个预料的句子的最终表示矩阵
                    in_batch_number=0
                    for index in batch_idxs:
                        sentence_vectors_buqi=batch_sentence_vectors[in_batch_number,:,:] #取出补齐后的数据
                        sentence_length=batch_real_sentence_length[in_batch_number]
                        nodeslength=2*sentence_length-1
                        sentence_nodes_vectors=sentence_vectors_buqi[0:self.config.embed_size,0:nodeslength] #取出补齐前的数据
                        sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
                        in_batch_number=in_batch_number+1
                    
                        ###将整个句子的所有向量保存下来
                        
                        sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
                        sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                        #index就是在short_sentence_corpus中的序号，而不是整个corpus中的序号
                        corpus_index=short_sentence_indexincorpus_list[index] #找出其在corpus中的序号
                        corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
                    
                    ######结束循环，我们将batch_index加上1
                    batch_index=batch_index+1
                    ######结束循环，我们将batch_index加上1
####################较长句子的语料集合，每个句子单独处理           
####################较长句子的语料集合，每个句子单独处理
####################较长句子的语料集合，每个句子处理           
####################较长句子的语料集合，每个句子处理
                logs.log("再处理较长的句子的语料，每个句子单独处理，开始")   

                
                long_sentence_num=len(long_sentence_indexincorpus_list)
                for index,sentence in enumerate(long_sentence_corpus):
                    logs.log("long_setence_index:{}/{}".format(index,long_sentence_num))
                    treeConstructionOrders=long_sentence_corpus_fixed_tree_constructionorder[index]#取出抽象语法树的构建过程
                    (_,sentence_nodes_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
                    #sentence_nodes_vectors是一个字典
                    sentence_nodes_vectors_list=[]
                    for kk in range(2*long_sentence_corpus_length[index]-1):
                        sentece_vectors=sentence_nodes_vectors[kk]
                        sentece_vectors=sentece_vectors[:,0]
                        sentece_vectors=sentece_vectors.astype(np.float32)#必须转换为32位类型
                        sentence_nodes_vectors_list.append(sentece_vectors)
                    #sentence_nodes每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
                    #index就是在long_sentence_corpus中的序号，而不是整个corpus中的序号
                    corpus_index=long_sentence_indexincorpus_list[index] #找出其在corpus中的序号
                    corpus_sentence_nodes_vectors[corpus_index]=sentence_nodes_vectors_list ###后续写入pkl文件
                #########################测试，将corpus设为比较小的语料。然后看看上面最后生成的corpus_sentence_nodes_vectors中的内容和下面的是否一致。
                ################测试上面的过程是否正确。正常执行时下面代码必须注释掉
#                for index,sentence in enumerate(corpus):
#                    treeConstructionOrders=corpus_fixed_tree_constructionorder[index]#取出抽象语法树的构建过程
#                    (_,sentence_nodes_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                    print ('测试')
                
                ####将三维矩阵保存起来
                with open(path, 'wb') as f:
                    pickle.dump(corpus_sentence_nodes_vectors, f)

                ## read
                #with open('vec_tree.pkl', 'rb') as f:
                #    haha = pickle.load(f)
                logs.log('相似性计算结束后，corpus的所有句子的最终向量表示(向量树)存储的位置是为%s'%path) 
                
#####################
#####################                
####下面的代码是不区分长短的时候用的。即全部语料走批处理时可以用。              
#                sentence_num=len(corpus)
#                #######读取词向量矩阵
#                wordEmbedding=np.array(self.We)#将列表转变为numpy数组，方便操作。因为列表不支持列切片
#                wordEmbedding_size=wordEmbedding.shape[0]
#                #######读取词向量矩阵
#                data_idxs=list(range(sentence_num)) 
#                i_set_batch_size=self.config.batch_size_using_model_notTrain
#                
#                #保存整个语料库的所有句子的最终向量表示
#                y = np.arange(sentence_num, dtype=np.float32) #由于淘宝的nsg算法的开源代码只能处理float32类型，因此这里就是float32
#                corpus_sentence_vectors=np.zeros_like(y)
#                corpus_sentence_vectors=np.tile(corpus_sentence_vectors,(300,1))
#                
#                #保存整个语料库的所有句子的最终向量表示
#                xiaojie_batch_num=(sentence_num-1)/i_set_batch_size #从0开始编号的batch数目
#                batch_index=0
#                for i in range(0,sentence_num,i_set_batch_size):
#                    
#                    logs.log("batch_index:{}/{}".format(batch_index,xiaojie_batch_num))
#                    
#                    real_batch_size = min(i+i_set_batch_size,sentence_num)-i
#                    batch_idxs=data_idxs[i:i+real_batch_size]
#                    batchCorpus=[corpus[index] for index in batch_idxs]#[i:i+batch_size]
#                    sizes=[corpus_sentence_length[index] for index in batch_idxs] #计算的速度要比检索的速度更快。因为检索是在50万个列表里进行搜索。我实际实验观察了一下，发现计算的速度更快。就是计算batch的长度列表
##                    sizes = list(map(len,batchCorpus))
#                    batch_max_sentence_length=max(sizes)
#                    x=[]
#                    for sentence in batchCorpus:
#                        sentence_len=len(sentence) #句子长度
#                        ones=np.ones_like(sentence)
#                        words_indexed=sentence-ones#注意，单词的索引编号是从1开始。但是对于词向量矩阵而言，都是从0开始的。所以，要进行减去1的操作，才能去取出词向量。
#                        L1 = wordEmbedding[:,words_indexed]
#                        #######
#                        ##这里添加手工验证，就是分析第几个语句中有那个编号为0的单词，这些单词是word2vec没有训练出词向量的单词。对于这些单词，我们从词向量矩阵的末尾取出零矩阵，用0-1即-1去索引。所以，编号为0.
#                        #######
#                        buqi=batch_max_sentence_length-sentence_len
#                        L2=np.zeros([wordEmbedding_size,buqi],np.float64)
#                        L=np.concatenate((L1,L2),axis=1)
#                        x.append(L)
#                    wordEmbedding_batchCorpus=np.array(x)
#                    batch_data_numpy=np.array(wordEmbedding_batchCorpus,np.float64)
#                    ###
#                    #corpusData是三个维度的numpy。第一个维度大小表示语料库中句子的数目；第二个维度大小是词向量的长度；第三个维度长度是语料库中最长句子的长度
#                    ###
##                    batch_real_sentence_length=[len(sentence) for sentence in batchCorpus]
#                    batch_real_sentence_length=sizes
#                    #处理batchCorpus的抽象语法树构建顺序的数据，同样需要补齐
#                    x=[]
#                    batchCorpus_fixed_tree_constructionorder=[corpus_fixed_tree_constructionorder[index] for index in batch_idxs]
#                    
#                    for sentence_fixed_tree_constructionorder in batchCorpus_fixed_tree_constructionorder:
#                        sentence_fixed_tree_constructionorder_len=sentence_fixed_tree_constructionorder.shape[1] #句子长度
#                        buqi=(batch_max_sentence_length-1)-sentence_fixed_tree_constructionorder_len#注意，抽象语法树构建的次数是句子的长度减去1。
#                        L2=np.zeros([3,buqi],np.int32)
#                        L=np.concatenate((sentence_fixed_tree_constructionorder,L2),axis=1)
#                        x.append(L)
#                    batch_fixed_tree_constructionorder=np.array(x)
#        #####################           
#                    feed={self.input:batch_data_numpy,self.batch_real_sentence_length:batch_real_sentence_length,self.batch_len:[real_batch_size],self.batch_treeConstructionOrders:batch_fixed_tree_constructionorder}
#        #                loss,_,_=sess.run([self.tensorLoss_fixed_tree,self.train_op,self.tfPrint],feed_dict=feed)#调试用代码行
#                    #####用run_options选项可以查看内存崩溃的情况。
##                    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
##                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed,options=run_options)
#                    loss,batch_sentence_vectors=sess.run([self.tensorLoss_fixed_tree,self.batch_sentence_vectors],feed_dict=feed)
#                    
#                    ###这个batch_sentence_vectors中保存的是三维矩阵。第一维表示样本在批中的编号，第二维度的长度是词向量的长度，第三维是按照batch中最长节点数目的长度来的（补齐数据）
#                    ###所以，我们要取出这个batch_sentence_vectors每个样本的向量树的话，就需要按照batch_real_sentence_length从补齐数据中再取回来。
#                    
#                    #####测试上面的过程是否正确。
#                    #####紧接着，我们不用tensorflow网络的批计算，而是用一个句子一个句子计算的方式进行计算。进行测试
##                    step=0
##                    for sentence in batchCorpus:
##                        treeConstructionOrders=batchCorpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
##                        (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
##                        step=step+1;
#                        ###接下来，比较root_vector在每一次循环中是否同前面的batch_setence_vectors中对应位置的向量相同即可判定，那个xiaojie_RvNN_fixed_tree_for_usingmodel函数中的add_loss_and_batch...有没有写正确。
#                    ####将批句子的最终表示矩阵添加到整个预料的句子的最终表示矩阵
#                    in_batch_number=0
#                    for index in batch_idxs:
#                        sentence_vectors_buqi=batch_sentence_vectors[in_batch_number,:,:] #取出补齐后的数据
#                        sentence_length=batch_real_sentence_length[in_batch_number]
#                        nodeslength=2*sentence_length-1
#                        sentence_nodes_vectors=sentence_vectors_buqi[0:self.config.embed_size,0:nodeslength] #取出补齐前的数据
#                        sentence_nodes_vectors=sentence_nodes_vectors.astype(np.float32)#必须转换为32位类型
#                        in_batch_number=in_batch_number+1
#                    
#                        ###将整个句子的所有向量保存下来
#                        
#                        sentence_nodes_vectors=np.transpose(sentence_nodes_vectors) #转置
#                        sentence_nodes_vectors_list=list(sentence_nodes_vectors) #每个元素是300维向量。 #假定句子的叶子和非叶子的总数即长度为29。
#                        #index就是在整个corpus中的序号
#                        corpus_sentence_nodes_vectors[index]=sentence_nodes_vectors_list ###后续写入pkl文件
#                    
#                    ######结束循环，我们将batch_index加上1
#                    batch_index=batch_index+1
#                    ######结束循环，我们将batch_index加上1
#                    
#                ####将三维矩阵保存起来
#                with open(path, 'wb') as f:
#                    pickle.dump(corpus_sentence_nodes_vectors, f)
#
#                ## read
#                #with open('vec_tree.pkl', 'rb') as f:
#                #    haha = pickle.load(f)
#                logs.log('相似性计算结束后，corpus的所有句子的最终向量表示(向量树)存储的位置是为%s'%path) 
###################################################################################################################
###################################################################################################################    
###################################################################################################################
###################################################################################################################    
##########下面的是以前的做法。一个句子一个句子的计算，就存在很大的问题。改为上面用网络去计算。
#                step=0
#                while step < len(corpus):
#                    print("step:",step)
#                    sentence = corpus[step]#取出的是单词的索引编号，而且下标是从1开始计数的。
#                    treeConstructionOrders=corpus_fixed_tree_constructionorder[step]#取出抽象语法树的构建过程
#                    (_,node_vectors,root_vector,_)=self.computelossAndVector_no_tensor_withAST(sentence,treeConstructionOrders)#这里面每一次都会再定义一些网络的结构。所以50次，就要调用最外层的，即重新add_model_vars，和重新获取计算图，避免计算图中的网络结构过大。
#                    features[:,step,0]=root_vector[:,0]
#                    sum_vector=np.zeros((self.config.embed_size,1),dtype=np.float64)
#                    num_nodes=0
#                    for (_,node_vector) in node_vectors.items():
#                        sum_vector=node_vector+sum_vector
#                        num_nodes+=1
#                        pass
#                    mean_node_vector=sum_vector/num_nodes
#                    features[:,step,1]=mean_node_vector[:,0]
#                    step+=1
#                    pass
#                pass
#        features_for_similarity_computation=features[:,:,0]#暂时用顶层节点的向量进行度量。
##        features_for_similarity_computation=features[:,:,1]#用所有节点的平均向量进行度量。
#        features_for_similarity_computation=np.transpose(features_for_similarity_computation)#为什么转置，就是因为pdist是分析行向量之间的相似性。
#        np.save('xiaojie_functions_vectors_dataset.npy',features_for_similarity_computation)
#######################################
########以前两两比较计算相似性
#        sq_dists = pdist(features_for_similarity_computation, metric='euclidean')
#        z=squareform(sq_dists) #将sq_dists转换为对角矩阵。
#        #由于pdist函数只能度量矩阵的行向量之间的距离，我们需要对矩阵进行转置。
#        #####将矩阵写入文件
#        np.savetxt('./3RvNNoutData/similarity.csv', z,delimiter=',')
#        logs.log('结束对语料库计算句与句的相似性') 
#        return z
#######################################
    
def test_RNN():
    
    """Test RNN model implementation.

    该程序处理的语料库必须不能有一个单词的句子。如果有这种句子，会导致程序出现不可知的异常。因为程序中写死对这种情况无法处理。
    """
    input("开始？")
    logs.log("------------------------------\n程序开始")
    config = Config()
    model = RNN_Model(config)#构建树，并且构建词典
    
    
###############################################################################################################
##在语料库上训练深度神经网络
#    logs.log("(3)>>>>  开始训练RvNN")
#    start_time = time.time()
#    #训练模型
#    
#    stats = model.train(restore=False)
##    stats = model.train(restore=True)
#    logs.log('Training time: {}'.format(time.time() - start_time))
#    logs.log("(3)<<<<  训练RvNN结束")
#    plt.plot(stats['complete_loss_history'])
#    plt.title('complete_loss_history')
#    plt.xlabel('Iteration')
#    plt.ylabel('Loss')
#    plt.savefig("complete_loss_history.png")
#    plt.show()
#
#    
#    plt.plot(stats['evalution_loss_history']) #同前面的不同，这个只有重构误差。不包括参数正则化的损失。
#    plt.title('evalution_loss_history')
#    plt.xlabel('Iteration')
#    plt.ylabel('evalution_loss')
#    plt.savefig("evalution_loss_history.png")
#    plt.show()
################################################################################################################
################################################################################################################
####计算语料库中样本的相似性
    best_weights_path='./weights/%s'%model.config.model_name
#    model.similarities(corpus=model.evalutionCorpus,weights_path=best_weights_path)
    model.similarities(corpus=model.fullCorpus,corpus_sentence_length=model.fullCorpus_sentence_length,weights_path=best_weights_path,corpus_fixed_tree_constructionorder=model.full_corpus_fixed_tree_constructionorder)
################################################################################################################


#    print ('Test')
#    print ('=-=-=')
#    #如果不训练模型的话，就直接用下面的三行代码，将最后训练的模型，永久保存下来。即去掉temp。
##    shutil.copyfile('./weights/%s.temp.data-00000-of-00001'%model.config.model_name, './weights/%s.data-00000-of-00001'%model.config.model_name)
##    shutil.copyfile('./weights/%s.temp.index'%model.config.model_name, './weights/%s.index'%model.config.model_name)
##    shutil.copyfile('./weights/%s.temp.meta'%model.config.model_name, './weights/%s.meta'%model.config.model_name) 
#    predictions, _ = model.predict(model.test_data, './weights/%s'%model.config.model_name)
#    labels = [t.root.label for t in model.test_data]
#    test_acc = np.equal(predictions, labels).mean()
#    print ('Test acc: {}'.format(test_acc))
    logs.log("程序结束\n------------------------------")

from train_weighted_RAE_configuration_diffW import configuration
def xiaojie_RNN_1():#实验1 #主要目标就是完成在BigCloneBench标签数据集上的训练以及检测
################################################################################################################
################################################################################################################

    
    ########第一步：为模型加载输入样本集合，并配置参数。
    logs.log("------------------------------\n为模型加载训练样本集合，并配置参数")
    #用字典来配置参数。
    configuration_dict=configuration
    print(configuration_dict)  
    config = Config(configuration_dict)
    model = RNN_Model(config,experimentID=1)#构建树，并且构建词典
###############################################################################################################
#在语料库上训练深度神经网络
#########训练在另外一个程序。我们这个程序时使用模型。
################################################################################################################
    
################################################################################################################
################################################################################################################
################################################################################################################
    #使用随机化的模型去评测。
    model.using_model_for_BigCloneBench_experimentID_1()

    
################################################################################################################

def xiaojie_RNN_2():#实验2 配合树卷积，拿到向量树。路哥提供id号，我返回向量树。
    #主要目标就是完成在BigCloneBench标签数据集上的训练
################################################################################################################
################################################################################################################

    
    ########第一步：为模型加载输入样本集合，并配置参数。
    logs.log("------------------------------\n为模型加载训练样本集合，并配置参数")
    #用字典来配置参数。
    configuration_dict={}
    configuration_dict['label_size']=2
    configuration_dict['early_stopping']=2 #如果在评估集上训练了10次，都没有小于之前在评估集上计算的损失值，那么就终止训练
    configuration_dict['max_epochs']=30
    configuration_dict['anneal_threshold']=0.99#损失值一直不怎么降低的时候，我们就降低学习率
    configuration_dict['anneal_by']=1.5
    configuration_dict['lr']=0.01
    configuration_dict['l2']=0.02
    configuration_dict['embed_size']=300
    configuration_dict['model_name']='experimentID_1_rnn_embed=%d_l2=%f_lr=%f.weights'%(configuration_dict['embed_size'], configuration_dict['lr'], configuration_dict['l2'])
    configuration_dict['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/'
    configuration_dict['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/3RvNNoutData/'
    configuration_dict['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
    ######由于要在BigCloneBench上进行训练。我们不再设置最大的截取长度。以前设置为200。
    configuration_dict['MAX_SENTENCE_LENGTH']=1000000#这个是设置的句子的最大长度，用于从中筛选训练语料库。但是训练用语料库train_corpus的最大长度其实是保存在self.max_sentence_length_train_Corpusz中，这个是用于构建网络的
#    configuration_dict['batch_size']=100 #训练模型时用
    configuration_dict['batch_size']=10 #训练模型时用 设置成为100以后，训练非常缓慢
    configuration_dict['batch_size_using_model_notTrain']=300 #使用模型时用
    configuration_dict['MAX_SENTENCE_LENGTH_for_Bigclonebench']=600 #训练模型时用 
    
    
    config = Config(configuration_dict)
    model = RNN_Model(config,experimentID=2)#构建树，并且构建词典
###############################################################################################################
#在语料库上训练深度神经网络
#########训练在另外一个程序。我们这个程序时使用模型。
################################################################################################################
    
################################################################################################################
################################################################################################################
################################################################################################################
    #使用随机化的模型去评测。
    model.using_model_for_BigCloneBench_experimentID_2()

def xiaojie_RNN_3():#实验3。盲测。即在未知标记数据集的情况下，直接面向Bcd_reduced的源码库进行检测。
    #主要目标就是完成在BigCloneBench标签数据集上的训练
################################################################################################################
################################################################################################################

    
    ########第一步：为模型加载输入样本集合，并配置参数。
    logs.log("------------------------------\n为模型加载训练样本集合，并配置参数")
    #用字典来配置参数。
    configuration_dict={}
    configuration_dict['label_size']=2
    configuration_dict['early_stopping']=2 #如果在评估集上训练了10次，都没有小于之前在评估集上计算的损失值，那么就终止训练
    configuration_dict['max_epochs']=30
    configuration_dict['anneal_threshold']=0.99#损失值一直不怎么降低的时候，我们就降低学习率
    configuration_dict['anneal_by']=1.5
    configuration_dict['lr']=0.01
    configuration_dict['l2']=0.02
    configuration_dict['embed_size']=296
    configuration_dict['model_name']='weighted_RAE_rnn_embed=%d_l2=%f_lr=%f.weights'%(configuration_dict['embed_size'], configuration_dict['lr'], configuration_dict['l2'])
    configuration_dict['IDIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/'
    configuration_dict['ODIR']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/3RvNNoutData/'
    configuration_dict['corpus_fixed_tree_constructionorder_file']='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
    ######由于要在BigCloneBench上进行训练。我们不再设置最大的截取长度。以前设置为200。
    configuration_dict['MAX_SENTENCE_LENGTH']=1000000#这个是设置的句子的最大长度，用于从中筛选训练语料库。但是训练用语料库train_corpus的最大长度其实是保存在self.max_sentence_length_train_Corpusz中，这个是用于构建网络的
#    configuration_dict['batch_size']=100 #训练模型时用
    configuration_dict['batch_size']=10 #训练模型时用 设置成为100以后，训练非常缓慢
    configuration_dict['batch_size_using_model_notTrain']=400 #使用模型时用，批规模
    configuration_dict['MAX_SENTENCE_LENGTH_for_Bigclonebench']=300 #使用模型和训练模型时用。并且长于该长度的句子，不作为训练语料。使用模型时，长于该长度的句子，可以继续计算。
    configuration_dict['corpus_fixed_tree_construction_parentType_weight_file']='./1corpusData/corpus_bcb_reduced.method.AstConstructionParentTypeWeight.txt'
  
    
    config = Config(configuration_dict)
    model = RNN_Model(config,experimentID=3)#构建树，并且构建词典
###############################################################################################################
#在语料库上训练深度神经网络
#########训练在另外一个程序。我们这个程序时使用模型。
################################################################################################################
    
################################################################################################################
################################################################################################################
################################################################################################################
    #使用随机化的模型去评测。
    model.using_model_for_BigCloneBench_experimentID_3()





def verification_corpus():
    ######校验数据文件，即corpus文件，astconstruction文件以及writepath文件等，是否一致。
    
    def linesOfFile(filepath):
        count = 0
        with open(filepath,'r') as fw:
            for i, item in enumerate(fw, start=1):  # MATLAB is 1-indexed
                count+=1
        return count
    filepath1='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/2word2vecOutData/corpus.int'
    filepath2='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.txt'
    filepath3='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/corpus_bcb_reduced.method.AstConstruction.txt'
    filepath4='G:/code_of_zengjie/DeepLearningCodeOfXiaoJie/Recursive_autoencoder_xiaojie/1corpusData/writerpath_bcb_reduced.method.txt'
    lines1=linesOfFile(filepath1)
    lines2=linesOfFile(filepath2)
    lines3=linesOfFile(filepath3)
    lines4=linesOfFile(filepath4)
    print(lines1,lines2,lines3,lines4)
    #验证抽象语法树构建长度+1与语料库中句子长度的是否一致。
    with open(filepath1,'r') as f1:
        with open(filepath2,'r') as f2:
            with open(filepath3,'r') as f3:
                with open(filepath4,'r') as f4:
                    for i in range(lines1):
                        line1=f1.readline()
                        line2=f2.readline()
                        line3=f3.readline()
                        line4=f4.readline()
                        words1= line1.strip().split()
                        words2= line2.strip().split()
                        length3=line3.strip('\n').strip(' ').split(' ')
                        if((len(words1))!=(len(words2))):
                            print(line4)
                            print('在corpus.int中的长度{}，同在txt中的长度{}不一致。'.format(len(words1),len(words2)))
                            input()
                            return 
                        if((len(words1))!=(1+(len(length3)))):
                            print(line4)
                            print('句子单词长度{}，不等于构建次数{}+1，'.format(len(words1),len(length3)))
                            input()
                            return 
                        pass
    print('校验完毕，没发现问题')
    return     
################################################################################################################
################################################################################################################
def save_to_pkl(python_content, pickle_name):
    with open(pickle_name, 'wb') as pickle_f:
        pickle.dump(python_content, pickle_f)
def read_from_pkl(pickle_name):
    with open(pickle_name, 'rb') as pickle_f:
        python_content = pickle.load(pickle_f)
    return python_content   

if __name__ == "__main__":
    xiaojie_RNN_1()