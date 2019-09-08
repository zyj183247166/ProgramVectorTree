# Code Clone Detection Based on Program Vector Tree

We use the Program Vector Tree to represent each function in  the target software system, and then use our porposed weigthed encoding mechanism to encode the tree into a fixed-sized vector, and at last use the Euclidean Distance and the set threshold to detect clone Pair. The experiment is different with the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie which train a deep neural network to learn out a classification model. The experiment here just use word2vec to learn word embeddings of program words and then use the porposed weigthed encoding mechanism to encode the program vector tree. Although no special deep neural network is trained here, for ease of calculation, we still use Tensoflow to construct a computation model and utilize the Gpu to perform computation. Because, there are more than 720000 functions which are compuation complex.  Besides, we perform programming here based on our previous code there. So some codes maybe the same as the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie, but the core theories are totally different.

Every question if you encounter, you can directyly concat my email zyj183247166@qq.com.
Because the GitHub has storage space constraints, the data after the step 1-3 below are shared at the Baidu Netdisk with the link and password as: 
LINK：https://pan.baidu.com/s/1zaX1YMLmLsr3EKXK8nrOjA PASSWORD：tzib
And, the database of BigCloneBench is processed and the data about the TrueClonePairs, FalseClonePairs and CloneTypoes are shared at the Baidu Netdisk with the link and password as:
LINK：https://pan.baidu.com/s/1S7iQnsjJgHnsh5NzHc196Q PASSWORD：whtc

After download the two dataset in the above links, there are two folders named as "1corpusData" and "SplitDataSet". Please copy them to the folder "CodeCloneDetection_ProgramVectorTree" as two subfolders. Otherwise, some programs cannot be runned correctly without these data.

# 0 Acknowledgement

I very appreciate Martin White,  Cong Fu and Jeffrey Svajlenko for offering much help about ast2bin, NSG and  BigCloneBench respectively. I email a lot to them for some help. 
ast2bin:[https://github.com/micheletufano/ast2bin](https://github.com/micheletufano/ast2bin)
NSG:https://github.com/ZJULearning/nsg
BigCloneBench:[https://github.com/jeffsvajlenko/BigCloneEval](https://github.com/jeffsvajlenko/BigCloneEval)

# 1 Hardware and software environment configuration
    the same as the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie
    
# 2 Steps of experiment
Step 1 and 2 here are the same with the step 1 and step 2 in  the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie
Step 3 here is the same with the Step 4 in the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie
Step 4 here is the same with the Step 5 in the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie
And, the work folder is changed from "Recursive_autoencoder_xiaojie" to "CodeCloneDetection_ProgramVectorTree"
## Step 1: process the ASTs of programs, get full binary trees and corpus of sentences at function granularity
## Step 2: copy the following files above to \1corpusData
corpus_bcb_reduced.method.txt
corpus_bcb_reduced.method.AstConstruction
notProcessedToCorpus_files_bcb_reduced.method
## Step 3: training word2vec on the corpus.
## Step 4: process word2vec.Out and the corpus file.
## Step 5: encode each program vector tree of each function into a fixed-sized vector (the dimension is 296).
1. Start terminal from anaconda
2. run the command 
    > cd \CodeCloneDetection_ProgramVectorTree\
    > python ./1_5_using_PVT_on_BigCloneBench.py
3. the results calculated for each vector of each function are stored in:
 ./vector/diffweight/ZHIXIN_himpq_using_PVT_BigCloneBenchFunction_ID_Map_Vector_root.xiaojiepkl
'himpq' is a random name that changes with each execution.

## Step 6: compare it with unweiggted RAE on the BigCloneBench data set
I preprocessed BigCloneBench , mainly analyzes the CLONES table and FALSEPOSITIVES table in this database.The former stores positive clones, while the latter stores negative clones. Through the analysis , we found that 25 functions in BigCloneBench were incorrectly marked. See functions_with_wrong_location_bigclonebench.txt for details. In addition, we removed duplicate clone pairs marked by BigCloneBench. See Duplicate_clone_pair_record.txt file. Finally, the remaining BigCloneBench clone pairs (positive or negative labels) are stored into all_pairs_id_xiaoje.pkl. We store each clone pair's corresponding clone type in the all_clone_id_pair_clonetype_xiaojie.pkl file. Most importantly, the functions' Numbers marked in BigCloneBench are inconsistent with those in our corpus corpus_bcb_reduce.method.txt. We map the function Numbers in BigCloneBench to our function Numbers (lines in the txt) in corpus_bcb_reduce.method.txt and storing them in the all_idmapline_xiaojie.pkl file.
1. getting the indicators. run the 3_evaluate_on_BigCloneBench.py file. the evaluation results of this model set with different distance thresholds are saved into: 
'./result/diffweight/ZHIXIN_himpq_PVT_metrices_root.xiaojiepkl'
2. visualize the indicators of two models.
    Let's go ahead and run
    python 4_visualizeTheIndicatorsOnBigCloneBench.py

## Step 7: directly apply the model to bcd_reduced source code library
Generates a vector representation for each function in the bcd_reduced source library
run the below program.
python 1_6_using_PVT_on_bcd_reduced.py

As the program is run with too many parameters, it is impossible to save all the vectors of functions in one time. We will save multiple PKL files into the vector directory in batches, and the naming rule is

0_fullCorpusLine_PVT_mean
1_fullCorpusLine_PVT_mean
....

A series of PKL files are generated.We then run the file 6_mergepkl.py we then Merge all PKL files and generate one: using_PVT_fullCorpusLine_Map_Vector.xiaojiepkl and along with using_PVT_fullCorpusLine_Map_Vector.npy The row number of the matrix corresponds to the function number in the whole corpus. There are 785438 lines in total, and each line of vector represents the corresponding function.


## Step 8: use NSG algorithm for cloning detection and report the results, , evaluating its scalability and geting results.
Step 8 here are the same with the step 9 in  the experiment at https://github.com/zyj183247166/Recursive_autoencoder_xiaojie
