＃DSSM-
问答系统中的语义匹配
目前想到2个优化方向：
1、模型优化：
传统的Esim、Bimpm网络，都是输入两句话来预测它们是否匹配；但是我们线上场景在预测时，一般根据用户输入的一句query来查找知识库中最相似的问法，所以主要是获得query的embedding特征再做KNN检索。
调研以后发现有2种模型我们可以拿来和TinyBERT融合：
（1）Sentence-BERT，属于孪生网络和BERT的结合，训练时做2分类任务，预测时可以获得query的语义向量
https://github.com/UKPLab/sentence-transformers

（2）DSSM/CNN-DSSM/LSTM-DSSM，按字输入、用多个隐藏层来捕获特征，也是经典的匹配模型
https://www.cnblogs.com/guoyaohua/p/9229190.html

2、检索优化
之前线上用的检索模型是基于树的annoy，可以改成基于图结构的HNSW算法，之前和一凡实验过，内存消耗、预测速度都要优于annoy
https://github.com/nmslib/hnswlib

