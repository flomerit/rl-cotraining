#!/usr/bin/python
# -*- coding: utf-8 -*-
import gym
from gym.utils import seeding
import tensorflow as tf
from biDirectionalGRU import biDirectionalGRU
from cnnClassifier import cnnClassifier
import numpy as np
import pandas as pd
import re
import json
import pickle as pkl
from keras.utils import to_categorical
import os
from gym import spaces
from sklearn.metrics import precision_recall_fscore_support

class CoTraining(gym.Env):

    def __init__(self):
        
        self.count = 0

        self.load_data_view1()      # 加载视图1和视图2的数据 

        self.episode = 0
        
        self.load_data_view2()
        
        self.model1 = None    # 用于视图1和视图2的训练
        self.model2 = None

        self.metrics_view1 = []    # 用于记录视图1和视图2的训练指标
        self.metrics_view2 = []
        
        self.rep = pkl.load(open('data/rep.pkl'))
        
        self.members = pkl.load(open('data/members.pkl'))
        
        self.view1_rep_X = self.view1_unlabelled[self.rep,:]  
        # 根据索引数组 self.rep 从数组 self.view1_unlabelled 中选择对应的行，并将结果赋值给 self.view1_rep_X。
        # 这里的 self.view1_unlabelled 可能是一个包含视图1的未标记数据的数组，而 self.rep 则是用于选择子集的索引数组
        self.view2_rep_X = self.view2_unlabelled[self.rep,:]
        
        self.num_units = 100  # 自注意力的隐含层数

        self.EMBEDDING_DIM = 100    # 嵌入层维度
        
        self.state = None

        self.seed()
        #self.reset()
        
        self.mode = 'train'
        self.action_space = spaces.Discrete(80)  # 智能体可以选择80种不同的动作之一来执行
        
    def toggleMode(self):
        if self.mode == 'train':
            self.mode = 'test'
        else:
            self.mode = 'train'

    def load_data_view1(self):
                
        self.view1_train_X = pkl.load(open('data/view1_train_X'))
        self.view1_train_y = pkl.load(open('data/view1_train_y'))
        
        self.view1_val_X = pkl.load(open('data/view1_val_X'))
        self.view1_val_y = pkl.load(open('data/view1_val_y'))
        
        self.view1_test_X = pkl.load(open('data/view1_test_X'))
        self.view1_test_y = pkl.load(open('data/view1_test_y'))
        
        self.view1_unlabelled = pkl.load(open('data/view1_unlabelled'))
        
        self.embedding_matrix1 = pkl.load(open('data/view1_embedding_matrix'))
        
        self.vocab_size1 = len(self.embedding_matrix1)
        
    def load_data_view2(self):
        
        self.view2_train_X = pkl.load(open('data/view2_train_X'))
        self.view2_train_y = pkl.load(open('data/view2_train_y'))
        
        self.view2_val_X = pkl.load(open('data/view2_val_X'))
        self.view2_val_y = pkl.load(open('data/view2_val_y'))
        
        self.view2_test_X = pkl.load(open('data/view2_test_X'))
        self.view2_test_y = pkl.load(open('data/view2_test_y'))
        
        self.view2_unlabelled = pkl.load(open('data/view2_unlabelled'))
        
        self.embedding_matrix2 = pkl.load(open('data/view2_embedding_matrix'))
        
        self.vocab_size2 = len(self.embedding_matrix2)
        
    
    def step(self, action):
        
        if self.mode == 'test':  # 测试

            reward1 = self.model1.evaluate(self.view1_test_X, self.view1_test_y)
            reward2 = self.model2.evaluate(self.view2_test_X, self.view2_test_y)
            
        else:  # 验证
            reward1 = self.model1.evaluate(self.view1_val_X, self.view1_val_y)
            reward2 = self.model2.evaluate(self.view2_val_X, self.view2_val_y)
            
        ## 这段代码实现了通过模型 self.model1 对未标记的视图1数据进行预测，然后将预测结果与视图2的未标记数据一起用于训练模型 self.model2。这样做的目的是利用视图1的预测信息来增强视图2的训练数据，从而改善模型的性能。
        labels1 = np.argmax(self.model1.predict(self.view1_unlabelled[self.members[action]]),axis=-1) 
        # 从未标记的视图1数据中，根据动作 action 使用索引数组 self.members[action] 选择出相应的数据样本。
        # 将选取的数据样本传递给模型 self.model1.predict 进行预测
        # 使用 np.argmax 函数在预测结果中沿着指定的轴（axis=-1，即沿着最后一个维度）找到概率最大的类别，返回其索引。
        self.view2_train_X = np.concatenate((self.view2_train_X,self.view2_unlabelled[self.members[action]]))
        # 使用了 NumPy 的 np.concatenate 函数将当前的训练数据 self.view2_train_X 与智能体选择的动作 action 对应的未标记数据样本 self.view2_unlabelled[self.members[action]] 沿着指定的轴进行连接，从而扩展了训练数据集。
        self.view2_train_y = np.concatenate((self.view2_train_y,to_categorical(labels1,num_classes=2)))
        # to_categorical的作用是将 labels1 中的每个整数标签转换为一个长度为 num_classes 的独热编码数组,扩展训练标签集
        self.model2 = cnnClassifier(self.vocab_size2, self.EMBEDDING_DIM, self.embedding_matrix2)

        self.model2.fit(self.view2_train_X, self.view2_train_y,self.view2_val_X,self.view2_val_y)

        
        ## 同理
        labels2 = np.argmax(self.model2.predict(self.view2_unlabelled[self.members[action]]),axis=-1)

        self.view1_train_X = np.concatenate((self.view1_train_X,self.view1_unlabelled[self.members[action]]))
        self.view1_train_y = np.concatenate((self.view1_train_y,to_categorical(labels2,num_classes=2)))
        
        self.model1 = biDirectionalGRU(self.vocab_size1, self.EMBEDDING_DIM, self.num_units, self.embedding_matrix1)
        self.model1.fit(self.view1_train_X, self.view1_train_y,self.view1_val_X,self.view1_val_y)


        if self.mode == 'test':
            reward1 = self.model1.evaluate(self.view1_test_X, self.view1_test_y) - reward1
            reward2 = self.model2.evaluate(self.view2_test_X, self.view2_test_y) - reward2
            
        else:
            reward1 = self.model1.evaluate(self.view1_val_X, self.view1_val_y) - reward1
            reward2 = self.model2.evaluate(self.view2_val_X, self.view2_val_y) - reward2

        if reward1 > 0 and reward2 > 0:
            reward = reward1 * reward2
        else:
            reward = 0
        
        tmp1 = self.model1.predict(self.view1_rep_X)  # 使用了 self.model1.predict 方法对视图1的表示数据 self.view1_rep_X 进行预测，得到预测结果 tmp1
        tmp2 = self.model2.predict(self.view2_rep_X)

        self.state = np.hstack((tmp1,tmp2)).reshape(-1)
        # np.hstack 函数要求参与水平堆叠的数组在除了沿着指定的轴（通常是第一个轴，即axis=1）之外的所有轴上具有相同的形状。
        # 当 .reshape 方法的参数中包含 -1 时，表示在该位置上自动计算维度大小，以使得重塑后的数组能够容纳原始数组的所有元素。
        info = {'reward1':reward1,
                'reward2':reward2,
                'action': action
                }
        
        done = False
        
        print info

        if reward == 0:
            self.count = self.count + 1
        
        if self.count > 5 or self.steps == 80:
            done = True
        
        if self.mode == 'test':
            v1 = np.argmax(self.model1.predict(self.view1_test_X), axis = -1)
            v2 = np.argmax(self.model2.predict(self.view2_test_X), axis = -1)
            m1 = precision_recall_fscore_support(np.argmax(self.view1_test_y,axis=-1), v1, average='macro')
            m2 = precision_recall_fscore_support(np.argmax(self.view2_test_y,axis=-1), v2, average='macro')

        else:
            v1 = np.argmax(self.model1.predict(self.view1_val_X), axis = -1) # np.argmax() 被用于获取模型对测试数据的预测结果中概率最大的类别的索引
            v2 = np.argmax(self.model2.predict(self.view2_val_X), axis = -1)
            m1 = precision_recall_fscore_support(np.argmax(self.view1_val_y,axis=-1), v1, average='macro')
            # precision_recall_fscore_support 是 sklearn 库中的一个函数，用于计算分类任务的 Precision（精确率）、Recall（召回率）和 F1-score。具体参数含义如下：
            # 第一个参数 y_true 是真实的类别标签；
            # 第二个参数 y_pred 是预测的类别标签；
            # average 参数用于指定如何计算多类别分类问题的指标。在这里，设定为 'macro' 表示对每个类别求指标的均值，不考虑类别不平衡的影响。
            m2 = precision_recall_fscore_support(np.argmax(self.view2_val_y,axis=-1), v2, average='macro')

        self.steps = self.steps+1

        self.metrics_view1[self.steps,:] = m1[:3] # 这行代码将视图1的性能指标（Precision、Recall、F1-score）保存在名为 self.metrics_view1 的数组中的第 self.steps 行
        self.metrics_view2[self.steps,:] = m2[:3]

        return (self.state, reward, done, info)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self): # 环境重置
        self.count = 0

        self.load_data_view1()       
        
        self.load_data_view2()

        if self.metrics_view1 != []:
            np.savetxt("results/metrics_view1_"+self.mode+"_"+str(self.episode)+".csv", self.metrics_view1, delimiter=",")
            np.savetxt("results/metrics_view2_"+self.mode+"_"+str(self.episode)+".csv", self.metrics_view2, delimiter=",")

        self.metrics_view1 = np.zeros((80,3))
        self.metrics_view2 = np.zeros((80,3))

        self.steps = 0

        self.model1 = biDirectionalGRU(self.vocab_size1, self.EMBEDDING_DIM, self.num_units, self.embedding_matrix1)
        self.model2 = cnnClassifier(self.vocab_size2, self.EMBEDDING_DIM, self.embedding_matrix2)

        self.model1.fit(self.view1_train_X, self.view1_train_y, self.view1_val_X, self.view1_val_y)
        self.model2.fit(self.view2_train_X, self.view2_train_y, self.view2_val_X, self.view2_val_y)

        tmp1 = self.model1.predict(self.view1_rep_X)
        tmp2 = self.model2.predict(self.view2_rep_X)

        self.state = np.hstack((tmp1,tmp2)).reshape(-1)

        self.done = False

        self.episode = self.episode + 1
        return self.state
