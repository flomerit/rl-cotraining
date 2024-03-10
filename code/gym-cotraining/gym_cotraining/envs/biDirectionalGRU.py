#!/usr/bin/python
# -*- coding: utf-8 -*-
import keras
from selfAttention import selfAttention

class biDirectionalGRU():

    def __init__(self, vocab_size, EMBEDDING_DIM, num_units, embedding_matrix): # 网络初始定义

        self.model = keras.models.Sequential() # 创建了一个 Keras 序列模型 self.model
        self.model.add(keras.layers.Embedding(input_dim=vocab_size,
                                         output_dim=EMBEDDING_DIM,
                                         mask_zero=False,
                                        weights = [embedding_matrix],trainable = False))
        # 这段代码向模型中添加了一个嵌入层 (Embedding)，并且配置了该嵌入层的参数。
        # input_dim: 指定了词汇表的大小，即词汇量的数量。
        # output_dim: 指定了词嵌入的维度，即每个词汇的嵌入向量的大小。
        # mask_zero: 指定了是否将输入中的零用于掩蔽，如果设置为 True，则表示输入序列中的零将被忽略，不参与计算。
        # weights: 指定了词嵌入矩阵的初始权重。在这里，使用了预训练的词嵌入矩阵 embedding_matrix 作为初始权重。这可以用于使用预训练的词嵌入进行初始化。
        # trainable: 指定了是否训练嵌入层的权重。在这里，设置为 False，表示嵌入层的权重是固定的，不会在训练过程中更新。这通常用于在使用预训练的词嵌入时保持其不变。
        self.model.add(keras.layers.Bidirectional(keras.layers.GRU(units=num_units,
                                                               return_sequences=True)))
        # 这段代码向模型中添加了一个双向 GRU 层，并且配置了该 GRU 层的参数。
        # units: 指定了 GRU 单元的数量，即隐藏状态的维度。
        # return_sequences: 指定了是否返回完整的输出序列。如果设置为 True，则表示输出完整的序列，即输出的形状将为 (batch_size, sequence_length, num_units)，其中 batch_size 是批量大小，sequence_length 是序列长度，num_units 是隐藏状态的维度。
        self.model.add(selfAttention(n_head = 1 , hidden_dim=num_units)) # 添加了一个自注意力层 (selfAttention)
        self.model.add(keras.layers.Dense(units=2,activation='softmax')) 
        # 这段代码向模型中添加了一个全连接层 (Dense)，并且配置了该全连接层的参数。
        # units: 指定了全连接层的输出维度，即神经元的数量。在这个例子中，设置为 2，表示输出为一个包含两个神经元的向量。
        # activation: 指定了激活函数。在这个例子中，使用了 softmax 激活函数，用于进行多分类任务
        # 配置了模型的优化器、损失函数和评估指标
        self.model.compile(
            optimizer='adam', # 使用 Adam 优化器，Adam 是一种常用的自适应学习率优化算法
            loss='binary_crossentropy', # 用二元交叉熵损失函数，适用于二分类问题
            metrics=['accuracy'], # 使用准确率作为模型的评估指标
        )
        print self.model.summary() # self.model.summary() 打印模型的整体结构以及每一层的详细信息，有助于进行模型的调试和优化

    def fit(self,X,y,X_val,y_val):  # 训练网络
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,min_lr=0.0001)
        # ReduceLROnPlateau 回调函数，它是一个在训练过程中动态调整学习率的回调函数
        # monitor='val_loss': 监视的指标，即在哪个指标上进行监控。在这里是验证集上的损失。
        # factor=0.5: 学习率缩放因子，当触发学习率更新时，将当前学习率乘以该因子。在这里设置为 0.5，表示每次更新时将学习率缩小一半。
        # patience=3: 忍耐期，即验证损失在多少个 epoch 内没有改善时触发学习率更新。在这里设置为 3，表示如果验证损失在连续 3 个 epoch 内没有减小，则触发学习率更新。
        # min_lr=0.0001: 学习率的下限，即学习率最小值。当学习率减小到该值时，将不再更新学习率。在这里设置为 0.0001。

        file_path = 'best_model_title.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', 
            save_best_only=True)
        # ModelCheckpoint 回调函数，用于在训练过程中保存最佳模型的权重。
        # filepath=file_path: 指定了保存最佳模型权重的文件路径，即模型权重将保存在名为 file_path 的文件中。
        # monitor='val_loss': 监视的指标，即在哪个指标上进行监控。在这里是验证集上的损失。
        # save_best_only=True: 设置为 True，表示只保存在验证集上损失最小的模型权重。如果设置为 False，则会保存每个 epoch 结束时的模型权重

        callbacks = [reduce_lr,model_checkpoint]
        self.model.fit(X,y,validation_data=(X_val,y_val),epochs=10,callbacks = callbacks)
        # 调用 self.model.fit() 方法来进行模型的训练。其中 X 和 y 是训练集的特征和标签，X_val 和 y_val 是验证集的特征和标签。
        # validation_data=(X_val, y_val) 参数指定了在训练过程中使用的验证集。epochs=10 指定了训练的轮数，即遍历整个训练集的次数。
        # callbacks=callbacks 参数指定了使用的回调函数列表，包括了动态调整学习率和保存最佳模型。

    def predict(self, X):
        self.model = keras.models.load_model('best_model_title.hdf5',custom_objects=selfAttention.get_custom_objects()) # 加载之前保存的最佳模型权重，指定了自定义层的名称和类的映射字典，以便正确地加载自定义层。
        return self.model.predict(X) # 使用加载的模型对输入数据 X 进行预测，返回预测结果
    
    def evaluate(self,X,y):
        return self.model.evaluate(X,y)[1] # 使用 self.model.evaluate(X, y) 函数对输入数据 X 和标签 y 进行评估。该函数返回一个包含损失值和指定的评估指标（在此处是准确率）的列表。
        # 由于 evaluate 方法返回的列表中的第二个元素是准确率，因此使用索引 [1] 来提取准确率



