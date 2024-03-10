from keras.layers import Layer   # layer 类用于创建自定义层，而 backend 模块提供了对底层张量操作的访问
from keras import backend as K   # Keras 已经作为 TensorFlow 的一部分，被整合进了 TensorFlow 2.0 中

class selfAttention(Layer):    # 这段代码定义了一个名为 selfAttention 的自定义层，继承自 Keras 中的 Layer 类--好像不是标准的自注意力机制
    def __init__(self, n_head, hidden_dim, penalty=0.1, **kwargs):
        self.n_head = n_head  # 多头自注意力机制中的头数
        self.penalty = penalty  # 惩罚项，默认为 0.1
        
        self.hidden_dim = hidden_dim    # 隐藏层维度
        super(selfAttention, self).__init__(**kwargs)  # 通过继承，selfAttention类获得了Layer类的所有属性和方法，
        
    def get_config(self):  # 目的是确保在保存模型时可以将自定义层的配置信息保存下来，以便在加载模型时能够正确地重建自定义层。
        config = {
            'n_head': self.n_head,
            'penalty': self.penalty,
            'hidden_dim': self.hidden_dim,
        }  # config，包含了自注意力层的参数
        base_config = super(selfAttention, self).get_config()  # 来获取父类（Layer 类）的配置信息
        return dict(list(base_config.items()) + list(config.items()))  # 合成字典并返回
    
    def build(self, input_shape):    # 在 Keras 中，build 方法用于创建层的权重，该方法在模型第一次使用时被调用，而且必须在调用层的 call 方法之前执行
        self.W1 = self.add_weight(name='w1', shape=(input_shape[2], self.hidden_dim), initializer='uniform',
                                  trainable=True)
        # 创建一个名为 W1 的权重，其形状为 (输入特征维度, 隐藏层维度)，并使用均匀分布的初始化器随机初始化权重的值，trainable=True: 布尔值，用于指定权重是否可训练。本例在模型的训练过程中被更新
        self.W2 = self.add_weight(name='W2', shape=(self.hidden_dim, self.n_head), initializer='uniform',
                                  trainable=True)
        super(selfAttention, self).build(input_shape) # 调用父类的 build 方法，完成层的构建过程
    def call(self, x, **kwargs): # 自注意力层的前向传播逻辑
        d1 = K.dot(x, self.W1)
        tanh1 = K.tanh(d1)
        d2 = K.dot(tanh1, self.W2)
        softmax1 = K.softmax(d2, axis=0)    # 这个操作在轴0上进行，即对于每个样本进行 softmax
        A = K.permute_dimensions(softmax1, (0, 2, 1))  # 对注意力权重进行维度置换，将注意力权重的最后两个维度进行置换，以便后续计算。
        emb_mat = K.batch_dot(A, x, axes=[2, 1])  # 使用注意力权重加权输入张量 x，得到加权后的张量 emb_mat。
        reshape = K.batch_flatten(emb_mat)    # 将加权后的张量 emb_mat 展平成一维张量
        eye = K.eye(self.n_head) # : 创建一个单位矩阵，用于计算正则化项
        prod = K.batch_dot(softmax1, A, axes=[1, 2])
        # axes 是一个长度为 2 的元组 (ax1, ax2)，则表示在 x 的第 ax1 轴和 y 的第 ax2 轴上执行点积。这种情况下，输出的形状是 (batch_size, x.shape[:ax1], y.shape[ax2:])。
        self.add_loss(self.penalty * K.sqrt(K.sum(K.square(prod - eye))))
        return reshape
    
    def compute_output_shape(self, input_shape): # 根据输入形状来计算并返回自定义层的输出形状。
        return (input_shape[0], input_shape[-1] * self.n_head,)
    
    @staticmethod
    def get_custom_objects(): # get_custom_objects 方法用于返回自定义层的名称和相应的 Python 类的映射字典，这样在加载模型时可以正确地识别和重建自定义层。
        return {'selfAttention': selfAttention}
