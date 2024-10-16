import numpy as np

def softmax(x):
    """计算 Softmax"""
    exp_x = np.exp(x - np.max(x))  # 数值稳定性
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """计算缩放点积注意力"""
    attention_scores = Q @ K.T  # 计算 Q 和 K 的点积
    d_k = K.shape[-1]  # Key 向量的维度
    attention_scores /= np.sqrt(d_k)  # 缩放

    attention_weights = softmax(attention_scores)  # 计算注意力权重
    output = attention_weights @ V  # 用注意力权重对 V 进行加权求和

    return output, attention_weights

class MultiHeadAttention:
    """多头注意力机制"""
    def __init__(self, num_heads, embedding_dim):
        self.num_heads = num_heads  # 头的数量
        self.depth = embedding_dim // num_heads  # 每个头的维度

        # 初始化权重矩阵
        self.Wq = np.random.rand(embedding_dim, embedding_dim)  # Query 权重
        self.Wk = np.random.rand(embedding_dim, embedding_dim)  # Key 权重
        self.Wv = np.random.rand(embedding_dim, embedding_dim)  # Value 权重

        self.Wo = np.random.rand(embedding_dim, embedding_dim)  # 输出权重

    def split_heads(self, x):
        """将输入分割为多个头"""
        batch_size, seq_len, embedding_dim = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.depth)  # (batch_size, seq_len, num_heads, depth)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def call(self, inputs):
        """前向传播"""
        batch_size, seq_len, _ = inputs.shape

        # 生成 Q、K、V
        Q = inputs @ self.Wq
        K = inputs @ self.Wk
        V = inputs @ self.Wv

        # 分割为多个头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算注意力
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V)

        # 连接多个头的输出
        attention_output = attention_output.transpose(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, depth)
        attention_output = attention_output.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, embedding_dim)

        # 输出线性层
        output = attention_output @ self.Wo

        return output, attention_weights

class FeedForward:
    """前馈神经网络"""
    def __init__(self, embedding_dim, ff_dim):
        self.W1 = np.random.rand(embedding_dim, ff_dim)  # 第一层权重
        self.b1 = np.random.rand(ff_dim)  # 第一层偏置
        self.W2 = np.random.rand(ff_dim, embedding_dim)  # 第二层权重
        self.b2 = np.random.rand(embedding_dim)  # 第二层偏置

    def call(self, x):
        """前向传播"""
        x = np.maximum(0, x @ self.W1 + self.b1)  # ReLU 激活
        return x @ self.W2 + self.b2  # 输出层

class EncoderLayer:
    """编码器层"""
    def __init__(self, num_heads, embedding_dim, ff_dim):
        self.mha = MultiHeadAttention(num_heads, embedding_dim)  # 多头注意力
        self.ffn = FeedForward(embedding_dim, ff_dim)  # 前馈网络

    def call(self, x):
        """前向传播"""
        attn_output, _ = self.mha.call(x)  # 计算注意力
        x = x + attn_output  # 残差连接
        x = x + self.ffn.call(x)  # 残差连接
        return x

class DecoderLayer:
    """解码器层"""
    def __init__(self, num_heads, embedding_dim, ff_dim):
        self.mha1 = MultiHeadAttention(num_heads, embedding_dim)  # 自注意力
        self.mha2 = MultiHeadAttention(num_heads, embedding_dim)  # 编码器-解码器注意力
        self.ffn = FeedForward(embedding_dim, ff_dim)  # 前馈网络

    def call(self, x, encoder_output):
        """前向传播"""
        attn_output1, _ = self.mha1.call(x)  # 自注意力
        x = x + attn_output1  # 残差连接

        attn_output2, _ = self.mha2.call(encoder_output)  # 编码器-解码器注意力
        x = x + attn_output2  # 残差连接

        x = x + self.ffn.call(x)  # 残差连接
        return x

class Transformer:
    """简单 Transformer 模型"""
    def __init__(self, num_heads, embedding_dim, ff_dim, num_layers):
        self.encoder_layers = [EncoderLayer(num_heads, embedding_dim, ff_dim) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(num_heads, embedding_dim, ff_dim) for _ in range(num_layers)]

    def call(self, encoder_input, decoder_input):
        """前向传播"""
        # 编码器部分
        for layer in self.encoder_layers:
            encoder_input = layer.call(encoder_input)

        # 解码器部分
        for layer in self.decoder_layers:
            decoder_input = layer.call(decoder_input, encoder_input)

        return decoder_input

# 示例输入
batch_size = 2
seq_len = 3
embedding_dim = 8  # 确保可以被 num_heads 整除
num_heads = 2  # 注意力头的数量
ff_dim = 16  # 前馈网络的维度
num_layers = 2  # 层数

encoder_input = np.random.rand(batch_size, seq_len, embedding_dim)  # 编码器输入
decoder_input = np.random.rand(batch_size, seq_len, embedding_dim)  # 解码器输入

# 创建 Transformer 模型并进行前向传播
transformer = Transformer(num_heads, embedding_dim, ff_dim, num_layers)
output = transformer.call(encoder_input, decoder_input)

print("Transformer Output:")
print(output)