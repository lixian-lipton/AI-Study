import numpy as np
np.random.seed(114514)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # 计算注意力分数
    attention_scores = np.matmul(Q, K.transpose(-2, -1))  # Q 和 K 的点积
    d_k = K.shape[-1]  # Key 向量的维度
    attention_scores /= np.sqrt(d_k)  # 缩放

    # 应用 softmax 来计算注意力权重
    attention_weights = softmax(attention_scores)

    # 使用注意力权重加权 V
    output = np.matmul(attention_weights, V)

    return output, attention_weights

def softmax(x):
    """计算 Softmax，确保数值稳定性"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def multi_head_attention(embed_size, num_heads, input, mask=None):
    if embed_size % num_heads != 0:
        raise ValueError("embed_size必须能够被num_heads整除")

    depth = embed_size // num_heads

    # 随机初始化 Wq, Wk, Wv, Wo 矩阵
    Wq = np.random.rand(embed_size, embed_size)
    Wk = np.random.rand(embed_size, embed_size)
    Wv = np.random.rand(embed_size, embed_size)
    Wo = np.random.rand(embed_size, embed_size)

    # 对输入做线性变换          input: (batch_size, seq_len, embed_size)
    Q = np.dot(input, Wq)  # (batch_size, seq_len, embed_size)
    K = np.dot(input, Wk)  # (batch_size, seq_len, embed_size)
    V = np.dot(input, Wv)  # (batch_size, seq_len, embed_size)

    # 分割成多个头
    Q = Q.reshape(Q.shape[0], Q.shape[1], num_heads, depth).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)
    K = K.reshape(K.shape[0], K.shape[1], num_heads, depth).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)
    V = V.reshape(V.shape[0], V.shape[1], num_heads, depth).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    # 计算缩放点积注意力
    attn_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

    # 连接多个头的输出
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(attn_output.shape[0], attn_output.shape[2], -1)  # (batch_size, seq_len, embed_size)

    # 最后通过 Wo 矩阵线性变换
    output = np.dot(attn_output, Wo)

    return output, attention_weights

# 测试示例
embed_size = 128
num_heads = 8
input = np.random.randn(10, 20, embed_size)  # (batch_size, seq_len, embed_size)
output, weights = multi_head_attention(embed_size, num_heads, input)

print(output.shape, weights.shape)  # 输出形状
print(output[0][0][:10])  # 输出第一个样本的前10个元素
print(weights[0][0][0][:10])  # 输出第一个样本第一个头的前10个注意力权重