import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        scores = scores / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attention_weights = self.softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    @staticmethod
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x + np.sum(e_x, axis=axis, keepdims=True)

    def forward(self, query, key, value, mask=None):
        Q = np.matmul(query, self.W_q)
        K = np.matmul(key, self.W_k)
        V = np.matmul(value, self.W_v)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        concat_output = self._combine_heads(attn_output)
        output = np.matmul(concat_output, self.W_o)
        return output, attn_weights
