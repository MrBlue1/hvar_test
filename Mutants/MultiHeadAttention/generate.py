import os
import shutil
from collections import OrderedDict

# ==================== 原始代码 ====================
# ==================== 2. 原始代码（MultiHeadAttention） ====================
ORIGINAL_CODE = '''import numpy as np

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
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

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
'''

# ==================== 3. 变异体生成（至少90个，覆盖全部18种违规） ====================
def generate_mutants():
    mutants = []
    # ---------- ROR ----------
    ror_list = [
        ("assert d_model % num_heads == 0", "assert d_model % num_heads != 0"),
        ("assert d_model % num_heads == 0", "assert d_model % num_heads > 0"),
        ("assert d_model % num_heads == 0", "assert d_model % num_heads < 0"),
        ("assert d_model % num_heads == 0", "assert d_model % num_heads >= 0"),
        ("assert d_model % num_heads == 0", "assert d_model % num_heads <= 0"),
        ("mask == 0", "mask != 0"),
        ("mask == 0", "mask > 0"),
        ("mask == 0", "mask < 0"),
        ("mask == 0", "mask >= 0"),
        ("mask == 0", "mask <= 0"),
        ("if mask is not None:", "if mask is None:"),
        ("if mask is not None:", "if mask is not None and True:"),
        ("if mask is not None:", "if mask is not None or False:"),
        ("if mask is not None:", "if mask is not None and False:"),
        ("if mask is not None:", "if mask is not None or True:"),
    ]
    for old, new in ror_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'ROR'))
    # ---------- AOR ----------
    aor_list = [
        ("self.d_k = d_model // num_heads", "self.d_k = d_model / num_heads"),
        ("self.d_k = d_model // num_heads", "self.d_k = d_model * num_heads"),
        ("self.d_k = d_model // num_heads", "self.d_k = d_model + num_heads"),
        ("self.d_k = d_model // num_heads", "self.d_k = d_model - num_heads"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores * np.sqrt(self.d_k)"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores + np.sqrt(self.d_k)"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores - np.sqrt(self.d_k)"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores / (np.sqrt(self.d_k) + 1e-8)"),
        ("e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))", "e_x = np.exp(x + np.max(x, axis=axis, keepdims=True))"),
        ("e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))", "e_x = np.exp(x * np.max(x, axis=axis, keepdims=True))"),
        ("e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))", "e_x = np.exp(x / np.max(x, axis=axis, keepdims=True))"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return e_x * np.sum(e_x, axis=axis, keepdims=True)"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return e_x - np.sum(e_x, axis=axis, keepdims=True)"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return e_x + np.sum(e_x, axis=axis, keepdims=True)"),
        ("* 0.01", "/ 0.01"),
        ("* 0.01", "+ 0.01"),
        ("* 0.01", "- 0.01"),
        ("* 0.01", "* 100"),
        ("* 0.01", "* 1e-6"),
        ("* 0.01", "* 1e6"),
        # 触发 division_by_zero
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores / 0.0"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores / (self.d_k * 0)"),
        # 触发 d_k_zero
        ("self.d_k = d_model // num_heads", "self.d_k = 0"),
        # 触发 sqrt_negative
        ("np.sqrt(self.d_k)", "np.sqrt(-self.d_k)"),
        # 触发 attention_weights_all_zero
        ("attention_weights = self.softmax(scores, axis=-1)", "attention_weights = np.zeros_like(scores)"),
    ]
    for old, new in aor_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'AOR'))
    # ---------- LOR ----------
    lor_list = [
        ("if mask is not None:", "if not (mask is None):"),
        ("if mask is not None:", "if mask is None and True:"),
        ("if mask is not None:", "if mask is None or False:"),
        ("assert d_model % num_heads == 0", "assert (d_model % num_heads == 0) and True"),
        ("assert d_model % num_heads == 0", "assert (d_model % num_heads == 0) or False"),
        ("assert d_model % num_heads == 0", "assert (d_model % num_heads == 0) and (True or False)"),
        ("if mask is not None:", "if mask is not None and mask.any():"),
        ("if mask is not None:", "if mask is not None or mask.any():"),
    ]
    for old, new in lor_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'LOR'))
    # ---------- COR ----------
    cor_list = [
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask != 0, -1e9, scores)"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, scores, -1e9)"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, -1e9, scores + 1)"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, -1e6, scores)"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, -np.inf, scores)"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, 0, scores)"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, -1e12, scores)"),
        ("if mask is not None:", "if mask is not None and mask.shape[-1] > 0:"),
    ]
    for old, new in cor_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'COR'))
    # ---------- UOI ----------
    uoi_list = [
        ("scores = scores / np.sqrt(self.d_k)", "scores = -scores / np.sqrt(self.d_k)"),
        ("output = np.matmul(attention_weights, V)", "output = -np.matmul(attention_weights, V)"),
        ("e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))", "e_x = np.exp(-(x - np.max(x, axis=axis, keepdims=True)))"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return -e_x / np.sum(e_x, axis=axis, keepdims=True)"),
        ("Q = np.matmul(query, self.W_q)", "Q = -np.matmul(query, self.W_q)"),
        ("K = np.matmul(key, self.W_k)", "K = +np.matmul(key, self.W_k)"),
        ("V = np.matmul(value, self.W_v)", "V = ~np.matmul(value, self.W_v)"),
        ("attention_weights = self.softmax(scores, axis=-1)", "attention_weights = -self.softmax(scores, axis=-1)"),
        ("scores = np.matmul(Q, K.transpose(0, 1, 3, 2))", "scores = np.matmul(-Q, K.transpose(0, 1, 3, 2))"),
        ("scores = np.matmul(Q, K.transpose(0, 1, 3, 2))", "scores = np.matmul(Q, -K.transpose(0, 1, 3, 2))"),
    ]
    for old, new in uoi_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'UOI'))
    # ---------- SDL ----------
    sdl_lines = [
        "        self.W_q = np.random.randn(d_model, d_model) * 0.01",
        "        self.W_k = np.random.randn(d_model, d_model) * 0.01",
        "        self.W_v = np.random.randn(d_model, d_model) * 0.01",
        "        self.W_o = np.random.randn(d_model, d_model) * 0.01",
        "        scores = scores / np.sqrt(self.d_k)",
        "        attention_weights = self.softmax(scores, axis=-1)",
        "        Q = self._split_heads(Q)",
        "        K = self._split_heads(K)",
        "        V = self._split_heads(V)",
        "        concat_output = self._combine_heads(attn_output)",
        "        output = np.matmul(concat_output, self.W_o)",
        "        assert d_model % num_heads == 0, \"d_model 必须能被 num_heads 整除\"",
    ]
    for line in sdl_lines:
        if line in ORIGINAL_CODE:
            new_code = ORIGINAL_CODE.replace(line + '\n', '', 1)
            mutants.append((new_code, 'SDL'))
    # ---------- ABS ----------
    abs_list = [
        ("scores = scores / np.sqrt(self.d_k)", "scores = np.abs(scores) / np.sqrt(self.d_k)"),
        ("e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))", "e_x = np.exp(np.abs(x - np.max(x, axis=axis, keepdims=True)))"),
        ("output = np.matmul(attention_weights, V)", "output = np.abs(np.matmul(attention_weights, V))"),
        ("Q = np.matmul(query, self.W_q)", "Q = np.abs(np.matmul(query, self.W_q))"),
        ("K = np.matmul(key, self.W_k)", "K = np.abs(np.matmul(key, self.W_k))"),
        ("V = np.matmul(value, self.W_v)", "V = np.abs(np.matmul(value, self.W_v))"),
        ("output = np.matmul(concat_output, self.W_o)", "output = np.abs(np.matmul(concat_output, self.W_o))"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return np.abs(e_x / np.sum(e_x, axis=axis, keepdims=True))"),
        ("scores = np.matmul(Q, K.transpose(0, 1, 3, 2))", "scores = np.abs(np.matmul(Q, K.transpose(0, 1, 3, 2)))"),
        ("attention_weights = self.softmax(scores, axis=-1)", "attention_weights = np.abs(self.softmax(scores, axis=-1))"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = np.abs(scores) / np.sqrt(np.abs(self.d_k))"),
        ("self.d_k = d_model // num_heads", "self.d_k = np.abs(d_model // num_heads)"),
    ]
    for old, new in abs_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'ABS'))
    # ---------- EQUI ----------
    equi_list = [
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores * (1.0 / np.sqrt(self.d_k))"),
        ("return output, attention_weights", "return (output, attention_weights)"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return e_x / (np.sum(e_x, axis=axis, keepdims=True))"),
        ("x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)", "x = x.reshape((batch_size, seq_len, self.num_heads, self.d_k))"),
        ("x = x.transpose(0, 2, 1, 3)", "x = x.transpose((0, 2, 1, 3))"),
        ("scores = np.matmul(Q, K.transpose(0, 1, 3, 2))", "scores = np.einsum('bhqd,bhkd->bhqk', Q, K)"),
        ("np.sqrt(self.d_k)", "self.d_k ** 0.5"),
        ("self.W_q = np.random.randn(d_model, d_model) * 0.01", "self.W_q = np.random.normal(0, 0.01, (d_model, d_model))"),
        ("batch_size, seq_len, _ = x.shape", "batch_size, seq_len, __ = x.shape"),
        ("scores = scores / np.sqrt(self.d_k)", "scores = scores / np.sqrt(self.d_k + 1e-12)"),
        ("if mask is not None:", "if mask is not None and mask.size > 0:"),
        ("np.where(mask == 0, -1e9, scores)", "np.where(mask == 0, -float('inf'), scores)"),
        ("return e_x / np.sum(e_x, axis=axis, keepdims=True)", "return e_x / np.sum(e_x, axis=axis, keepdims=True)  # softmax"),
        ("self.d_k = d_model // num_heads", "self.d_k = int(d_model / num_heads)"),
        # 触发 log_zero_warning
        ("e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))", "e_x = np.exp(x)"),
    ]
    for old, new in equi_list:
        if old in ORIGINAL_CODE:
            mutants.append((ORIGINAL_CODE.replace(old, new, 1), 'EQUI'))
    # 额外的 softmax 等价变异
    original_softmax = '''    @staticmethod
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)'''
    mutant_softmax = '''    @staticmethod
    def softmax(x, axis=-1):
        max_x = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - max_x)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)'''
    if original_softmax in ORIGINAL_CODE:
        mutants.append((ORIGINAL_CODE.replace(original_softmax, mutant_softmax, 1), 'EQUI'))
    return mutants


# 生成并保存
if __name__ == "__main__":
    mutants = generate_mutants()
    
    # 去重
    unique = OrderedDict()
    for code, mtype in mutants:
        if code not in unique:
            unique[code] = mtype
    
    print(f"总变异体数量: {len(unique)}")
    
    # 创建目录
    if os.path.exists("mutants"):
        shutil.rmtree("mutants")
    os.makedirs("mutants")
    
    # 保存原始代码
    with open("mutants/M00.py", "w", encoding="utf-8") as f:
        f.write(ORIGINAL_CODE)
    
    # 保存变异体
    idx = 1
    mutant_files = []
    for code, mtype in unique.items():
        fname = f"M{idx:02d}.py"
        with open(os.path.join("mutants", fname), "w", encoding="utf-8") as f:
            f.write(code)
        mutant_files.append((fname, mtype))
        idx += 1
    
    # 生成索引
    with open("MutantsIndex.py", "w", encoding="utf-8") as f:
        f.write("OPERATOR_MAPPING = {\n")
        f.write("    'M00': 'ORIG',\n")
        for fname, mtype in mutant_files:
            f.write(f"    '{fname[:-3]}': '{mtype}',\n")
        f.write("}\n")
    
    print(f"已生成 {len(mutant_files)} 个变异体")