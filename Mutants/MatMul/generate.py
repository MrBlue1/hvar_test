import os
import shutil
from pathlib import Path
import random

# 更新后的行为类型列表
all_behavior_types_new = [
    # ========== 输入/结构违规 ==========
    "dimension_mismatch",          # 0: 内部维度不匹配
    "empty_input",                 # 1: A 或 B 为空矩阵
    "irregular_rows",              # 2: 矩阵行长度不一致
    "non_numeric_input",           # 3: 矩阵元素包含非数值
    "scalar_input",                # 4: 输入为标量而非二维列表
    "wrong_iteration_bound",       # 5: 循环边界错误导致越界或遗漏

    # ========== 数值异常 ==========
    "nan_in_output",               # 6: 输出含有 NaN
    "inf_in_output",               # 7: 输出含有 Inf 或 -Inf
    "overflow_warning",            # 8: 触发溢出警告
    "underflow_to_zero",           # 9: 极小值下溢为 0.0
    "sign_error",                  # 10: 输出符号错误
    "precision_loss",              # 11: 大数加小数导致精度严重丢失

    # ========== 数学性质违规 ==========
    "incorrect_product_value",     # 12: 任意元素计算错误
    "wrong_output_shape",          # 13: 输出形状不是 (m x p)
    "identity_violation",          # 14: A * I != A 或 I * A != A
    "zero_property_violation",     # 15: 零矩阵乘任意矩阵结果非全零
    "transpose_property_violation",# 16: (A@B)^T != B^T @ A^T

    # ========== 性能/副作用违规 ==========
    "aliasing_side_effect",        # 17: 修改了输入矩阵的元素
]

def generate_oracle():
    """生成原始代码 M00.py"""
    oracle_code = '''def matmul(A, B):
    """
    矩阵乘法函数
    A: m x n 矩阵
    B: n x p 矩阵
    返回: m x p 矩阵 C = A @ B
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C
'''
    return oracle_code

def generate_mutants():
    """生成所有变异体"""
    mutants = []
    mutant_id = 1
    
    # 基础代码模板
    base_code = '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C
'''
    
    # ========== 1. ROR变异体 (关系运算符替换) ==========
    # 扩展更多range修改
    ror_mutants = [
        ("range(m) -> range(m-1)", "for i in range(m-1):", "i"),
        ("range(m) -> range(m+1)", "for i in range(m+1):", "i"),
        ("range(m) -> range(0)", "for i in range(0):", "i"),
        ("range(m) -> range(1, m)", "for i in range(1, m):", "i"),
        ("range(m) -> range(m, 0, -1)", "for i in range(m, 0, -1):", "i"),
        ("range(p) -> range(p-1)", "for j in range(p-1):", "j"),
        ("range(p) -> range(p+1)", "for j in range(p+1):", "j"),
        ("range(p) -> range(0)", "for j in range(0):", "j"),
        ("range(p) -> range(1, p)", "for j in range(1, p):", "j"),
        ("range(n) -> range(n-1)", "for k in range(n-1):", "k"),
        ("range(n) -> range(n+1)", "for k in range(n+1):", "k"),
        ("range(n) -> range(0)", "for k in range(0):", "k"),
        ("range(n) -> range(1, n)", "for k in range(1, n):", "k"),
        ("range(n) -> range(n, -1, -1)", "for k in range(n, -1, -1):", "k"),
    ]
    
    for desc, replacement, loop_var in ror_mutants:
        code = base_code
        if loop_var == "i":
            code = code.replace("for i in range(m):", replacement)
        elif loop_var == "j":
            code = code.replace("for j in range(p):", replacement)
        elif loop_var == "k":
            code = code.replace("for k in range(n):", replacement)
        mutants.append((f"M{mutant_id:02d}", "ROR", code, desc))
        mutant_id += 1
    
    # ========== 2. AOR变异体 (算术运算符替换) ==========
    # 扩展更多算术运算符替换
    aor_mutants = [
        # 基本算术运算替换
        ("+ -> -", "sum_val -= A[i][k] * B[k][j]"),
        ("+ -> *", "sum_val *= A[i][k] * B[k][j]"),
        ("+ -> /", "sum_val += A[i][k] / (B[k][j] if B[k][j] != 0 else 1)"),
        ("* -> +", "sum_val += A[i][k] + B[k][j]"),
        ("* -> -", "sum_val += A[i][k] - B[k][j]"),
        ("* -> /", "sum_val += A[i][k] / (B[k][j] if B[k][j] != 0 else 1)"),
        ("* -> //", "sum_val += A[i][k] // (B[k][j] if B[k][j] != 0 else 1)"),
        ("* -> **", "sum_val += A[i][k] ** B[k][j]"),
        # 初始化值修改
        ("= 0 -> = 1", "sum_val = 1"),
        ("= 0 -> = -1", "sum_val = -1"),
        ("= 0 -> = 0.5", "sum_val = 0.5"),
        ("[0] * p -> [1] * p", "C = [[1] * p for _ in range(m)]"),
        ("[0] * p -> [-1] * p", "C = [[-1] * p for _ in range(m)]"),
        # 复合赋值运算符
        ("+= -> -=", "sum_val -= A[i][k] * B[k][j]"),
        ("+= -> *=", "sum_val *= A[i][k] * B[k][j]"),
        ("+= -> /=", "sum_val /= (A[i][k] * B[k][j] if A[i][k] * B[k][j] != 0 else 1)"),
    ]
    
    for desc, replacement in aor_mutants:
        code = base_code
        if "+ ->" in desc or "-= ->" in desc or "*= ->" in desc or "/= ->" in desc:
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif "* ->" in desc:
            if "**" in replacement:
                code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
            else:
                code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif "= 0" in desc:
            code = code.replace("sum_val = 0", replacement)
        elif "[0] * p" in desc:
            code = code.replace("C = [[0] * p for _ in range(m)]", replacement)
        mutants.append((f"M{mutant_id:02d}", "AOR", code, desc))
        mutant_id += 1
    
    # 单独处理表达式修改（这部分之前有问题）
    expr_mutants = [
        ("A[i][k] * B[k][j] -> A[i][k] + B[k][j]", "sum_val += A[i][k] + B[k][j]"),
        ("A[i][k] * B[k][j] -> A[i][k] - B[k][j]", "sum_val += A[i][k] - B[k][j]"),
        ("A[i][k] * B[k][j] -> A[i][k] / (B[k][j] + 1e-10)", "sum_val += A[i][k] / (B[k][j] + 1e-10)"),
    ]
    
    for desc, replacement in expr_mutants:
        code = base_code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        mutants.append((f"M{mutant_id:02d}", "AOR", code, desc))
        mutant_id += 1
    
    # ========== 3. LOR变异体 (逻辑运算符替换) ==========
    lor_mutants = [
        ("添加 and 条件", 
         '''    if m > 0 and n > 0 and p > 0:
        C = [[0] * p for _ in range(m)]
    else:
        return []''',
         "添加维度检查 and"),
        ("添加 or 条件", 
         '''    if m > 0 or n > 0:
        C = [[0] * p for _ in range(m)]
    else:
        return []''',
         "添加维度检查 or"),
        ("and -> or", 
         '''    if m > 0 or n > 0 or p > 0:
        C = [[0] * p for _ in range(m)]
    else:
        return []''',
         "维度检查 and 变 or"),
        ("not 条件", 
         '''    if not (m > 0 and n > 0 and p > 0):
        return []
    C = [[0] * p for _ in range(m)]''',
         "添加 not 条件"),
        ("组合条件", 
         '''    if (m > 0 and n > 0) or p == 0:
        return []
    C = [[0] * p for _ in range(m)]''',
         "复杂逻辑条件"),
    ]
    
    for desc, replacement, full_desc in lor_mutants:
        code = base_code.replace("    C = [[0] * p for _ in range(m)]", replacement)
        mutants.append((f"M{mutant_id:02d}", "LOR", code, full_desc))
        mutant_id += 1
    
    # ========== 4. COR变异体 (条件运算符替换) ==========
    cor_mutants = [
        ("添加维度检查", 
         '''    if len(A[0]) != len(B):
        return None
    
    m = len(A)
    n = len(A[0])
    p = len(B[0])''',
         "添加维度检查 !="),
        ("!= -> ==", 
         '''    if len(A[0]) == len(B):
        return None
    
    m = len(A)
    n = len(A[0])
    p = len(B[0])''',
         "维度检查 != 变 =="),
        ("添加空检查", 
         '''    if not A or not B or not A[0] or not B[0]:
        return []
    
    m = len(A)
    n = len(A[0])
    p = len(B[0])''',
         "添加空输入检查"),
        ("> -> >=", 
         '''    if len(A[0]) >= len(B):
        return None
    
    m = len(A)
    n = len(A[0])
    p = len(B[0])''',
         "条件 > 变 >="),
        ("< -> <=", 
         '''    if len(A[0]) <= len(B):
        return None
    
    m = len(A)
    n = len(A[0])
    p = len(B[0])''',
         "条件 < 变 <="),
        ("添加类型检查", 
         '''    if not isinstance(A, list) or not isinstance(B, list):
        return None
    
    m = len(A)
    n = len(A[0])
    p = len(B[0])''',
         "添加类型检查"),
    ]
    
    for desc, check_code, full_desc in cor_mutants:
        code = base_code.replace("    m = len(A)\n    n = len(A[0])\n    p = len(B[0])", check_code)
        mutants.append((f"M{mutant_id:02d}", "COR", code, full_desc))
        mutant_id += 1
    
    # ========== 5. UOI变异体 (一元运算符插入) ==========
    uoi_mutants = [
        ("-i", "for i in range(-m, 0):", "i"),
        ("-j", "for j in range(-p, 0):", "j"),
        ("-k", "for k in range(-n, 0):", "k"),
        ("-A", "sum_val += -A[i][k] * B[k][j]", "A"),
        ("-B", "sum_val += A[i][k] * -B[k][j]", "B"),
        ("-sum", "C[i][j] = -sum_val", "sum"),
        ("not A", "if not A:\n        return []", "A"),
        ("not B", "if not B:\n        return []", "B"),
        ("~i", "for i in range(~m, 0):", "i"),
        ("+A", "sum_val += +A[i][k] * B[k][j]", "A"),
    ]
    
    for desc, replacement, target in uoi_mutants:
        code = base_code
        if target == "i":
            code = code.replace("for i in range(m):", replacement)
        elif target == "j":
            code = code.replace("for j in range(p):", replacement)
        elif target == "k":
            code = code.replace("for k in range(n):", replacement)
        elif target == "A" and "not" not in desc:
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif target == "B":
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif target == "sum":
            code = code.replace("C[i][j] = sum_val", replacement)
        elif target == "A" and "not" in desc:
            code = base_code.replace("    m = len(A)", replacement + "\n    m = len(A)")
        elif target == "B" and "not" in desc:
            code = base_code.replace("    m = len(A)", replacement + "\n    m = len(A)")
        mutants.append((f"M{mutant_id:02d}", "UOI", code, desc))
        mutant_id += 1
    
    # ========== 6. SDL变异体 (语句删除) ==========
    sdl_mutants = [
        ("删除 C 初始化", 
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # C = [[0] * p for _ in range(m)]  # 删除初始化
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "删除C初始化"),
        ("删除 sum_val 初始化",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            # sum_val = 0  # 删除初始化
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "删除sum_val初始化"),
        ("删除 return C",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    # return C  # 删除返回语句''',
         "删除return语句"),
        ("删除 C[i][j] 赋值",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            # C[i][j] = sum_val  # 删除赋值
            
    return C''',
         "删除C[i][j]赋值"),
        ("删除内层循环",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            # for k in range(n):  # 删除内层循环
            #     sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "删除内层循环"),
        ("删除维度获取",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    # p = len(B[0])  # 删除p的获取
    p = len(B)
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "删除p的正确获取"),
    ]
    
    for desc, code, full_desc in sdl_mutants:
        mutants.append((f"M{mutant_id:02d}", "SDL", code, full_desc))
        mutant_id += 1
    
    # ========== 7. ABS变异体 (绝对值相关) ==========
    abs_mutants = [
        ("abs(A)", "sum_val += abs(A[i][k]) * B[k][j]", "A取绝对值"),
        ("abs(B)", "sum_val += A[i][k] * abs(B[k][j])", "B取绝对值"),
        ("abs(A*B)", "sum_val += abs(A[i][k] * B[k][j])", "乘积取绝对值"),
        ("abs(sum_val)", "C[i][j] = abs(sum_val)", "sum取绝对值"),
        ("abs(C)", "return [[abs(x) for x in row] for row in C]", "结果取绝对值"),
        ("-abs(A)", "sum_val += -abs(A[i][k]) * B[k][j]", "A取负绝对值"),
        ("abs(-A)", "sum_val += abs(-A[i][k]) * B[k][j]", "负A取绝对值"),
        ("abs(A)+abs(B)", "sum_val += abs(A[i][k]) + abs(B[k][j])", "绝对值相加"),
    ]
    
    for desc, replacement, full_desc in abs_mutants:
        code = base_code
        if "A取绝对值" in full_desc or "A取负绝对值" in full_desc or "负A取绝对值" in full_desc:
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif "B取绝对值" in full_desc:
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif "乘积取绝对值" in full_desc:
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        elif "sum取绝对值" in full_desc:
            code = code.replace("C[i][j] = sum_val", replacement)
        elif "结果取绝对值" in full_desc:
            code = code.replace("return C", replacement)
        elif "绝对值相加" in full_desc:
            code = code.replace("sum_val += A[i][k] * B[k][j]", replacement)
        mutants.append((f"M{mutant_id:02d}", "ABS", code, full_desc))
        mutant_id += 1
    
    # ========== 8. EQUI等价变异体 ==========
    equi_mutants = [
        ("展开循环计算", 
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for k in range(n):
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
                
    return C''',
         "循环顺序调整"),
        ("使用列表推导",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
            
    return C''',
         "列表推导实现"),
        ("先转置B",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # 转置B
    B_T = [[B[i][j] for i in range(n)] for j in range(p)]
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B_T[j][k]
            C[i][j] = sum_val
            
    return C''',
         "转置B实现"),
        ("使用zip",
         '''def matmul(A, B):
    m = len(A)
    p = len(B[0])
    
    # 转置B
    B_T = list(zip(*B))
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            C[i][j] = sum(a * b for a, b in zip(A[i], B_T[j]))
            
    return C''',
         "使用zip实现"),
        ("使用enumerate",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i, row in enumerate(A):
        for j in range(p):
            for k in range(n):
                C[i][j] += row[k] * B[k][j]
                
    return C''',
         "使用enumerate"),
        ("提前计算B的列",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    # 预先提取B的列
    B_cols = [[B[k][j] for k in range(n)] for j in range(p)]
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B_cols[j][k] for k in range(n))
            
    return C''',
         "预先提取B列"),
    ]
    
    for desc, code, full_desc in equi_mutants:
        mutants.append((f"M{mutant_id:02d}", "EQUI", code, full_desc))
        mutant_id += 1
    
    # ========== 9. 针对未命中类型的专门变异体 ==========
    targeted_mutants = [
        # empty_input - 删除空输入检查
        ("empty_input_1", "COR", 
         '''def matmul(A, B):
    # 故意不检查空输入，可能导致空输入错误
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "删除所有输入检查，容易在空输入时出错"),
    
        # aliasing_side_effect - 修改输入矩阵
        ("aliasing_1", "SDL", 
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                A[i][k] = A[i][k] * 2  # 修改输入矩阵A
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "修改输入矩阵A的元素"),
        
        ("aliasing_2", "SDL", 
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                B[k][j] = B[k][j] * 2  # 修改输入矩阵B
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "修改输入矩阵B的元素"),
    
        # wrong_iteration_bound - 错误的循环边界
        ("wrong_bound_1", "ROR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m+2):  # 错误的边界，可能导致越界
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "外层循环边界过大"),
        
        ("wrong_bound_2", "ROR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p+2):  # 错误的边界
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "中层循环边界过大"),
    
        # incorrect_product_value - 错误的乘积计算
        ("wrong_product_1", "AOR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] + B[k][j]  # 使用加法代替乘法
            C[i][j] = sum_val
            
    return C''',
         "加法代替乘法"),
        
        ("wrong_product_2", "AOR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 1  # 错误的初始值
            for k in range(n):
                sum_val *= A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "错误初始值和累积方式"),
    
        # dimension_mismatch - 维度不匹配
        ("dimension_mismatch_1", "COR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    # 故意注释掉维度检查，允许维度不匹配
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "无维度检查，允许不匹配维度"),
        
        ("dimension_mismatch_2", "SDL",
         '''def matmul(A, B):
    m = len(A)
    n = len(B)  # 错误的维度获取
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j]
            C[i][j] = sum_val
            
    return C''',
         "错误获取n维度"),
    
        # overflow_warning 和 underflow_to_zero 可以通过已有的AOR变异体配合大数值触发
        # 再添加一些专门可能触发溢出/下溢的变异体
        
        ("overflow_prone", "AOR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += (A[i][k] ** 2) * (B[k][j] ** 2)  # 平方运算容易溢出
            C[i][j] = sum_val
            
    return C''',
         "使用平方运算，容易溢出"),
        
        ("underflow_prone", "AOR",
         '''def matmul(A, B):
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    C = [[0] * p for _ in range(m)]
    
    for i in range(m):
        for j in range(p):
            sum_val = 0
            for k in range(n):
                sum_val += A[i][k] * B[k][j] / 1e100  # 除以大数容易下溢
            C[i][j] = sum_val
            
    return C''',
         "除以大数，容易下溢"),
    ]
    
    for _, mut_type, code, desc in targeted_mutants:
        mutants.append((f"M{mutant_id:02d}", mut_type, code, desc))
        mutant_id += 1
    
    # 确保至少有60个变异体
    # 如果还不够，添加更多ROR变异体
    additional_ror = [
        ("range(m) -> range(m, 0, -1)", "for i in range(m, 0, -1):", "i", "反向遍历i"),
        ("range(p) -> range(p, 0, -1)", "for j in range(p, 0, -1):", "j", "反向遍历j"),
        ("range(n) -> range(n, 0, -1)", "for k in range(n, 0, -1):", "k", "反向遍历k"),
    ]
    
    while mutant_id <= 60:
        for desc, replacement, loop_var, full_desc in additional_ror:
            if mutant_id > 60:
                break
            code = base_code
            if loop_var == "i":
                code = code.replace("for i in range(m):", replacement)
            elif loop_var == "j":
                code = code.replace("for j in range(p):", replacement)
            elif loop_var == "k":
                code = code.replace("for k in range(n):", replacement)
            mutants.append((f"M{mutant_id:02d}", "ROR", code, full_desc))
            mutant_id += 1
    
    # 截取前65个（或实际生成的数量）
    return mutants[:max(65, len(mutants))]

def save_mutants(mutants_dir="mutants"):
    """保存所有变异体到文件"""
    # 创建目录
    Path(mutants_dir).mkdir(exist_ok=True)
    
    # 保存原始代码
    oracle_path = Path(mutants_dir) / "M00.py"
    with open(oracle_path, 'w', encoding='utf-8') as f:
        f.write(generate_oracle())
    print(f"已保存: {oracle_path}")
    
    # 保存变异体
    operator_mapping = {'M00': 'ORIG'}
    
    mutants_list = generate_mutants()
    print(f"\n共生成 {len(mutants_list)} 个变异体")
    
    for mutant_id, mut_type, code, desc in mutants_list:
        mutant_path = Path(mutants_dir) / f"{mutant_id}.py"
        with open(mutant_path, 'w', encoding='utf-8') as f:
            f.write(code)
        operator_mapping[mutant_id] = mut_type
        print(f"已保存: {mutant_id}.py ({mut_type}: {desc})")
    
    return operator_mapping

# 生成MutantsIndex.py
def generate_mutant_index(operator_mapping):
    """生成MutantsIndex.py文件"""
    index_content = "# MutantsIndex.py - 变异体索引文件\n"
    index_content += "# 记录每个变异体的变异类型\n\n"
    index_content += "OPERATOR_MAPPING = {\n"
    
    for mutant_id in sorted(operator_mapping.keys()):
        index_content += f"    '{mutant_id}': '{operator_mapping[mutant_id]}',\n"
    
    index_content += "}\n\n"
    index_content += "# 变异类型统计\n"
    index_content += "OPERATOR_STATS = {\n"
    
    type_counts = {}
    for mut_type in operator_mapping.values():
        type_counts[mut_type] = type_counts.get(mut_type, 0) + 1
    
    for mut_type, count in sorted(type_counts.items()):
        index_content += f"    '{mut_type}': {count},\n"
    
    index_content += "}\n"
    
    with open("MutantsIndex.py", 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"\n已生成 MutantsIndex.py")
    print("\n变异类型统计:")
    for mut_type, count in sorted(type_counts.items()):
        print(f"  {mut_type}: {count} 个")

# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("变异体生成器")
    print("=" * 60)
    print()
    
    # 生成并保存变异体
    operator_mapping = save_mutants()
    
    # 生成索引文件
    generate_mutant_index(operator_mapping)
    
    print(f"\n✓ 完成！共生成 {len(operator_mapping)-1} 个变异体")
    print(f"  原始代码: M00.py")
    print(f"  变异体: M01.py ~ M{len(operator_mapping)-1:02d}.py")