import os
import shutil
import math
from pathlib import Path
from typing import List, Tuple

ORIGINAL_CODE = '''import math

def gelu(x):
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    elif isinstance(x, list):
        return [gelu(item) for item in x]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
'''

def create_mutants_folder():
    mutants_dir = Path('mutants')
    if mutants_dir.exists():
        shutil.rmtree(mutants_dir)
    mutants_dir.mkdir(exist_ok=True)
    return mutants_dir

def save_mutant(mutants_dir, filename, code):
    with open(mutants_dir / filename, 'w', encoding='utf-8') as f:
        f.write(code)

def generate_all_mutants():
    """生成至少70个语法正确的变异体"""
    mutants_dir = create_mutants_folder()
    
    # 保存原始代码
    save_mutant(mutants_dir, 'M00.py', ORIGINAL_CODE)
    
    mutants = []
    idx = 1
    
    # ========== 1. NaN/Inf 变异体 (4个) ==========
    mutants.append(('M01', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return float("nan")'
    )))
    mutants.append(('M02', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return float("inf")'
    )))
    mutants.append(('M03', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return -float("inf")'
    )))
    
    # ========== 2. Underflow/Overflow 变异体 (2个) ==========
    mutants.append(('M04', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 1e-308 * x'
    )))
    mutants.append(('M05', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 1e308 * x'
    )))
    
    # ========== 3. 破坏符号保持的变异体 (2个) ==========
    mutants.append(('M06', 'UOI', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return -0.5 * x * (1 + math.erf(x / math.sqrt(2)))'
    )))
    mutants.append(('M07', 'UOI', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 0.5 * abs(x) * (1 + math.erf(x / math.sqrt(2)))'
    )))
    
    # ========== 4. 破坏零点对称的变异体 (1个) ==========
    mutants.append(('M08', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2))) + 0.1'
    )))
    
    # ========== 5. 列表长度不匹配变异体 (3个) ==========
    mutants.append(('M09', 'SDL', ORIGINAL_CODE.replace(
        'return [gelu(item) for item in x]',
        'return [gelu(item) for item in x] + [0]'
    )))
    mutants.append(('M10', 'SDL', ORIGINAL_CODE.replace(
        'return [gelu(item) for item in x]',
        'return [gelu(item) for item in x if item > 0]'
    )))
    mutants.append(('M11', 'SDL', ORIGINAL_CODE.replace(
        'return [gelu(item) for item in x]',
        'return [0] * len(x)'
    )))
    
    # ========== 6. 破坏单调性的变异体 (3个) ==========
    mutants.append(('M12', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2))) + 0.1 * math.sin(100 * x)'
    )))
    mutants.append(('M13', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2))) * (-1 if x < 0 else 1)'
    )))
    mutants.append(('M14', 'AOR', ORIGINAL_CODE.replace(
        '0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        '0.5 * x * (1 - math.erf(x / math.sqrt(2)))'
    )))
    
    # ========== 7. 破坏渐近行为的变异体 (2个) ==========
    mutants.append(('M15', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 0.5 * x * (1 + math.erf(x / 1000))'
    )))
    mutants.append(('M16', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 0.5 * x * 2'
    )))
    
    # ========== 8. 精度损失和erf近似误差 (2个) ==========
    mutants.append(('M17', 'AOR', ORIGINAL_CODE.replace(
        'math.erf(x / math.sqrt(2))',
        '(x / math.sqrt(2)) - (x / math.sqrt(2))**3 / 3'
    )))
    mutants.append(('M18', 'AOR', ORIGINAL_CODE.replace(
        'math.erf(x / math.sqrt(2))',
        'math.tanh(x / math.sqrt(2))'
    )))
    
    # ========== 9. 数据损坏变异体 (2个) ==========
    mutants.append(('M19', 'UOI', ORIGINAL_CODE.replace(
        'if isinstance(x, (int, float)):',
        'if isinstance(x, (int, float)):\n    x = x + 1'
    )))
    mutants.append(('M20', 'UOI', ORIGINAL_CODE.replace(
        'if isinstance(x, (int, float)):',
        'if isinstance(x, (int, float)):\n    x = x * 2'
    )))
    
    # ========== 10. 副作用泄露变异体 (1个) ==========
    mutants.append(('M21', 'UOI', '''import math

_counter = 0

def gelu(x):
    global _counter
    _counter += 1
    if isinstance(x, (int, float)):
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    elif isinstance(x, list):
        return [gelu(item) for item in x]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
'''))
    
    # ========== 11. 内存爆炸变异体 (1个) ==========
    mutants.append(('M22', 'SDL', ORIGINAL_CODE.replace(
        'return [gelu(item) for item in x]',
        'return [gelu(item) for item in x for _ in range(1000)]'
    )))
    
    # ========== 12. 范围违规变异体 (2个) ==========
    mutants.append(('M23', 'AOR', ORIGINAL_CODE.replace(
        'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))',
        'return 1e200 * x'
    )))
    mutants.append(('M24', 'AOR', ORIGINAL_CODE.replace(
        '0.5 * x',
        '1e200 * x'
    )))
    
    # ========== 13. 算术运算符替换变异体 (15个) ==========
    aor_mutants = [
        ('M25', '0.5 * x', '0.5 + x'),
        ('M26', '0.5 * x', '0.5 - x'),
        ('M27', '0.5 * x', '0.5 / x'),
        ('M28', 'x / math.sqrt(2)', 'x * math.sqrt(2)'),
        ('M29', 'x / math.sqrt(2)', 'x + math.sqrt(2)'),
        ('M30', '1 + math.erf', '1 - math.erf'),
        ('M31', '0.5 * x', 'x ** 0.5'),
        ('M32', '0.5 * x', '0.6 * x'),
        ('M33', '0.5 * x', '1.0 * x'),
        ('M34', 'math.sqrt(2)', 'math.sqrt(3)'),
        ('M35', 'math.erf(x / math.sqrt(2))', 'math.erf(x)'),
        ('M36', 'math.erf(x / math.sqrt(2))', 'math.erfc(x / math.sqrt(2))'),
        ('M37', '0.5 * x * (1 + math.erf)', 'x * 0.5 * (1 + math.erf)'),
        ('M38', '0.5 * x', '0.5 ** x'),
        ('M39', 'math.sqrt(2)', '2 ** 0.5'),
    ]
    for i, (name, old, new) in enumerate(aor_mutants, start=25):
        mutants.append((name, 'AOR', ORIGINAL_CODE.replace(old, new)))
    
    # ========== 14. 条件运算符替换变异体 (10个) ==========
    cor_mutants = [
        ('M40', 'if isinstance(x, (int, float)):', 'if True:'),
        ('M41', 'if isinstance(x, (int, float)):', 'if False:'),
        ('M42', 'elif isinstance(x, list):', 'else:'),
        ('M43', 'elif isinstance(x, list):', 'if isinstance(x, list):'),
        ('M44', 'else:', 'elif True:'),
        ('M45', 'if isinstance(x, (int, float)):', 'if x is not None and isinstance(x, (int, float)):'),
        ('M46', 'if isinstance(x, (int, float)):', 'if isinstance(x, (int, float)) and True:'),
        ('M47', 'if isinstance(x, (int, float)):', 'if isinstance(x, (int, float)) or False:'),
        ('M48', 'elif isinstance(x, list):', 'elif type(x) == list:'),
        ('M49', 'raise TypeError', 'raise ValueError'),
    ]
    for i, (name, old, new) in enumerate(cor_mutants, start=40):
        mutants.append((name, 'COR', ORIGINAL_CODE.replace(old, new)))
    
    # ========== 15. 逻辑运算符替换变异体 (8个) ==========
    lor_mutants = [
        ('M50', 'isinstance(x, (int, float))', 'isinstance(x, (int, float)) and True'),
        ('M51', 'isinstance(x, (int, float))', 'isinstance(x, (int, float)) or False'),
        ('M52', 'isinstance(x, list)', 'not isinstance(x, list)'),
        ('M53', 'if isinstance(x, (int, float)):', 'if isinstance(x, (int, float)) and x is not None:'),
        ('M54', 'elif isinstance(x, list):', 'elif isinstance(x, list) or isinstance(x, tuple):'),
        ('M55', 'isinstance(x, (int, float))', 'type(x) in (int, float)'),
        ('M56', 'isinstance(x, list)', 'isinstance(x, (list, tuple))'),
        ('M57', 'raise TypeError', 'if True: raise TypeError'),
    ]
    for i, (name, old, new) in enumerate(lor_mutants, start=50):
        mutants.append((name, 'LOR', ORIGINAL_CODE.replace(old, new)))
    
    # ========== 16. 一元运算符变异体 (10个) ==========
    uoi_mutants = [
        ('M58', 'return 0.5 * x', 'return -0.5 * x'),
        ('M59', 'return 0.5 * x', 'return +0.5 * x'),
        ('M60', 'math.erf(x / math.sqrt(2))', '-math.erf(x / math.sqrt(2))'),
        ('M61', 'math.sqrt(2)', '-math.sqrt(2)'),
        ('M62', '[gelu(item) for item in x]', '[gelu(-item) for item in x]'),
        ('M63', '[gelu(item) for item in x]', '[gelu(abs(item)) for item in x]'),
        ('M64', 'if isinstance(x, (int, float)):', 'if not isinstance(x, (int, float)):'),
        ('M65', '0.5 * x', '-0.5 * x'),
        ('M66', 'return 0.5 * x', 'return abs(0.5 * x)'),
        ('M67', '1 + math.erf', '-(1 + math.erf)'),
    ]
    for i, (name, old, new) in enumerate(uoi_mutants, start=58):
        mutants.append((name, 'UOI', ORIGINAL_CODE.replace(old, new)))
    
    # ========== 17. 语句删除变异体 (5个) ==========
    sdl_mutants = [
        ('M68', 'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))', 'pass'),
        ('M69', 'math.erf(x / math.sqrt(2))', '0'),
        ('M70', '/ math.sqrt(2)', ''),
        ('M71', 'raise TypeError', 'pass'),
        ('M72', 'return [gelu(item) for item in x]', 'return x'),
    ]
    for i, (name, old, new) in enumerate(sdl_mutants, start=68):
        code = ORIGINAL_CODE.replace(old, new)
        if old == 'return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))':
            code = code.replace('return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))', 'pass\n        return None')
        mutants.append((name, 'SDL', code))
    
    # ========== 18. 等价变异体 (8个) ==========
    equi_mutants = [
        ('M73', '0.5 * x * (1 + math.erf(x / math.sqrt(2)))', 'x * (0.5 + 0.5 * math.erf(x / math.sqrt(2)))'),
        ('M74', '0.5 * x * (1 + math.erf(x / math.sqrt(2)))', 'x * (1 - 0.5 * math.erfc(x / math.sqrt(2)))'),
        ('M75', 'math.sqrt(2)', 'pow(2, 0.5)'),
        ('M76', 'isinstance(x, (int, float))', 'type(x) in (int, float)'),
        ('M77', 'isinstance(x, list)', 'type(x) == list'),
        ('M78', 'return [gelu(item) for item in x]', 'result = []\n        for item in x:\n            result.append(gelu(item))\n        return result'),
        ('M79', 'f"Unsupported type: {type(x)}"', '"Unsupported type: " + str(type(x))'),
        ('M80', '0.5', '1/2'),
    ]
    for i, (name, old, new) in enumerate(equi_mutants, start=73):
        mutants.append((name, 'EQUI', ORIGINAL_CODE.replace(old, new)))
    
    # 保存所有变异体
    operator_mapping = {'M00': 'ORIG'}
    for name, mtype, code in mutants:
        save_mutant(mutants_dir, f'{name}.py', code)
        operator_mapping[name] = mtype
    
    # 保存索引文件
    index_content = "OPERATOR_MAPPING = {\n"
    for name, mtype in sorted(operator_mapping.items()):
        index_content += f"    '{name}': '{mtype}',\n"
    index_content += "}\n"
    Path('MutantsIndex.py').write_text(index_content, encoding='utf-8')
    
    print(f"生成了 {len(mutants)} 个变异体")
    return mutants, operator_mapping

if __name__ == "__main__":
    generate_all_mutants()