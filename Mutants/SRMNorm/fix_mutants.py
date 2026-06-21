# fix_mutants.py - 放在与M00.py同级目录运行
import os
import re
import ast

def check_and_fix_mutant(filepath):
    """检查并尝试修复变异体文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixes = []
    
    # Fix 1: 修复科学计数法中的空格 (如 "1e -6" -> "1e-6")
    content = re.sub(r'(\d)e\s*-\s*(\d)', r'\1e-\2', content)
    content = re.sub(r'(\d)e\s*\+\s*(\d)', r'\1e+\2', content)
    
    # Fix 2: 修复小数点前导零缺失 (如 ".5" -> "0.5")
    content = re.sub(r'(?<!\d)\.(?=\d)', r'0.', content)
    
    # Fix 3: 检查括号匹配
    open_parens = content.count('(')
    close_parens = content.count(')')
    if open_parens != close_parens:
        fixes.append(f"括号不匹配: {open_parens} vs {close_parens}")
    
    # Fix 4: 检查赋值语法 (针对 cannot assign to expression)
    content = re.sub(r'(\w+)\s*=\s*=\s*', r'\1 == ', content)  # == 误写为 =
    
    # Fix 5: 确保类名正确（针对M78等）
    if 'class' in content and 'RMSNorm' not in content:
        # 检查是否有 LayerNorm 或其他类名，统一改为 RMSNorm
        content = re.sub(r'class\s+(\w+)(\(.*\)):', r'class RMSNorm\2:', content)
        if 'RMSNorm' not in original_content:
            fixes.append("统一类名为 RMSNorm")
    
    # 尝试语法检查
    try:
        ast.parse(content)
        status = "OK"
    except SyntaxError as e:
        status = f"SyntaxError: {e.msg} (line {e.lineno})"
        fixes.append(f"无法自动修复: {e.msg}")
    
    # 如果有修改，写回
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ {os.path.basename(filepath)}: 已自动修复 -> {status}")
    else:
        if status == "OK":
            print(f"✓ {os.path.basename(filepath)}: 正常")
        else:
            print(f"✗ {os.path.basename(filepath)}: {status}")
    
    return status == "OK"

# 检查所有变异体
mutant_files = [f for f in os.listdir('.') if f.startswith('M') and f.endswith('.py')]
mutant_files.sort()

valid_count = 0
error_files = []

for mf in mutant_files:
    if check_and_fix_mutant(mf):
        valid_count += 1
    else:
        error_files.append(mf)

print(f"\n{'='*50}")
print(f"有效变异体: {valid_count}/{len(mutant_files)}")
if error_files:
    print(f"需手动修复: {error_files}")
    print("\n请手动检查这些文件的对应行号：")
    for ef in error_files:
        print(f"  - {ef}")