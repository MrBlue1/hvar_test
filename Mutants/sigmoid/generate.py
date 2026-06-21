import os
import ast
import copy
import random
from typing import List, Tuple, Dict

# ========== 配置 ==========
MUTANTS_DIR = "mutants"
ORIGINAL_CODE = '''import math
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
'''

# ========== AST节点定位工具 ==========
class NodeLocator(ast.NodeVisitor):
    """为每个AST节点分配唯一ID"""
    def __init__(self):
        self.node_to_id = {}
        self.id_to_node = {}
        self.counter = 0
    
    def visit(self, node):
        self.node_to_id[node] = self.counter
        self.id_to_node[self.counter] = node
        self.counter += 1
        self.generic_visit(node)

class TargetedTransformer(ast.NodeTransformer):
    """只替换指定ID的节点"""
    def __init__(self, target_id: int, new_node: ast.AST):
        self.target_id = target_id
        self.new_node = new_node
        self.locator = None
        self.current_id = 0
    
    def visit(self, node):
        if hasattr(node, '_node_id'):
            current_id = node._node_id
        else:
            current_id = self.current_id
            self.current_id += 1
        
        if current_id == self.target_id:
            return self.new_node
        return self.generic_visit(node)

def unparse(tree):
    """将AST转回代码字符串"""
    import astor
    return astor.to_source(tree)

# ========== 变异体生成器 ==========
class MutantGenerator:
    def __init__(self):
        self.mutants: List[Tuple[str, str]] = []  # (name, code)
        self.operator_map: Dict[str, str] = {'M00': 'ORIG'}
        self.counter = 1
        self.tree = ast.parse(ORIGINAL_CODE)
        self.locator = NodeLocator()
        self.locator.visit(self.tree)
        
        # 为所有节点添加ID属性（用于后续定位）
        self._add_node_ids(self.tree)
    
    def _add_node_ids(self, node):
        """递归为节点添加_id属性"""
        node._node_id = self.locator.node_to_id.get(node, -1)
        for child in ast.iter_child_nodes(node):
            self._add_node_ids(child)
    
    def add_mutant(self, code: str, operator: str) -> str:
        """添加变异体，返回变异体名称"""
        name = f"M{self.counter:02d}"
        self.mutants.append((name, code))
        self.operator_map[name] = operator
        self.counter += 1
        return name
    
    def _replace_node(self, target_node: ast.AST, new_node: ast.AST) -> ast.AST:
        """替换指定节点并返回新AST"""
        target_id = target_node._node_id
        transformer = TargetedTransformer(target_id, new_node)
        new_tree = transformer.visit(copy.deepcopy(self.tree))
        self._add_node_ids(new_tree)
        return new_tree
    
    # ========== ROR: 关系运算符替换 ==========
    def generate_ror_mutants(self):
        class RORFinder(ast.NodeVisitor):
            def __init__(self):
                self.targets = []
            def visit_Compare(self, node):
                if len(node.ops) == 1:
                    if isinstance(node.ops[0], ast.GtE):
                        self.targets.append((node, ast.Gt()))
                    elif isinstance(node.ops[0], ast.Gt):
                        self.targets.append((node, ast.GtE()))
                    elif isinstance(node.ops[0], ast.LtE):
                        self.targets.append((node, ast.Lt()))
                    elif isinstance(node.ops[0], ast.Lt):
                        self.targets.append((node, ast.LtE()))
                    elif isinstance(node.ops[0], ast.Eq):
                        self.targets.append((node, ast.NotEq()))
                    elif isinstance(node.ops[0], ast.NotEq):
                        self.targets.append((node, ast.Eq()))
                self.generic_visit(node)
        
        finder = RORFinder()
        finder.visit(self.tree)
        
        for target_node, new_op in finder.targets:
            new_compare = copy.deepcopy(target_node)
            new_compare.ops = [new_op]
            new_tree = self._replace_node(target_node, new_compare)
            self.add_mutant(unparse(new_tree), 'ROR')
    
    # ========== AOR: 算术运算符替换 ==========
    def generate_aor_mutants(self):
        op_map = {
            ast.Add: [ast.Sub(), ast.Mult(), ast.Div()],
            ast.Sub: [ast.Add(), ast.Mult(), ast.Div()],
            ast.Mult: [ast.Add(), ast.Sub(), ast.Div()],
            ast.Div: [ast.Add(), ast.Sub(), ast.Mult()]
        }
        
        class AORFinder(ast.NodeVisitor):
            def __init__(self):
                self.targets = []
            def visit_BinOp(self, node):
                if type(node.op) in op_map:
                    for new_op in op_map[type(node.op)]:
                        self.targets.append((node, new_op))
                self.generic_visit(node)
        
        finder = AORFinder()
        finder.visit(self.tree)
        
        for target_node, new_op in finder.targets[:15]:  # 限制数量避免过多
            new_binop = copy.deepcopy(target_node)
            new_binop.op = new_op
            new_tree = self._replace_node(target_node, new_binop)
            self.add_mutant(unparse(new_tree), 'AOR')
    
    # ========== LOR: 逻辑运算符替换 ==========
    def generate_lor_mutants(self):
        op_map = {
            ast.And: ast.Or(),
            ast.Or: ast.And()
        }
        
        class LORFinder(ast.NodeVisitor):
            def __init__(self):
                self.targets = []
            def visit_BoolOp(self, node):
                if type(node.op) in op_map:
                    self.targets.append((node, op_map[type(node.op)]))
                self.generic_visit(node)
        
        finder = LORFinder()
        finder.visit(self.tree)
        
        for target_node, new_op in finder.targets:
            new_boolop = copy.deepcopy(target_node)
            new_boolop.op = new_op
            new_tree = self._replace_node(target_node, new_boolop)
            self.add_mutant(unparse(new_tree), 'LOR')
    
    # ========== COR: 条件运算符替换 ==========
    def generate_cor_mutants(self):
        # 替换if条件中的比较运算符（与ROR类似但只针对if条件）
        class CORFinder(ast.NodeVisitor):
            def __init__(self):
                self.targets = []
            def visit_If(self, node):
                if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
                    if isinstance(node.test.ops[0], ast.GtE):
                        self.targets.append((node.test, ast.Gt()))
                    elif isinstance(node.test.ops[0], ast.Gt):
                        self.targets.append((node.test, ast.GtE()))
                self.generic_visit(node)
        
        finder = CORFinder()
        finder.visit(self.tree)
        
        for target_node, new_op in finder.targets:
            new_compare = copy.deepcopy(target_node)
            new_compare.ops = [new_op]
            new_tree = self._replace_node(target_node, new_compare)
            self.add_mutant(unparse(new_tree), 'COR')
    
    # ========== UOI: 一元运算符插入/删除 ==========
    def generate_uoi_mutants(self):
        class UOIFinder(ast.NodeVisitor):
            def __init__(self):
                self.unary_ops = []  # 可删除的一元运算符
                self.expressions = []  # 可插入负号的表达式
            
            def visit_UnaryOp(self, node):
                if isinstance(node.op, ast.USub):
                    self.unary_ops.append(node)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                if node.id == 'x':
                    self.expressions.append(node)
            
            def visit_Num(self, node):
                self.expressions.append(node)
            
            def visit_BinOp(self, node):
                self.expressions.append(node)
        
        finder = UOIFinder()
        finder.visit(self.tree)
        
        # 删除一元负号
        for target_node in finder.unary_ops[:3]:
            new_tree = self._replace_node(target_node, target_node.operand)
            self.add_mutant(unparse(new_tree), 'UOI')
        
        # 插入一元负号
        for target_node in finder.expressions[:5]:
            new_unary = ast.UnaryOp(op=ast.USub(), operand=copy.deepcopy(target_node))
            new_tree = self._replace_node(target_node, new_unary)
            self.add_mutant(unparse(new_tree), 'UOI')
    
    # ========== SDL: 语句删除 ==========
    def generate_sdl_mutants(self):
        class SDLFinder(ast.NodeVisitor):
            def __init__(self):
                self.statements = []
            
            def visit_Assign(self, node):
                self.statements.append(node)
            
            def visit_Return(self, node):
                # 不删除return语句，会导致语法错误
                pass
        
        finder = SDLFinder()
        finder.visit(self.tree)
        
        for target_node in finder.statements[:5]:
            # 用pass替换语句
            new_tree = self._replace_node(target_node, ast.Pass())
            self.add_mutant(unparse(new_tree), 'SDL')
    
    # ========== ABS: 绝对值相关 ==========
    def generate_abs_mutants(self):
        class ABSFinder(ast.NodeVisitor):
            def __init__(self):
                self.targets = []
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'exp':
                    self.targets.append(node)
                self.generic_visit(node)
        
        finder = ABSFinder()
        finder.visit(self.tree)
        
        for target_node in finder.targets:
            # 将 exp(x) 改为 exp(abs(x))
            new_call = copy.deepcopy(target_node)
            abs_call = ast.Call(
                func=ast.Name(id='abs', ctx=ast.Load()),
                args=new_call.args,
                keywords=[]
            )
            new_call.args = [abs_call]
            new_tree = self._replace_node(target_node, new_call)
            self.add_mutant(unparse(new_tree), 'ABS')
    
    # ========== EQUI: 等价变异 ==========
    def generate_equi_mutants(self):
        # 手动生成几个数学等价变换
        equi_mutations = [
            # 1. 1/(1+z) 改为 (1+z-z)/(1+z)
            '''import math
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return (1 + z - z) / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)
''',
            # 2. z/(1+z) 改为 1 - 1/(1+z)
            '''import math
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return 1 - 1 / (1 + z)
''',
            # 3. 使用 tanh 等价形式
            '''import math
def sigmoid(x):
    return 0.5 * (1 + math.tanh(x / 2))
''',
            # 4. 使用条件表达式简化
            '''import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
''',
            # 5. 另一种等价形式
            '''import math
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))
'''
        ]
        
        for i, code in enumerate(equi_mutations[:5]):
            self.add_mutant(code, 'EQUI')
    
    # ========== 额外变异：常数值修改 ==========
    def generate_constant_mutants(self):
        class ConstFinder(ast.NodeVisitor):
            def __init__(self):
                self.constants = []
            
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    self.constants.append(node)
        
        finder = ConstFinder()
        finder.visit(self.tree)
        
        for target_node in finder.constants[:5]:
            old_val = target_node.value
            if isinstance(old_val, (int, float)):
                if old_val == 0:
                    new_val = 1
                elif old_val == 1:
                    new_val = 0
                elif old_val > 0:
                    new_val = old_val * 2
                else:
                    new_val = old_val / 2
                
                new_const = ast.Constant(value=new_val)
                new_tree = self._replace_node(target_node, new_const)
                self.add_mutant(unparse(new_tree), 'CRP')  # Constant Replacement
    
    # ========== 生成所有变异体 ==========
    def generate_all(self):
        print("生成ROR变异体...")
        self.generate_ror_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成AOR变异体...")
        self.generate_aor_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成LOR变异体...")
        self.generate_lor_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成COR变异体...")
        self.generate_cor_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成UOI变异体...")
        self.generate_uoi_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成SDL变异体...")
        self.generate_sdl_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成ABS变异体...")
        self.generate_abs_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成EQUI变异体...")
        self.generate_equi_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        print("生成CRP变异体...")
        self.generate_constant_mutants()
        print(f"当前变异体数量: {len(self.mutants)}")
        
        # 如果变异体数量不足60，补充额外的AOR变异
        while len(self.mutants) < 60:
            print("补充额外变异体...")
            self.generate_aor_mutants()
            if len(self.mutants) >= 60:
                break
            self.generate_uoi_mutants()


# ========== 写入文件 ==========
def write_mutants(generator: MutantGenerator):
    os.makedirs(MUTANTS_DIR, exist_ok=True)
    
    # 写入 M00.py
    with open(os.path.join(MUTANTS_DIR, "M00.py"), "w", encoding="utf-8") as f:
        f.write(ORIGINAL_CODE)
    print(f"已写入 M00.py (原始代码)")
    
    # 写入所有变异体
    for name, code in generator.mutants:
        path = os.path.join(MUTANTS_DIR, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
    
    print(f"已写入 {len(generator.mutants)} 个变异体")
    
    # 写入 MutantsIndex.py
    with open("MutantsIndex.py", "w", encoding="utf-8") as f:
        f.write("# 变异体索引文件\n")
        f.write("# 格式: {'变异体名称': '变异算子'}\n\n")
        f.write("OPERATOR_MAPPING = {\n")
        for name in sorted(generator.operator_map.keys()):
            f.write(f"    '{name}': '{generator.operator_map[name]}',\n")
        f.write("}\n")
    
    print("\nMutantsIndex.py 已生成")
    print(f"OPERATOR_MAPPING 包含 {len(generator.operator_map)} 个条目")


# ========== 主程序 ==========
if __name__ == "__main__":
    print("开始生成变异体...")
    gen = MutantGenerator()
    gen.generate_all()
    write_mutants(gen)
    
    # 打印统计信息
    print("\n=== 生成统计 ===")
    op_counts = {}
    for op in gen.operator_map.values():
        op_counts[op] = op_counts.get(op, 0) + 1
    
    for op, count in sorted(op_counts.items()):
        print(f"  {op}: {count} 个变异体")
    
    print(f"\n总计: {len(gen.mutants)} 个变异体")