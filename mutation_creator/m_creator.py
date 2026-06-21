import ast
import astunparse  # pip install astunparse (或Python 3.9+用ast.unparse)
import copy
import os

class MutantGenerator(ast.NodeTransformer):
    def __init__(self):
        self.mutants = []
        self.mutant_id = 0
        
    # ========== AOR: 算术运算符替换 ==========
    def visit_BinOp(self, node):
        aor_ops = {
            ast.Add: [ast.Sub, ast.Mult, ast.Div],
            ast.Sub: [ast.Add, ast.Mult, ast.Div],
            ast.Mult: [ast.Add, ast.Sub, ast.Div],
            ast.Div: [ast.Add, ast.Sub, ast.Mult],
        }
        
        if type(node.op) in aor_ops:
            for new_op_type in aor_ops[type(node.op)]:
                new_node = copy.deepcopy(node)
                new_node.op = new_op_type()
                self._save_mutant(new_node, f"AOR_{type(node.op).__name__}_to_{new_op_type.__name__}")
        
        return self.generic_visit(node)
    
    # ========== ROR: 关系运算符替换 ==========
    def visit_Compare(self, node):
        ror_ops = {
            ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
            ast.NotEq: [ast.Eq, ast.LtE, ast.GtE],
            ast.Lt: [ast.LtE, ast.Gt, ast.Eq],
            ast.Gt: [ast.GtE, ast.Lt, ast.Eq],
            ast.LtE: [ast.Lt, ast.GtE, ast.NotEq],
            ast.GtE: [ast.Gt, ast.LtE, ast.NotEq],
        }
        
        if node.ops and type(node.ops[0]) in ror_ops:
            for new_op_type in ror_ops[type(node.ops[0])]:
                new_node = copy.deepcopy(node)
                new_node.ops = [new_op_type()]
                self._save_mutant(new_node, f"ROR_{type(node.ops[0]).__name__}_to_{new_op_type.__name__}")
        
        return self.generic_visit(node)
    
    # ========== LOR: 逻辑运算符替换 ==========
    def visit_BoolOp(self, node):
        lor_ops = {
            ast.And: ast.Or,
            ast.Or: ast.And,
        }
        
        if type(node.op) in lor_ops:
            new_node = copy.deepcopy(node)
            new_node.op = lor_ops[type(node.op)]()
            self._save_mutant(new_node, f"LOR_{type(node.op).__name__}_to_{lor_ops[type(node.op)].__name__}")
        
        return self.generic_visit(node)
    
    # ========== UOI: 一元运算符插入 ==========
    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.UAdd):
            new_node = copy.deepcopy(node)
            new_node.op = ast.USub()
            self._save_mutant(new_node, "UOI_UAdd_to_USub")
        elif isinstance(node.op, ast.USub):
            new_node = copy.deepcopy(node)
            new_node.op = ast.UAdd()
            self._save_mutant(new_node, "UOI_USub_to_UAdd")
        
        return self.generic_visit(node)
    
    def _save_mutant(self, mutated_node, op_desc):
        """保存变异体到列表"""
        self.mutant_id += 1
        tree_copy = copy.deepcopy(self.original_tree)
        
        # 找到对应节点并替换（简化版，实际需更精确的节点定位）
        # 这里使用节点ID或路径来定位更准确
        self.mutants.append({
            'id': self.mutant_id,
            'operator': op_desc,
            'tree': tree_copy,
            'node': mutated_node
        })

def generate_mutants(source_file, output_dir="mutants"):
    with open(source_file, 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    # 为每个可能的变异点生成变异体
    generator = MutantGenerator()
    generator.original_tree = tree
    
    # 这里需要更精细的实现：遍历树并记录所有可变异点
    # 然后为每个点生成一个变异版本
    visitor = CollectMutations(generator)
    visitor.visit(tree)
    
    # 导出到文件
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mutant in enumerate(generator.mutants[:60]):  # 限制60个
        filename = os.path.join(output_dir, f"M{i+1:02d}.py")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Mutant {i+1}: {mutant['operator']}\n")
            f.write(astunparse.unparse(mutant['tree']))
    
    print(f"已生成 {len(generator.mutants)} 个变异体（导出前60个）")

# 使用
generate_mutants("M00.py")