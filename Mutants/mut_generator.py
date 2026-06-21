#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能变异体自动生成器 (Smart Mutant Generator)
支持JSON配置驱动和自动生成模式
"""

import ast
import json
import random
import copy
import argparse
from typing import Dict, List, Optional, Union
from pathlib import Path


class MutationCollector(ast.NodeVisitor):
    """收集所有可变异点"""
    
    def __init__(self):
        self.points = []
    
    def visit_Compare(self, node):
        for op in node.ops:
            self.points.append({
                'type': 'ROR',
                'lineno': node.lineno,
                'col_offset': node.col_offset,
                'original': self._get_op_name(op),
                'ast_node': node
            })
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        self.points.append({
            'type': 'AOR',
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'original': self._get_op_name(node.op),
            'ast_node': node
        })
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        self.points.append({
            'type': 'LOR',
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'original': self._get_op_name(node.op),
            'ast_node': node
        })
        self.generic_visit(node)
    
    def visit_UnaryOp(self, node):
        self.points.append({
            'type': 'UOR',
            'lineno': node.lineno,
            'col_offset': node.col_offset,
            'original': self._get_op_name(node.op),
            'ast_node': node
        })
        self.generic_visit(node)
    
    def _get_op_name(self, op):
        mapping = {
            ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
            ast.Eq: '==', ast.NotEq: '!=', ast.Is: 'is', ast.IsNot: 'is not',
            ast.In: 'in', ast.NotIn: 'not in',
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.FloorDiv: '//', ast.Mod: '%', ast.Pow: '**',
            ast.And: 'and', ast.Or: 'or',
            ast.UAdd: '+', ast.USub: '-', ast.Not: 'not', ast.Invert: '~'
        }
        return mapping.get(type(op), 'unknown')


class MutantGenerator:
    """变异体生成器核心类"""
    
    OP_RULES = {
        'ROR': ['<', '<=', '>', '>=', '==', '!=', 'is', 'is not'],
        'AOR': ['+', '-', '*', '/', '//', '%', '**'],
        'LOR': ['and', 'or'],
        'UOR': ['+', '-', 'not', '~']
    }
    
    def __init__(self, source_file: str, output_dir: str = "./mutants"):
        self.source = Path(source_file)
        self.output = Path(output_dir)
        self.output.mkdir(exist_ok=True)
        
        with open(source_file, 'r', encoding='utf-8') as f:
            self.code = f.read()
        
        self.tree = ast.parse(self.code)
        collector = MutationCollector()
        collector.visit(self.tree)
        self.points = collector.points
    
    def generate_from_json(self, json_path: str) -> List[str]:
        """
        从JSON文件生成变异体
        JSON格式: {"M01": "ROR", "M02": "AOR", ...}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"📋 从 {json_path} 加载配置，共 {len(config)} 个变异体")
        return self._generate_by_config(config)
    
    def generate_auto(self, count: int, prefix: str = "M") -> List[str]:
        """自动生成指定数量的变异体"""
        config = {}
        available = self.points.copy()
        random.shuffle(available)
        
        for i in range(1, count + 1):
            if not available:
                available = self.points.copy()
                random.shuffle(available)
            point = available.pop()
            config[f"{prefix}{i:02d}"] = point['type']
        
        print(f"🎲 自动生成 {count} 个变异体配置")
        return self._generate_by_config(config)
    
    def _generate_by_config(self, config: Dict[str, str]) -> List[str]:
        """根据配置字典生成"""
        files = []
        
        for mutant_id, op_type in config.items():
            candidates = [p for p in self.points if p['type'] == op_type]
            if not candidates:
                print(f"⚠️  {mutant_id}: 无 {op_type} 变异点")
                continue
            
            point = random.choice(candidates)
            original = point['original']
            replacements = [op for op in self.OP_RULES[op_type] if op != original]
            new_op = random.choice(replacements) if replacements else original
            
            filepath = self._create_file(mutant_id, op_type, point, original, new_op)
            if filepath:
                files.append(filepath)
                print(f"✅ {mutant_id}: {op_type} L{point['lineno']} '{original}'→'{new_op}'")
        
        return files
    
    def _create_file(self, mid: str, optype: str, point: Dict, orig: str, new: str) -> str:
        """创建变异体文件"""
        try:
            new_tree = copy.deepcopy(self.tree)
            
            # 应用变异
            for node in ast.walk(new_tree):
                if (hasattr(node, 'lineno') and node.lineno == point['lineno'] and
                    hasattr(node, 'col_offset') and node.col_offset == point['col_offset']):
                    
                    if self._mutate_node(node, optype, new):
                        break
            
            ast.fix_missing_locations(new_tree)
            mutated_code = ast.unparse(new_tree)
            
            # 构建文件内容
            header = f'''"""MUTANT: {mid} | OP: {optype} | LOC: L{point['lineno']} | {orig} -> {new}"""\n\n'''
            content = header + mutated_code
            
            filename = f"{mid}.py"
            filepath = self.output / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ {mid} 失败: {e}")
            return None
    
    def _mutate_node(self, node, op_type: str, new_op: str) -> bool:
        """执行节点变异"""
        op_map = {
            '<': ast.Lt(), '<=': ast.LtE(), '>': ast.Gt(), '>=': ast.GtE(),
            '==': ast.Eq(), '!=': ast.NotEq(), 'is': ast.Is(), 'is not': ast.IsNot(),
            '+': ast.Add(), '-': ast.Sub(), '*': ast.Mult(), '/': ast.Div(),
            '//': ast.FloorDiv(), '%': ast.Mod(), '**': ast.Pow(),
            'and': ast.And(), 'or': ast.Or(),
            'not': ast.Not(), '~': ast.Invert(), '+': ast.UAdd(), '-': ast.USub()
        }
        
        try:
            if op_type == 'ROR' and isinstance(node, ast.Compare):
                node.ops = [op_map[new_op]]
                return True
            elif op_type == 'AOR' and isinstance(node, ast.BinOp):
                node.op = op_map[new_op]
                return True
            elif op_type == 'LOR' and isinstance(node, ast.BoolOp):
                node.op = op_map[new_op]
                return True
            elif op_type == 'UOR' and isinstance(node, ast.UnaryOp):
                node.op = op_map[new_op]
                return True
        except:
            pass
        return False
    
    def stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("变异点统计:")
        for op in ['ROR', 'AOR', 'LOR', 'UOR']:
            count = len([p for p in self.points if p['type'] == op])
            print(f"  {op}: {count}")
        print(f"  TOTAL: {len(self.points)}")
        print("="*50)


def main():
    print('请输入源程序\n')
    parser = argparse.ArgumentParser(description='变异体自动生成器')
    parser.add_argument('source', help='源程序文件 (如 M00.py)')
    parser.add_argument('-c', '--config', help='JSON配置文件路径')
    parser.add_argument('-n', '--number', type=int, help='自动生成数量')
    parser.add_argument('-o', '--output', default='./mutants', help='输出目录')
    parser.add_argument('-p', '--prefix', default='M', help='文件前缀')
    
    args = parser.parse_args()
    
    # 初始化
    gen = MutantGenerator(args.source, args.output)
    gen.stats()
    
    # 生成模式选择
    if args.config:
        files = gen.generate_from_json(args.config)
    elif args.number:
        files = gen.generate_auto(args.number, args.prefix)
    else:
        # 默认生成示例
        print("\n使用示例配置生成...")
        example_config = {f"M{i:02d}": random.choice(['ROR', 'AOR', 'LOR']) 
                         for i in range(1, 7)}
        files = gen._generate_by_config(example_config)
    
    print(f"\n🎉 完成！共生成 {len(files)} 个变异体 -> {args.output}/")


if __name__ == "__main__":
    main()