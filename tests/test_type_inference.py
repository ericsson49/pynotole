import unittest

import ast
import astor

from pynotole.type_inference.inference import infer_types_with_mypy, infer_variable_types
from utils.pyast_utils import parse_stmts as parse

def to_src(stmts):
    assert isinstance(stmts, list)
    return astor.to_source(ast.Module(stmts))

class TypeInferenceTestCase(unittest.TestCase):
    def compare_stmt(self, s1, s2, orig):
        assert isinstance(s1, ast.FunctionDef)
        assert isinstance(s2, ast.FunctionDef)
        assert isinstance(orig, ast.FunctionDef)
        assert s1.name == s2.name == orig.name
        assert len(s1.body) >= len(orig.body)
        assert len(s2.body) >= len(orig.body)
        s1_defs, s1_body = s1.body[:-len(orig.body)], s1.body[-len(orig.body):]
        s2_defs, s2_body = s2.body[:-len(orig.body)], s2.body[-len(orig.body):]
        self.assertEqual(to_src(s1_body), to_src(orig.body))
        self.assertEqual(to_src(s2_body), to_src(orig.body))

        self.assertEqual(to_src(s1_defs), to_src(s2_defs))

    def compare(self, stmts1, stmts2, orig):
        self.assertEqual(len(stmts1), len(stmts2))
        self.assertEqual(len(stmts1), len(orig))
        for s1, s2, o in zip(stmts1, stmts2, orig):
            self.compare_stmt(s1, s2, o)


    def test_assign(self):
        stmts = parse('''
def f(a: int):
    b = a
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'b': 'int'}, var_types)

    def test_if_int_str(self):
        stmts = parse('''
def f(a: int):
    if a:
        c = 1
    else:
        c = ''
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'c': 'object'}, var_types)

    def test_if_int_bool(self):
        stmts = parse('''
def f(a: int):
    if a:
        c = 1
    else:
        c = True
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'c': 'int'}, var_types)

    def test_if_int_bool_str(self):
        stmts = parse('''
def f(a: int):
    if a == 0:
        c = 1
    elif a > 0:
        c = True
    else:
        c = ''
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'c': 'object'}, var_types)

    def test_if_ann_assign(self):
        stmts = parse('''
def f(a: int):
    if a:
        c: int = 1
        c_2 = c
    else:
        c_1 = True
        c_2 = c_1
    b = c_2
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'c_1': 'bool', 'c_2': 'int', 'b': 'int'}, var_types)

    def test_while(self):
        stmts = parse('''
def f(n: int):
    a = 0
    while n:
        a = a + 1
        n = n - 1
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'a': 'int', 'n': 'int'}, var_types)

    def test_while_ssa(self):
        stmts = parse('''
def f(n: int):
    a = False
    a_1 = a
    n_1 = n
    while n_1:
        a_2 = a_1 + 1
        n_2 = n_1 - 1
        a_1 = a_2
        n_1 = n_2
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'a': 'bool', 'a_1': 'int', 'a_2': 'int', 'n_1': 'int', 'n_2': 'int'}, var_types)

    def test_for(self):
        stmts = parse('''
def f(n: int):
    a = True
    for i in range(n):
        a = a + i
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'a': 'int', 'i': 'int'}, var_types)

    def test_for_ssa(self):
        stmts = parse('''
def f(n: int):
    a = False
    a_1 = a
    for i in range(n):
        a_2 = a_1 + 1
        a_1 = a_2
''')
        var_types, = infer_variable_types(stmts).values()
        self.assertEqual({'a': 'bool', 'a_1': 'int', 'a_2': 'int', 'i': 'int'}, var_types)



if __name__ == '__main__':
    unittest.main()
