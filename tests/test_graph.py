import ast
import astor
import unittest

from pynotole.cfg import CFGBuilder, DummyBlock, IndexingLabelFactory
from pynotole.graph import *
from pynotole.var_range_splitter import split_var_ranges

from utils.graph_utils import *
from utils.ssa_utils import *

labeler = IndexingLabelFactory()
bbf = PyASTBBlockFactory()


class CFGBuilderTest(unittest.TestCase):
    def base_checks(self, cfg):
        start_node = get_start_node(cfg.graph)
        end_node = get_start_node(cfg.graph.reverse())
        self.assertEqual(cfg.entry, start_node)
        self.assertEqual(cfg.exit, end_node)

    def test_on_assign(self):
        b = CFGBuilder(labeler, bbf)
        entry, exit = b.get_entry(), b.get_exit()

        l0 = b.on_stmt(b.get_label(entry, 0), parse_stmt('a = b'), exit)
        b.add_edge(b.add_node(entry, DummyBlock()), l0)
        cfg = b.build()

        self.base_checks(cfg)

        self.assertEqual([l0], cfg.get_succs(entry))
        self.assertEqual({'a'}, cfg.get_defs(l0))
        self.assertEqual({'b'}, cfg.get_uses(l0))
        self.assertEqual([exit], cfg.get_succs(l0))

    def test_on_expr(self):
        b = CFGBuilder(labeler, bbf)
        entry, exit = b.get_entry(), b.get_exit()

        l0 = b.on_stmt(b.get_label(entry, 0), parse_stmt('f(a, b, c)'), exit)
        b.add_edge(b.add_node(entry, DummyBlock()), l0)
        cfg = b.build()

        self.base_checks(cfg)

        self.assertEqual([l0], cfg.get_succs(entry))
        self.assertEqual(set(), cfg.get_defs(l0))
        self.assertEqual({'a', 'b', 'c', 'f'}, cfg.get_uses(l0))
        self.assertEqual([exit], cfg.get_succs(l0))

    def test_on_if(self):
        b = CFGBuilder(labeler, bbf)
        entry, exit = b.get_entry(), b.get_exit()

        l0 = b.on_stmt(b.get_label(entry, 0), parse_stmt('if a:\n  x = b\nelse:\n  x = c'), exit)
        b.add_edge(b.add_node(entry, DummyBlock()), l0)
        cfg = b.build()

        self.base_checks(cfg)

        self.assertEqual([l0], cfg.get_succs(entry))
        self.assertEqual(set(), cfg.get_defs(l0))
        self.assertEqual({'a'}, cfg.get_uses(l0))

        l1, l2 = cfg.get_succs(l0)
        self.assertEqual([exit], cfg.get_succs(l1))
        self.assertEqual({'x'}, cfg.get_defs(l1))
        self.assertEqual({'b'}, cfg.get_uses(l1))

        self.assertEqual([exit], cfg.get_succs(l2))
        self.assertEqual({'x'}, cfg.get_defs(l2))
        self.assertEqual({'c'}, cfg.get_uses(l2))

    def test_on_while(self):
        b = CFGBuilder(labeler, bbf)
        entry, exit = b.get_entry(), b.get_exit()

        l0 = b.on_stmt(b.get_label(entry, 0), parse_stmt('while a:\n  a = a - 1'), exit)
        b.add_edge(b.add_node(entry, DummyBlock()), l0)
        cfg = b.build()

        self.base_checks(cfg)

        self.assertEqual([l0], cfg.get_succs(entry))
        self.assertEqual(set(), cfg.get_defs(l0))
        self.assertEqual({'a'}, cfg.get_uses(l0))

        body, loop_exit = list(cfg.get_succs(l0))

        self.assertEqual({'a'}, cfg.get_defs(body))
        self.assertEqual({'a'}, cfg.get_uses(body))
        self.assertEqual([l0], cfg.get_succs(body))

        self.assertEqual(exit, loop_exit)

        lvs = live_vars(cfg)
        self.assertEqual({
            entry: {'a'},
            l0: {'a'},
            body: {'a'},
            exit: set()
        }, lvs)

        phis = phi_nodes(cfg)

        self.assertEqual({l0: {'a'}}, phis)

    def test_complex(self):
        b = CFGBuilder(labeler, bbf)
        entry, exit = b.get_entry(), b.get_exit()

        code = '''
n = 5
i = 0
while n:
    i = i + 1
    n = n -1
_return_ = i
        '''

        cfg = b.make_cfg(parse_stmts(code))

        self.base_checks(cfg)

        l0, = cfg.get_succs(entry)
        self.assertEqual({'n'}, cfg.get_defs(l0))
        self.assertEqual(set(), cfg.get_uses(l0))

        l1, = cfg.get_succs(l0)

        l_head, = cfg.get_succs(l1)

        l_body, l_exit = cfg.get_succs(l_head)

        self.assertEqual({'i'}, cfg.get_defs(l_body))
        self.assertEqual({'i'}, cfg.get_uses(l_body))

        l_body_2, = cfg.get_succs(l_body)
        self.assertEqual({'n'}, cfg.get_defs(l_body_2))
        self.assertEqual({'n'}, cfg.get_uses(l_body_2))
        self.assertEqual([l_head], cfg.get_succs(l_body_2))

        self.assertEqual({'_return_'}, cfg.get_defs(l_exit))
        self.assertEqual({'i'}, cfg.get_uses(l_exit))
        self.assertEqual([exit], cfg.get_succs(l_exit))

        lvs = live_vars(cfg)
        self.assertEqual(8, len(lvs))
        self.assertEqual(set(), lvs[entry])
        self.assertEqual(set(), lvs[l0])
        self.assertEqual({'n'}, lvs[l1])
        self.assertEqual({'n', 'i'}, lvs[l_head])
        self.assertEqual({'n', 'i'}, lvs[l_body])
        self.assertEqual({'n', 'i'}, lvs[l_body_2])
        self.assertEqual({'i'}, lvs[l_exit])
        self.assertEqual(set(), lvs[exit])

        phis = phi_nodes(cfg)
        self.assertEqual(1, len(phis))
        self.assertEqual({'n', 'i'}, phis[l_head])


def stmts_to_str(st: ast.stmt | list[ast.stmt]) -> str:
    if isinstance(st, ast.stmt):
        return astor.to_source(st)
    else:
        return ''.join([astor.to_source(s) for s in st])

class SplitVarRangesTestCase(unittest.TestCase):
    def compare(self, a, b):
        a_code = astor.to_source(ast.parse(astor.to_source(a)))
        b_code = astor.to_source(ast.parse(astor.to_source(b)))
        if ast.dump(ast.parse(a_code)) != ast.dump(ast.parse(b_code)):
            import difflib
            pairs = set()
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a_code, b=b_code, autojunk=False).get_opcodes():
                if tag == 'replace':
                    pairs.add((a_code[i1-1:i2].strip(), b_code[j1-1:j2].strip()))
                elif tag != 'equal':
                    self.assertEqual(a_code, b_code)
            unify(self, pairs)

    def test_split_var_ranges_simple(self):
        f = parse_stmt('''
def f(b):
    a = b
''')

        expected = parse_stmt('''
def f(b):
    a = b
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_if(self):
        f = parse_stmt('''
def f(c, b):
    if c:
        a = b
''')

        expected = parse_stmt('''
def f(c, b):
    if c:
        a = b
        ''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_if_else(self):
        f = parse_stmt('''
def f(c, b, d):
    if c:
        a = b
    else:
        a = d
    x = a
''')

        expected = parse_stmt('''
def f(c, b, d):
    if c:
        a_1 = b
        a_2 = a_1
    else:
        a = d
        a_2 = a
    x = a_2
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_if_else_return(self):
        f = parse_stmt('''
def f(c, b, d):
    if c:
        a = b
    else:
        a = d
    return a
''')

        expected = parse_stmt('''
def f(c, b, d):
    if c:
        a_1 = b
        a_2 = a_1
    else:
        a = d
        a_2 = a
    return a_2
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_while(self):
        f = parse_stmt('''
def f(n):
    a = 0
    while n:
        a = a + n
        n = n - 1
''')

        expected = parse_stmt('''
def f(n):
    a = 0
    a_1 = a
    n_2 = n
    while n_2:
        a_2 = a_1 + n_2
        n_3 = n_2 - 1
        a_1 = a_2
        n_2 = n_3
        
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_for(self):
        f = parse_stmt('''
def f(ls):
    a = 0
    for i in ls:
        a = a + i
''')

        expected = parse_stmt('''
def f(ls):
    a = 0
    a_1 = a
    for i in ls:
        a_2 = a_1 + i
        a_1 = a_2
        
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_while_break(self):
        f = parse_stmt('''
def f(n):
    a = 0
    while n:
        n = n - 1
        if n == CONST:
            break
''')

        expected = parse_stmt('''
def f(n):
    a = 0
    n_1 = n
    while n_1:
        n_2 = n_1 - 1
        if n_2 == CONST:
            break
        else:
            n_1 = n_2
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_while_continue(self):
        f = parse_stmt('''
def f(n):
    a = 0
    while n:
        n = n - 1
        if n == 4:
            continue
''')

        expected = parse_stmt('''
def f(n):
    a = 0
    n_1 = n
    while n_1:
        n_2 = n_1 - 1
        if n_2 == 4:
            n_1 = n_2
            continue
        else:
            n_1 = n_2
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_for_break(self):
        f = parse_stmt('''
def f(ls):
    a = 0
    for i in ls:
        a = a + i
        if i == 3:
            break
''')

        expected = parse_stmt('''
def f(ls):
    a = 0
    a_2 = a
    for i in ls:
        a_3 = a_2 + i
        if i == 3:
            break
        else:
            a_2 = a_3
''')
        self.compare(expected, split_var_ranges(f))

    def test_split_var_ranges_for_continue(self):
        f = parse_stmt('''
def f(ls):
    a = 0
    for i in ls:
        a = a + i
        if i == 3:
            continue
''')

        expected = parse_stmt('''
def f(ls):
    a = 0
    a_1 = a
    for i in ls:
        a_2 = a_1 + i
        if i == 3:
            a_1 = a_2
            continue
        else:
            a_1 = a_2
''')
        self.compare(expected, split_var_ranges(f))


class InstrBlockTestCase(unittest.TestCase):
    def test_defs_uses_simple(self):
        bb = BBlock(SimpleInstructionRenamer, mk_instrs('a = b\nc = a'))
        self.assertEqual(({'a', 'c'}, {'b'}), bb.get_defs_uses())

    def test_defs_uses_expr_simple(self):
        bb = BBlock(SimpleInstructionRenamer, mk_instrs('a'))
        self.assertEqual((set(), {'a'}), bb.get_defs_uses())

    def test_defs_uses_pyast(self):
        bb = BBlock(AstStmtRenamer, parse_stmts('a = b\nc = a'))
        self.assertEqual(({'a', 'c'}, {'b'}), bb.get_defs_uses())

    def test_defs_uses_expr_pyast(self):
        bb = BBlock(AstStmtRenamer, parse_stmts('a'))
        self.assertEqual((set(), {'a'}), bb.get_defs_uses())


class AstStmtRenamerTestCase(unittest.TestCase):
    def test_defs_uses_expr(self):
        self.assertEqual(set(), AstStmtRenamer.get_defs_uses(parse_stmt('1'))[1])
        self.assertEqual({'a'}, AstStmtRenamer.get_defs_uses(parse_stmt('a'))[1])
        self.assertEqual({'a'}, AstStmtRenamer.get_defs_uses(parse_stmt('a + 5'))[1])

        self.assertEqual({'a'}, AstStmtRenamer.get_defs_uses(parse_stmt('a.f'))[1])
        self.assertEqual({'a'}, AstStmtRenamer.get_defs_uses(parse_stmt('a[1]'))[1])
        self.assertEqual({'a', 'b'}, AstStmtRenamer.get_defs_uses(parse_stmt('a[b.f]'))[1])
        self.assertEqual({'a', 'b'}, AstStmtRenamer.get_defs_uses(parse_stmt('a.f[b]'))[1])

        self.assertEqual({'a', 'b'}, AstStmtRenamer.get_defs_uses(parse_stmt('a + b'))[1])
        self.assertEqual({'a', 'b', 'c'}, AstStmtRenamer.get_defs_uses(parse_stmt('a*b + c'))[1])
        self.assertEqual({'a', 'b'}, AstStmtRenamer.get_defs_uses(parse_stmt('a>b'))[1])
        self.assertEqual({'a', 'b', 'd', 'e'}, AstStmtRenamer.get_defs_uses(parse_stmt('a*e>b*d'))[1])
        self.assertEqual({'a'}, AstStmtRenamer.get_defs_uses(parse_stmt('-a'))[1])

        self.assertEqual({'a', 'b', 'c'}, AstStmtRenamer.get_defs_uses(parse_stmt('a or b > c'))[1])
        self.assertEqual({'a', 'b', 'c'}, AstStmtRenamer.get_defs_uses(parse_stmt('a and b > c'))[1])
        self.assertEqual({'a'}, AstStmtRenamer.get_defs_uses(parse_stmt('not a'))[1])

        self.assertEqual({'f'}, AstStmtRenamer.get_defs_uses(parse_stmt('f()'))[1])
        self.assertEqual({'f'}, AstStmtRenamer.get_defs_uses(parse_stmt('f(1)'))[1])
        self.assertEqual({'a', 'f'}, AstStmtRenamer.get_defs_uses(parse_stmt('f(a)'))[1])
        self.assertEqual({'a', 'b', 'f'}, AstStmtRenamer.get_defs_uses(parse_stmt('f(a, b)'))[1])
        self.assertEqual({'a', 'b', 'f'}, AstStmtRenamer.get_defs_uses(parse_stmt('f(a, k=b)'))[1])

        self.assertEqual({'a', 'b'}, AstStmtRenamer.get_defs_uses(parse_stmt('(a, b)'))[1])

        self.assertEqual({'a', 'b', 'c'}, AstStmtRenamer.get_defs_uses(parse_stmt('a if b else c'))[1])

    def test_defs_uses_assign(self):
        self.assertEqual(({'a'}, set()), AstStmtRenamer.get_defs_uses(parse_stmt('a = 1')))
        self.assertEqual(({'a'}, {'b'}), AstStmtRenamer.get_defs_uses(parse_stmt('a = b')))
        self.assertEqual(({'a'}, {'a'}), AstStmtRenamer.get_defs_uses(parse_stmt('a = a')))
        self.assertEqual(({'a'}, {'b', 'c'}), AstStmtRenamer.get_defs_uses(parse_stmt('a = b + c')))

        self.assertEqual((set(), {'a', 'b'}), AstStmtRenamer.get_defs_uses(parse_stmt('a.f = b')))
        self.assertEqual((set(), {'a', 'b', 'c'}), AstStmtRenamer.get_defs_uses(parse_stmt('a[c] = b')))
        self.assertEqual((set(), {'a', 'b', 'c'}), AstStmtRenamer.get_defs_uses(parse_stmt('a.f[c] = b')))

        self.assertEqual(({'a', 'b'}, {'c'}), AstStmtRenamer.get_defs_uses(parse_stmt('a,b = c')))
        self.assertEqual(({'a'}, {'b', 'c'}), AstStmtRenamer.get_defs_uses(parse_stmt('a, b.f = c')))
        self.assertEqual(({'b'}, {'a', 'i', 'c'}), AstStmtRenamer.get_defs_uses(parse_stmt('a[i], b = c')))

    def check_equal(self, a: ast.stmt, b: ast.stmt):
        self.assertEqual(astor.to_source(a), astor.to_source(b))

    def test_rename_uses_expr(self):
        exprs = ['a', '1', '(a, b)', '[a]', '{b}', '{a:b}', 'a+b', '-a', 'a == b',
                 'a.f', 'a[i]', 'a.f[i]', 'a[b.g]',
                 'f()', 'f(a)', 'f(a,b)', 'f(f(a), b)', 'f(a+b)', 'f(a,k=d)']

        for expr in exprs:
            uses = AstStmtRenamer.get_defs_uses(parse_stmt(expr))[1]
            for v in uses:
                v_ = f'{v}1'
                self.check_equal(parse_stmt(expr.replace(v, v_)), AstStmtRenamer.rename_uses(parse_stmt(expr), {v: v_}))
                self.check_equal(parse_stmt(expr), AstStmtRenamer.rename_uses(parse_stmt(expr), {'some_other_var': v_}))

    def test_rename_uses(self):
        self.check_equal(parse_stmt('assert 1'), AstStmtRenamer.rename_uses(parse_stmt('assert 1'), {'a': 'a0'}))
        self.check_equal(parse_stmt('assert a0'), AstStmtRenamer.rename_uses(parse_stmt('assert a'), {'a': 'a0'}))
        self.check_equal(parse_stmt('assert a'), AstStmtRenamer.rename_uses(parse_stmt('assert a'), {'b': 'b0'}))

        self.check_equal(parse_stmt('1'), AstStmtRenamer.rename_uses(parse_stmt('1'), {'a': 'a0'}))
        self.check_equal(parse_stmt('b'), AstStmtRenamer.rename_uses(parse_stmt('b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('b0'), AstStmtRenamer.rename_uses(parse_stmt('b'), {'b': 'b0'}))

        self.check_equal(parse_stmt('a = 1'), AstStmtRenamer.rename_uses(parse_stmt('a = 1'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a = b'), AstStmtRenamer.rename_uses(parse_stmt('a = b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a = b0'), AstStmtRenamer.rename_uses(parse_stmt('a = b'), {'b': 'b0'}))

        self.check_equal(parse_stmt('a: int = 1'), AstStmtRenamer.rename_uses(parse_stmt('a: int = 1'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a: int = b'), AstStmtRenamer.rename_uses(parse_stmt('a: int = b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a: int = b0'), AstStmtRenamer.rename_uses(parse_stmt('a: int = b'), {'b': 'b0'}))

    def test_rename_defs(self):
        self.check_equal(parse_stmt('assert 1'), AstStmtRenamer.rename_defs(parse_stmt('assert 1'), {'a': 'a0'}))
        self.check_equal(parse_stmt('assert a'), AstStmtRenamer.rename_defs(parse_stmt('assert a'), {'a': 'a0'}))
        self.check_equal(parse_stmt('assert a'), AstStmtRenamer.rename_defs(parse_stmt('assert a'), {'b': 'b0'}))

        self.check_equal(parse_stmt('a0 = 1'), AstStmtRenamer.rename_defs(parse_stmt('a = 1'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a0 = b'), AstStmtRenamer.rename_defs(parse_stmt('a = b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a0 = a'), AstStmtRenamer.rename_defs(parse_stmt('a = a'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a = 1'), AstStmtRenamer.rename_defs(parse_stmt('a = 1'), {'b': 'b0'}))
        self.check_equal(parse_stmt('a = b'), AstStmtRenamer.rename_defs(parse_stmt('a = b'), {'b': 'b0'}))

        self.check_equal(parse_stmt('a0: int = b'), AstStmtRenamer.rename_defs(parse_stmt('a: int = b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a: int = b'), AstStmtRenamer.rename_defs(parse_stmt('a: int = b'), {'b': 'b0'}))

        self.check_equal(parse_stmt('a.f = b'), AstStmtRenamer.rename_defs(parse_stmt('a.f = b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a.f = b'), AstStmtRenamer.rename_defs(parse_stmt('a.f = b'), {'b': 'b0'}))

        self.check_equal(parse_stmt('a[i] = b'), AstStmtRenamer.rename_defs(parse_stmt('a[i] = b'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a[i] = b'), AstStmtRenamer.rename_defs(parse_stmt('a[i] = b'), {'b': 'b0'}))
        self.check_equal(parse_stmt('a[i] = b'), AstStmtRenamer.rename_defs(parse_stmt('a[i] = b'), {'i': 'i0'}))

        self.check_equal(parse_stmt('a0, b = b, a'), AstStmtRenamer.rename_defs(parse_stmt('a, b = b, a'), {'a': 'a0'}))
        self.check_equal(parse_stmt('a, b0 = b, a'), AstStmtRenamer.rename_defs(parse_stmt('a, b = b, a'), {'b': 'b0'}))
        self.check_equal(parse_stmt('a0, b0 = b, a'), AstStmtRenamer.rename_defs(parse_stmt('a, b = b, a'), {'b': 'b0', 'a': 'a0'}))

        self.check_equal(parse_stmt('a0, b[i] = b, a'), AstStmtRenamer.rename_defs(parse_stmt('a, b[i] = b, a'), {'a': 'a0', 'b': 'b0'}))
        self.check_equal(parse_stmt('b.f, a0 = b, a'), AstStmtRenamer.rename_defs(parse_stmt('b.f, a = b, a'), {'a': 'a0', 'b': 'b0'}))

class SSAConvertorTest(unittest.TestCase):
    def test_make_SSA_cfg_builder(self):
        builder = CFGBuilder(labeler, bbf)
        code = '''
x = 0
y = 0
while a:
    if t:
        y = 1
        x = 0
    else:
        tmp = x
        x = y
        y = tmp
f(x,y)
        '''
        cfg = builder.make_cfg(parse_stmts(code))
        a,b = make_SSA(cfg)
        pass

    def test_make_SSA(self):
        cfg = CFG.make('r', 'E', [
            ('r', mk_block(''), ['A']),
            ('A', mk_block(''), ['B', 'C']),
            ('B', mk_block('y = 1\nx = 0'), ['D']),
            ('C', mk_block('tmp = x\nx = y\ny = tmp'), ['D', 'E']),
            ('D', mk_block('y = f(x, y)'), ['A', 'E']),
            ('E', mk_block('f(x, y)'), []),
        ])

        expected_instrs = {
            'r': [], 'A': [],
            'B': mk_instrs('y2 = 1\nx2 = 0'),
            'C': mk_instrs('tmp = x1\nx3 = y1\ny3 = tmp'),
            'D': mk_instrs('y5 = f(x4, y4)'),
            'E': mk_instrs('f(x5, y6)')
        }
        expected_phis = {
            'A': [PhiFunc('x1', {'D': 'x4', 'r': 'x'}), PhiFunc('y1', {'D': 'y5', 'r': 'y'})],
            'D': [PhiFunc('x4', {'B': 'x2', 'C': 'x3'}), PhiFunc('y4', {'B': 'y2', 'C': 'y3'})],
            'E': [PhiFunc('x5', {'D': 'x4', 'C': 'x3'}), PhiFunc('y6', {'D': 'y5', 'C': 'y3'})],
        }

        block_instrs, phi_funcs = make_SSA(cfg)

        self.assertEqual(block_instrs.keys(), block_instrs.keys())

        instr_places = extract_places_from_blocks(block_instrs)
        phi_places = extract_places_from_phis(phi_funcs)

        expected_instr_places = extract_places_from_blocks(expected_instrs)
        expected_phi_places = extract_places_from_phis(expected_phis)

        matches = match_places(self, instr_places | phi_places, expected_instr_places | expected_phi_places)
        unify(self, matches)


if __name__ == '__main__':
    unittest.main()
