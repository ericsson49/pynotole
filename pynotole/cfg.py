import ast

from toolz.itertoolz import mapcat
from toolz.dicttoolz import dissoc
from typing import Any, Collection, Iterable, Generic, Mapping, Never, TypeAlias, TypeVar, Union

from .graph import Graph
from .ast_utils import deconstruct_expr, extract_load_exprs_from_stmt, get_arg_names


AST = TypeVar('AST')
B = TypeVar('B')
L: TypeAlias = Any
N = TypeVar('N')
T = TypeVar('T')


class LabelFactory:

    def entry_label(self) -> L: ...

    def exit_label(self) -> L: ...

    def child_label(self, n: L, child: int | str) -> L: ...


class IndexingLabelFactory(LabelFactory):
    def entry_label(self) -> tuple:
        return ()

    def exit_label(self) -> tuple:
        return 'exit'

    def child_label(self, n: tuple, child: int | str):
        return n + (child,)


class SimpleLabelFactory(LabelFactory):
    def __init__(self):
        self.labels = set()

    def entry_label(self) -> str:
        return 'entry'

    def exit_label(self) -> str:
        return 'exit'

    def child_label(self, n: str, child: int | str) -> str:
        label = f'L_{len(self.labels)}'
        self.labels.add(label)
        return label


class InstructionRenamer(Generic[T]):
    @classmethod
    def get_defs_uses(cls, instr: T) -> tuple[Collection[str], Collection[str]]:
        ...

    @classmethod
    def rename_defs(cls, instr: T, var_map: Mapping[str, str]) -> T:
        ...

    @classmethod
    def rename_uses(cls, instr: T, var_map: Mapping[str, str]) -> T:
        ...


class BBlock(Generic[T]):
    def __init__(self, instr_renamer: type[InstructionRenamer[T]], instrs: list[T]):
        self.instr_renamer = instr_renamer
        assert isinstance(instrs, list)
        self._instrs = instrs

    def __len__(self) -> int:
        return len(self._instrs)

    def get_instr_defs_uses(self, i: int) -> tuple[Collection[str], Collection[str]]:
        return self.instr_renamer.get_defs_uses(self._instrs[i])

    def get_defs_uses(self) -> tuple[Collection[str],Collection[str]]:
        defs = set()
        for i in range(len(self)):
            i_defs, _ = self.get_instr_defs_uses(i)
            defs |= i_defs
        uses = set()
        for i in reversed(range(len(self))):
            kill, gen = self.get_instr_defs_uses(i)
            uses = (uses - set(kill)) | set(gen)
        return defs, uses

    def rename_instr_uses(self, i: int, var_map: Mapping[str, str]):
         self._instrs[i] = self.instr_renamer.rename_uses(self._instrs[i], var_map)

    def rename_instr_defs(self, i: int, var_map: Mapping[str, str]):
        self._instrs[i] = self.instr_renamer.rename_defs(self._instrs[i], var_map)


class BBlockFactory(Generic[AST]):
    def unapply_block(self, n: AST) -> None | list[AST]: ...
    def unapply_while(self, n: AST) -> None | tuple[BBlock[AST], list[AST]]: ...
    def unapply_for(self, n: AST) -> None | tuple[BBlock[AST], BBlock, list[AST]]: ...
    def unapply_if(self, n: AST) -> None | tuple[BBlock[AST], list[AST], list[AST]]: ...
    def unapply_break(self, n: AST) -> None | tuple[()]: ...
    def unapply_continue(self, n: AST) -> None | tuple[()]: ...
    def unapply_return(self, n: AST) -> None | tuple[()] | BBlock[AST]: ...
    def unapply_simple(self, n: AST) -> BBlock[AST]: ...


class CFG(Graph[N], Generic[N,B]):
    def __init__(self, entry, exit, edges: Iterable[tuple[N, N]], blocks: dict[N, B]):
        super().__init__(edges)
        assert self.get_dom_helper().start == entry
        self.entry = entry
        self.exit = exit
        self.blocks = blocks

    @classmethod
    def make(cls, entry, exit, nodes: Iterable[tuple[N, B, Iterable[N]]]):
        edges = [(n, s) for n, _, succs in nodes for s in succs]
        blocks = {n: b for n, b, _ in nodes}
        return CFG(entry, exit, edges, blocks)

    def get_defs(self, n: N) -> Iterable[str]:
        if n in self.blocks:
            return self.blocks[n].get_defs_uses()[0]
        else:
            return set()

    def get_uses(self, n: N) -> Iterable[str]:
        if n in self.blocks:
            return self.blocks[n].get_defs_uses()[1]
        else:
            return set()


class _DummyRenamer(InstructionRenamer[Never]):
    pass


class DummyBlock(BBlock[Never]):
    def __init__(self):
        super().__init__(_DummyRenamer, [])

    def get_defs_uses(self) -> tuple[Collection[str],Collection[str]]:
        return set(), set()


class CFGBuilder(Generic[AST]):
    def __init__(self, labeler: LabelFactory, bblock_factory: BBlockFactory[AST]):
        self.labeler = labeler
        self.bblock_factory = bblock_factory
        self.loops: list[tuple[L, L]] = []
        self.nodes: dict[L, BBlock[AST]] = {}
        self.edges: list[tuple[L, L]] = []

    def get_entry(self): return self.labeler.entry_label()

    def get_exit(self): return self.labeler.exit_label()

    def get_label(self, parent: L, child: str|int) -> L:
        return self.labeler.child_label(parent, child)

    def add_node(self, block_id: L, block: BBlock) -> L:
        assert block is not None
        self.nodes[block_id] = block
        return block_id

    def add_edge(self, frm: object, to: object):
        self.edges.append((frm, to))

    def add_block(self, block_id: L, block: BBlock, to: L) -> L:
        self.add_node(block_id, block)
        self.add_edge(block_id, to)
        return block_id

    def on_stmt(self, idx: L, s: AST, nxt: L) -> L:
        bbf = self.bblock_factory

        if (st := bbf.unapply_block(s)) is not None:
            stmts = st
            if len(stmts) == 0:
                return nxt
            else:
                indices = [self.get_label(idx, i) for i in range(len(stmts))]
                cont = nxt
                for idx, stmt in reversed(list(zip(indices, stmts))):
                    cont = self.on_stmt(idx, stmt, cont)
                return cont
        elif st := bbf.unapply_while(s):
            test, body = st
            loop_head = self.add_node(self.get_label(idx, 'while-test'), test)
            self.loops.append((loop_head, nxt))
            body_label = self.on_stmt(self.get_label(idx, 'while-body'), body, loop_head)
            del self.loops[-1]
            self.add_edge(loop_head, body_label)
            self.add_edge(loop_head, nxt)
            return loop_head
        elif st := bbf.unapply_for(s):
            pre_head, pre_body, body = st
            test = DummyBlock()
            loop_head = self.add_node(self.get_label(idx, 'for-test'), test)
            self.loops.append((loop_head, nxt))
            body_label = self.on_stmt(self.get_label(idx, 'for-body'), body, loop_head)
            del self.loops[-1]
            pre_body_label = self.add_node(self.get_label(idx, 'for-pre-body'), pre_body)
            self.add_edge(pre_body_label, body_label)
            self.add_edge(loop_head, pre_body_label)
            self.add_edge(loop_head, nxt)
            return self.add_block(self.get_label(idx, 'for-pre-head'), pre_head, loop_head)
        elif st := bbf.unapply_if(s):
            test, body, orelse = st
            if_head = self.add_node(self.get_label(idx, 'if-test'), test)
            assert len(body) > 0
            body_entry = self.on_stmt(self.get_label(idx, 'if-then'), body, nxt)
            if len(orelse) > 0:
                else_entry = self.on_stmt(self.get_label(idx, 'if-else'), orelse, nxt)
            else:
                else_entry = self.add_block(
                    self.get_label(self.get_label(idx, 'if-else'), 0), DummyBlock(), nxt)
            self.add_edge(if_head, body_entry)
            self.add_edge(if_head, else_entry)
            return if_head
        elif (st := bbf.unapply_break(s)) is not None:
            return self.add_block(idx, DummyBlock(), self.loops[-1][1])
        elif (st := bbf.unapply_continue(s)) is not None:
            return self.add_block(idx, DummyBlock(), self.loops[-1][0])
        elif (st := bbf.unapply_return(s)) is not None:
            if st == ():
                return self.get_exit()
            else:
                assert isinstance(st, BBlock)
                return self.add_block(idx, st, self.get_exit())
        else:
            block = bbf.unapply_simple(s)
            return self.add_block(idx, block, nxt)

    def build(self):
        return CFG(self.get_entry(), self.get_exit(), self.edges, self.nodes)

    def make_cfg(self, body: AST):
        entry = self.add_node(self.labeler.entry_label(), DummyBlock())
        exit_ = self.add_node(self.labeler.exit_label(), DummyBlock())
        first_l = self.on_stmt(entry, body, exit_)
        self.add_edge(entry, first_l)
        return self.build()


PyAST: TypeAlias = Union[ast.stmt, list[ast.stmt]]


def _get_uses(e: ast.expr) -> set[str]:
    match e:
        case ast.Name(v):
            return {v}
        case ast.Lambda(args, body):
            return _get_uses(body) - set(get_arg_names(args))
        case _:
            sub_exprs, _ = deconstruct_expr(e)
            return set(mapcat(_get_uses, sub_exprs))


def _rename_uses(e: ast.expr, var_map: Mapping[str, str]) -> ast.expr:
    match e:
        case ast.Name(v) if v in var_map:
            return ast.Name(var_map[v])
        case ast.Lambda(args, body):
            var_map_ = dissoc(var_map, *get_arg_names(args))
            body_ = _rename_uses(body, var_map_)
            return ast.Lambda(args, body_)
        case _:
            sub_exprs, builder = deconstruct_expr(e)
            renamed_sub_exprs = [_rename_uses(e, var_map) for e in sub_exprs]
            return builder(renamed_sub_exprs)


class AstStmtRenamer(InstructionRenamer[ast.stmt]):
    @classmethod
    def get_defs_uses(cls, instr: ast.stmt) -> tuple[Collection[str], Collection[str]]:
        def get_target_defs_uses(e: ast.expr):
            match e:
                case ast.Name(tgt):
                    return {tgt}, set()
                case ast.Attribute(e, _):
                    return set(), _get_uses(e)
                case ast.Subscript(e, idx) if not isinstance(idx, (ast.Tuple, ast.slice)):
                    return set(), _get_uses(e) | _get_uses(idx)
                case ast.Tuple(elts):
                    defs, uses = set(), set()
                    for e in elts:
                        d, u = get_target_defs_uses(e)
                        defs |= d
                        uses |= u
                    return defs, uses
                case _:
                    assert False
        match instr:
            case ast.Assert(test, None):
                return (set(), _get_uses(test))
            case ast.Assign([tgt], value):
                defs, uses = get_target_defs_uses(tgt)
                return defs, uses | _get_uses(value)
            case ast.Assign([ast.Name(target)], value):
                return ({target}, _get_uses(value))
            case ast.Assign([ast.Attribute(v, _)], value):
                return (set(), list(mapcat(_get_uses, [v, value])))
            case ast.Assign([ast.Subscript(tgt, idx)], value) if not isinstance(idx, (ast.Tuple, ast.slice)):
                return (set(), list(mapcat(_get_uses, [tgt, value])))
            case ast.AnnAssign(ast.Name(target), _, None):
                return (set(), set())
            case ast.AnnAssign(ast.Name(target), _, value):
                return ({target}, _get_uses(value))
            case ast.Expr(value):
                return (set(), _get_uses(value))
            case _:
                assert False, instr

    @classmethod
    def rename_defs(cls, instr: ast.stmt, var_map: Mapping[str, str]) -> ast.stmt:
        def rename_target(e: ast.expr):
            match e:
                case ast.Name(v):
                    return ast.Name(var_map.get(v, v))
                case ast.Attribute() | ast.Subscript():
                    return e
                case ast.Tuple(elts):
                    return ast.Tuple([rename_target(e) for e in elts])
                case _:
                    assert False
        match instr:
            case ast.Assert():
                return instr
            case ast.Assign([target], value):
                return ast.Assign([rename_target(target)], value)
            case ast.AnnAssign(ast.Name() as target, anno, value, is_simple):
                return ast.AnnAssign(rename_target(target), anno, value, is_simple)
            case ast.Expr():
                return instr
            case _:
                assert False, instr

    @classmethod
    def rename_uses(cls, instr: ast.stmt, var_map: Mapping[str, str]) -> ast.stmt:
        exprs, builder = extract_load_exprs_from_stmt(instr)
        exprs_ = [_rename_uses(e, var_map) for e in exprs]
        return builder(exprs_)


class PyASTBBlockFactory(BBlockFactory[PyAST]):
    def block_from_stmt(self, stmt):
        return BBlock(AstStmtRenamer, [stmt])

    def block_from_expr(self, expr):
        return BBlock(AstStmtRenamer, [ast.Expr(expr)])

    def unapply_block(self, n: PyAST) -> None | list[PyAST]:
        match n:
            case [*stmts]:
                return stmts

    def unapply_while(self, n: PyAST) -> None | tuple[BBlock, list[PyAST]]:
        match n:
            case ast.While(test, body, []):
                test_block = self.block_from_expr(test)
                return test_block, body

    def unapply_for(self, n: PyAST) -> None | tuple[BBlock, BBlock, list[PyAST]]:
        match n:
            case ast.For(ast.Name(target), iter, body, []):
                pre_head = self.block_from_expr(iter)
                pre_body = self.block_from_stmt(ast.Assign([ast.Name(target)], ast.Constant(None)))
                return pre_head, pre_body, body
            case ast.For():
                assert False

    def unapply_if(self, n: PyAST) -> None | tuple[BBlock, list[PyAST], list[PyAST]]:
        match n:
            case ast.If(test, body, orelse):
                return self.block_from_expr(test), body, orelse

    def unapply_break(self, n: PyAST) -> None | tuple:
        match n:
            case ast.Break():
                return ()

    def unapply_continue(self, n: PyAST) -> None | tuple:
        match n:
            case ast.Continue():
                return ()

    def unapply_return(self, n: PyAST) -> None | tuple[()] | BBlock:
        match n:
            case ast.Return(None):
                return ()
            case ast.Return(value):
                return self.block_from_expr(value)

    def unapply_simple(self, n: PyAST) -> BBlock:
        match n:
            case ast.Assert() | ast.Assign() | ast.AnnAssign() | ast.Expr():
                return self.block_from_stmt(n)
            case ast.Pass():
                return DummyBlock()
            case _:
                assert False, n
