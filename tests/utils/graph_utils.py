import ast

from pynotole.cfg import *
from pynotole.ssa import *
from pynotole.graph import *


def parse_stmts(code: str) -> list[ast.stmt]:
    return ast.parse(code).body


def parse_stmt(code: str) -> ast.stmt:
    stmts = parse_stmts(code)
    assert len(stmts) == 1
    return stmts[0]


@dataclass(eq=True, frozen=True)
class SimpleInstr:
    vdef: str | None
    fid: str
    uses: tuple[str | int, ...]

    def __repr__(self) -> str:
        uses = ', '.join(map(str, self.uses))
        value = f'{self.fid}({uses})'
        if self.vdef is not None:
            return f'{self.vdef} := {value}'
        else:
            return value


class SimpleInstructionRenamer(InstructionRenamer):
    @classmethod
    def get_defs_uses(cls, instr: SimpleInstr) -> tuple[Collection[str], Collection[str]]:
        defs = {instr.vdef} if instr.vdef is not None else set()
        uses = [v for v in instr.uses if isinstance(v, str)]
        return defs, uses

    @classmethod
    def rename_defs(cls, instr: SimpleInstr, var_map: Mapping[str, str]) -> SimpleInstr:
        vdef_ = var_map.get(instr.vdef, instr.vdef)
        return replace(instr, vdef=vdef_)

    @classmethod
    def rename_uses(cls, instr: SimpleInstr, var_map: Mapping[str, str]) -> SimpleInstr:
        uses_ = tuple(var_map.get(v, v) for v in instr.uses)
        return replace(instr, uses=uses_)


def mk_instrs(s: str) -> list[SimpleInstr]:
    def convert_arg(a: ast.expr) -> str | int:
        match a:
            case ast.Constant(int(n)):
                return n
            case ast.Name(x):
                return x
            case _:
                assert False

    def convert_value(e: ast.expr) -> tuple[str, tuple[str | int, ...]]:
        match e:
            case ast.Constant(int(n)):
                return 'id', (n,)
            case ast.Name(x):
                return 'id', (x,)
            case ast.Call(ast.Name(f), args, []):
                return f, tuple(convert_arg(a) for a in args)
            case _:
                assert False

    def convert_stmt(s: ast.stmt):
        match s:
            case ast.Expr(value):
                fid, args = convert_value(value)
                return SimpleInstr(None, fid, args)
            case ast.Assign([ast.Name(v)], value):
                fid, args = convert_value(value)
                return SimpleInstr(v, fid, args)
            case _:
                assert False

    return [convert_stmt(stmt) for stmt in parse_stmts(s)]


def mk_block(instrs: str) -> BBlock[SimpleInstr]:
    return BBlock(SimpleInstructionRenamer, mk_instrs(instrs))

