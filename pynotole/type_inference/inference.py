import ast
import astor

from mypy import nodes
from mypy.erasetype import remove_instance_last_known_values
from mypy.join import join_type_list
from mypy import types

from toolz.itertoolz import mapcat, groupby
from toolz.dicttoolz import valmap

from .mypy_helper import call_mypy
from .. import fail_if
from ..ast_utils import replace_func_body, process_funcs


def parse_fq_type_name(ty: ast.Attribute):
    match ty:
        case ast.Attribute(ast.Name(id), attr):
            return (id, attr)
        case ast.Attribute(ast.Attribute() as ty2, attr):
            return (parse_fq_type_name(ty2)) + (attr,)


def type_to_str(t) -> str:
    match t:
        case types.Instance() as inst:
            if inst.last_known_value is not None:
                assert False
            if len(inst.args) == 0:
                return inst.type.fullname
            else:
                return f"{inst.type.fullname}[{', '.join(type_to_str(a) for a in inst.args)}]"
        case types.TypeVarType() as tv:
            return tv.fullname
        case types.UninhabitedType():
            return 'Never'
        case types.AnyType():
            return 'Any'
        case types.TupleType() as tpl:
            return f"tuple[{', '.join(type_to_str(i) for i in tpl.items)}]"
        # case types.CallableType() as callable:
        #     return f"Callable[[{', '.join(type_to_str(i) for i in callable.arg_types)}], {type_to_str(callable.ret_type)}]"
    assert False


def _post_process_type_str(s):
    def convert(ty):
        match ty:
            case ast.Name() as n:
                return n
            case ast.Attribute(ast.Name('builtins' | '__main__'), attr, ctx):
                return ast.Name(attr, ctx)
            case ast.Attribute() as attr:
                parts = parse_fq_type_name(ty)
                if parts[0] in ['high_level', 'pyrsistent']:
                    return ast.Name(parts[-1], attr.ctx)
                else:
                    return attr
            case ast.Subscript(value, ast.Tuple(elts)):
                return ast.Subscript(convert(value), ast.Tuple([convert(elt) for elt in elts]))
            case ast.Subscript(value, slice):
                return ast.Subscript(convert(value), convert(slice))
            case _:
                assert False

    ty = convert(ast.parse(s, mode='eval').body)
    return ''.join(astor.to_source(ty, pretty_string=lambda x: x)).rstrip()


def gather_var_types(res):
    def process(n: nodes.Node) -> set[tuple[str, nodes.Expression]]:
        if isinstance(n, nodes.Block):
            body = n.body
            return set(mapcat(process, body))
        elif isinstance(n, nodes.AssignmentStmt) and len(n.lvalues) == 1 \
                and isinstance(n.lvalues[0], nodes.NameExpr) and n.type is None:
            assert n.rvalue is not None
            ne = n.lvalues[0]
            return {(ne.name, n.rvalue)}
        elif isinstance(n, nodes.IfStmt):
            if n.else_body is None:
                return set(mapcat(process, n.body))
            else:
                return set(mapcat(process, n.body + [n.else_body]))
        elif isinstance(n, nodes.WhileStmt):
            assert n.else_body is None
            return process(n.body)
        elif isinstance(n, nodes.ForStmt):
            fail_if(n.else_body is not None, 'for else is not supported')
            fail_if(not isinstance(n.index, nodes.NameExpr), 'only variables are supported as for a target')
            return process(n.body) | {(n.index.name, n.index)}
        else:
            return set()

    result = {}

    def get_expr_type(e: nodes.Expression):
        if isinstance(e, nodes.NameExpr) and e.node and isinstance(e.node, nodes.Var):
            return e.node.type
        else:
            return remove_instance_last_known_values(res.types[e])

    for d in res.files['__main__'].defs:
        if isinstance(d, (nodes.FuncDef, nodes.Decorator)):
            func = d.func if isinstance(d, nodes.Decorator) else d
            var_exprs = process(func.body)
            var_types = [(n, get_expr_type(e)) for n, e in var_exprs]

            if var_types:
                result[func.name] = valmap(
                    lambda vals: _post_process_type_str(type_to_str(join_type_list([v for k,v in vals]))),
                    groupby(0, var_types))
    return result


def select_errors(errs, suffix):
    res = []
    for err in errs:
        ps = err.split(':')
        if ps[2].endswith(suffix):
            res.append(err)
    return res


def _gather_ann_assigns(stmt):
    match stmt:
        case [*stmts]:
            return set(mapcat(_gather_ann_assigns, stmts))
        case ast.AnnAssign(ast.Name(v), _, _):
            return {v}
        case ast.AnnAssign():
            assert False
        case ast.If(_, body, orelse):
            return set(mapcat(_gather_ann_assigns, body + orelse))
        case ast.While(_, body, []):
            return set(mapcat(_gather_ann_assigns, body))
        case ast.For(_, _, body, []):
            return set(mapcat(_gather_ann_assigns, body))
        case _:
            return set()


def invoke_mypy(code, mypy_path):
    res = call_mypy(['-c', code], mypy_path=mypy_path, export_types=True)
    ignored_errors = {'[empty-body]', '[var-annotated]', '[assignment]'}
    errors = [err for err in select_errors(res.errors, 'error') if not any(tag in err for tag in ignored_errors)]
    if len(errors) != 0:
        assert False
    return res


def infer_variable_types(module_defs, mypy_path=None):
    code = astor.to_source(ast.Module(module_defs))
    res = invoke_mypy(code, mypy_path=mypy_path)
    return gather_var_types(res)


def prepend_stmts(defs: list[ast.stmt], body: list[ast.stmt]) -> list[ast.stmt]:
    match body:
        case [ast.Expr(ast.Constant(str())) as string_doc, *rest]:
            return [string_doc] + defs + rest
        case _:
            return defs + body

def infer_types_with_mypy(module_defs, mypy_path=None):
    func_var_types = infer_variable_types(module_defs, mypy_path)

    def prepend_var_defs(func: ast.FunctionDef) -> ast.FunctionDef:
        if func.name in func_var_types:
            annotated_vars = _gather_ann_assigns(func.body)
            typed_var_defs = [
                ast.AnnAssign(ast.Name(v, ast.Store()), ast.Name(ty, ast.Load()), None, 1)
                for v, ty in func_var_types[func.name].items() if v not in annotated_vars
            ]
            return replace_func_body(func, prepend_stmts(typed_var_defs, func.body))
        else:
            return func

    res_defs = process_funcs(module_defs, prepend_var_defs)
    return res_defs


