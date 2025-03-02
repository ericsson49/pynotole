import ast

from toolz.itertoolz import concat
from typing import Callable, Sequence, Tuple


def get_arg_names(args: ast.arguments):
    return [arg.arg for arg in args.args]


def update_func(f: ast.FunctionDef, **kwargs):
    name = kwargs['name'] if 'name' in kwargs else f.name
    args = kwargs['args'] if 'args' in kwargs else f.args
    body = kwargs['body'] if 'body' in kwargs else f.body
    decorator_list = kwargs['decorator_list'] if 'decorator_list' in kwargs else f.decorator_list
    returns = kwargs['returns'] if 'returns' in kwargs else f.returns
    return ast.FunctionDef(name, args, body, decorator_list, returns)


def process_funcs(defs: list[ast.stmt], processor) -> list[ast.stmt]:
    return list(map(lambda s: processor(s) if isinstance(s, ast.FunctionDef) else s, defs))


def deconstruct_expr(e: ast.expr) -> Tuple[Sequence[ast.expr], Callable[[Sequence[ast.expr]],ast.expr]]:
    match e:
        case ast.Constant() | ast.Name():
            return [], lambda _: e
        case ast.Attribute(value, attr):
            return [value], lambda es: ast.Attribute(es[0], attr)
        case ast.Subscript(value, idx) if not (isinstance(idx, tuple)):
            return [value, idx], lambda es: ast.Subscript(es[0], es[1])
        case ast.BinOp(left, op, right):
            return [left, right], lambda es: ast.BinOp(es[0], op, es[1])
        case ast.BoolOp(op, values):
            return values, lambda es: ast.BoolOp(op, values)
        case ast.Compare(left, op, rights):
            return [left] + rights, lambda es: ast.Compare(es[0], op, es[1:])
        case ast.UnaryOp(op, value):
            return [value], lambda es: ast.UnaryOp(op, es[0])
        case ast.Call(func, args, kwargs):
            args_len = len(args)
            kws = [kw.arg for kw in kwargs]
            return ([func] + args + [kw.value for kw in kwargs],
                    lambda es: ast.Call(es[0], es[1:1+args_len], [ast.keyword(k, e) for k, e in zip(kws, es[1+args_len:1+args_len+len(kws)])]))
        case ast.IfExp(test, body, orelse):
            body_lam = ast.Lambda(ast.arguments([], []), body)
            else_lam = ast.Lambda(ast.arguments([], []), orelse)
            def reconstruct(es):
                test, body_lam, else_lam = es
                return ast.IfExp(test, body_lam.body, else_lam.body)
            return [test, body_lam, else_lam], reconstruct
        case ast.Tuple(elts):
            return elts, lambda es: ast.Tuple(es)
        case ast.List(elts):
            return elts, lambda es: ast.List(es)
        case ast.Set(elts):
            return elts, lambda es: ast.Set(es)
        case ast.Dict(keys, values):
            return list(concat(zip(keys,values))), lambda es: ast.Dict(es[:-1:2],es[1::2])
        case ast.GeneratorExp(elt, [ast.comprehension(ast.Name(tgt), iter, ifs)]):
            lam_args = [ast.arg(tgt, None)]
            def mk_lambda(body):
                return ast.Lambda(ast.arguments([], lam_args), body)
            sub_exprs = [iter, mk_lambda(elt)] + list(map(mk_lambda, ifs))
            def reconstruct(es):
                iter, map_lam, *if_lams = es
                comp = ast.comprehension(ast.Name(tgt), iter, [if_lam.body for if_lam in if_lams])
                return ast.GeneratorExp(map_lam.body, [comp])
            return sub_exprs, reconstruct
        case ast.ListComp(elt, generators):
            sub_exprs, gen_expr_builder = deconstruct_expr(ast.GeneratorExp(elt, generators))
            return sub_exprs, lambda es: ast.ListComp(*_extract_generator_exp_attrs(gen_expr_builder(es)))
        case ast.SetComp(elt, generators):
            sub_exprs, gen_expr_builder = deconstruct_expr(ast.GeneratorExp(elt, generators))
            return sub_exprs, lambda es: ast.SetComp(*_extract_generator_exp_attrs(gen_expr_builder(es)))
        case ast.DictComp(key, value, generators):
            sub_exprs, gen_expr_builder = deconstruct_expr(ast.GeneratorExp(ast.Tuple([key, value]), generators))
            def reconstruct(es):
                elt, generators = _extract_generator_exp_attrs(gen_expr_builder(es))
                match elt:
                    case ast.Tuple([k, v]):
                        return ast.DictComp(k, v, generators)
                    case _:
                        assert False
            return sub_exprs, reconstruct
        case _:
            assert False, e


def _extract_generator_exp_attrs(e: ast.expr):
    match e:
        case ast.GeneratorExp(elt, generators):
            return elt, generators
        case _:
            assert False

