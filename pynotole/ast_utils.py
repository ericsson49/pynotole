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


def _deconstruct_target(target):
    if isinstance(target, ast.Tuple):
        elt_sub_exprs, elt_builders = [], []
        for elt in target.elts:
            sub_exprs, elt_builder = _deconstruct_target(elt)
            elt_sub_exprs.append(sub_exprs)
            elt_builders.append(elt_builder)
        def builder(exprs):
            curr_exprs = exprs
            elts_ = []
            for i, b in enumerate(elt_builders):
                sub_exprs = curr_exprs[:len(elt_sub_exprs[i])]
                curr_exprs = curr_exprs[len(elt_sub_exprs[i]):]
                elts_.append(b(sub_exprs))
            return ast.Tuple(elts_, ast.Load())
        return list(concat(elt_sub_exprs)), builder
    else:
        return deconstruct_expr(target)


def extract_load_exprs_from_stmt(s: ast.stmt):
    match s:
        case ast.Assert(test, None):
            return [test], lambda exprs: ast.Assert(exprs[0], None)
        case ast.Assert(test, msg):
            return [test, msg], lambda exprs: ast.Assert(exprs[0], exprs[1])
        case ast.Expr(value):
            return [value], lambda exprs: ast.Expr(exprs[0])
        case ast.Assign([target], value):
            sub_exprs, builder = _deconstruct_target(target)
            return [*sub_exprs, value], lambda exprs: ast.Assign([builder(exprs[:len(sub_exprs)])], exprs[len(sub_exprs)])
        case ast.AnnAssign(target, anno, None, is_simple):
            target_sub_exprs, target_builder = deconstruct_expr(target)
            return target_sub_exprs, lambda exprs: ast.AnnAssign(target_builder(exprs), anno, None, is_simple)
        case ast.AnnAssign(target, anno, value, is_simple):
            target_sub_exprs, target_builder = deconstruct_expr(target)
            return [*target_sub_exprs, value], lambda exprs: ast.AnnAssign(target_builder(exprs[:-1]), anno, exprs[-1], is_simple)
        case ast.If(test, body, orelse):
            return [test], lambda exprs: ast.If(exprs[0], body, orelse)
        case ast.While(test, body, orelse):
            return [test], lambda exprs: ast.While(exprs[0], body, orelse)
        case _:
            assert False

