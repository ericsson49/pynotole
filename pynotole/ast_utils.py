import ast


def get_arg_names(args: ast.arguments):
    return [arg.arg for arg in args.args]


def replace_func_body(f: ast.FunctionDef, body_: list[ast.stmt]) -> ast.FunctionDef:
    return ast.FunctionDef(f.name, f.args, body_, f.decorator_list, f.returns)

