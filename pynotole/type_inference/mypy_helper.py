from io import StringIO
from mypy import build, main, nodes


def call_mypy(args, mypy_path:str=None, export_types=False, plugin_class=None):
    stdout = StringIO()
    stderr = StringIO()

    fscache = main.FileSystemCache()
    sources, opts = main.process_options(args, stdout=stdout, stderr=stderr, fscache=fscache)
    if export_types:
        opts.export_types = True
        opts.preserve_asts = True
    if mypy_path is not None:
        opts.mypy_path.append(mypy_path)

    if plugin_class is not None:
        plugins = [plugin_class(opts)]
    else:
        plugins = None

    res = build.build(sources, opts, None, None, fscache, stdout, stderr, extra_plugins=plugins)
    return res


def mypy_parse(code: str, mypy_path:str=None):
    args = ["-c", code]
    return call_mypy(args, mypy_path=mypy_path, export_types=True)


def deconstruct(e: nodes.Expression) -> list[nodes.Expression]:
    if isinstance(e, (nodes.IntExpr, nodes.NameExpr)):
        return []
    elif isinstance(e, nodes.MemberExpr):
        return [e.expr]
    elif isinstance(e, nodes.IndexExpr):
        assert e.analyzed is None
        assert not isinstance(e.index, nodes.SliceExpr)
        return [e.base, e.index]
    elif isinstance(e, nodes.CallExpr):
        return [e.callee, *e.args]
    elif isinstance(e, nodes.UnaryExpr):
        return [e.expr]
    elif isinstance(e, nodes.OpExpr):
        return [e.left, e.right]
    elif isinstance(e, nodes.ComparisonExpr):
        return e.operands[:]
    elif isinstance(e, nodes.TupleExpr):
        return e.items[:]
    elif isinstance(e, nodes.ListExpr):
        return e.items[:]
    elif isinstance(e, nodes.SetExpr):
        return e.items[:]
    elif isinstance(e, nodes.DictExpr):
        return [elem for item in e.items for elem in item]
    elif isinstance(e, nodes.ConditionalExpr):
        return [e.cond, e.if_expr, e.else_expr]
    else:
        assert False

