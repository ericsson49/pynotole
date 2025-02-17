from io import StringIO
from mypy import build, main


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
