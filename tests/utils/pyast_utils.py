import ast


def parse_stmts(code: str) -> list[ast.stmt]:
    return ast.parse(code).body


def parse_stmt(code: str) -> ast.stmt:
    stmts = parse_stmts(code)
    assert len(stmts) == 1
    return stmts[0]
