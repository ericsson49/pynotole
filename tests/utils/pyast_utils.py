import ast


def parse_stmt(s):
    stmts = ast.parse(s).body
    assert len(stmts) == 1
    return stmts[0]
