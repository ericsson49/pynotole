import ast

from toolz.dicttoolz import valmap

from .cfg import CFGBuilder, IndexingLabelFactory, PyASTBBlockFactory
from .ssa import make_SSA, calc_SSA_info, rename_blocks
from .ast_utils import replace_func_body


def split_var_ranges(f: ast.FunctionDef) -> ast.FunctionDef:
    cfg = CFGBuilder(IndexingLabelFactory(), PyASTBBlockFactory()).make_cfg(f.body)
    blocks, phi_funcs = make_SSA(cfg)

    mp = {}
    for to, phis in phi_funcs.items():
        for phi in phis:
            for fr, v in phi.uses.items():
                if fr not in mp:
                    mp[fr] = []
                mp[fr].append((phi.vdef, v))

    def extract_expr(idx):
        instrs = blocks[idx]
        assert len(instrs) == 1
        assert isinstance(instrs[0], ast.Expr)
        return instrs[0].value

    def extract_target(idx):
        instrs = blocks[idx]
        assert len(instrs) == 1
        assert isinstance(instrs[0], ast.Assign)
        assert len(instrs[0].targets) == 1
        return instrs[0].targets[0]

    def mk_phi_copies(idx):
        res = []
        if idx in mp:
            for to, fr in sorted(mp[idx]):
                res.append(ast.Assign([ast.Name(to)], ast.Name(fr)))
        return res

    def process_stmts(prefix, stmts):
        res = []
        for i, st in enumerate(stmts):
            match st:
                case ast.If(_, body, orelse):
                    test_ = extract_expr(prefix + (i, 'if-test'))
                    body_ = process_stmts(prefix + (i, 'if-then',), body)
                    if len(orelse) != 0:
                        orelse_ = process_stmts(prefix + (i, 'if-else',), orelse)
                    else:
                        if prefix + (i, 'if-else', 0) in mp:
                            orelse_ = mk_phi_copies(prefix + (i, 'if-else', 0))
                        else:
                            orelse_ = []
                    res.append(ast.If(test_, body_, orelse_))
                case ast.While(_, body, []):
                    test_ = extract_expr(prefix + (i, 'while-test'))
                    body_ = process_stmts(prefix + (i, 'while-body',), body)
                    res.append(ast.While(test_, body_, []))
                case ast.For(_, _, body, []):
                    res.extend(mk_phi_copies(prefix + (i, 'for-pre-head')))
                    target = extract_target(prefix + (i, 'for-pre-body'))
                    iter_ = extract_expr(prefix + (i, 'for-pre-head'))
                    body_ = process_stmts(prefix + (i, 'for-body'), body)
                    res.append(ast.For(target, iter_, body_, []))
                case ast.Break():
                    res.append(ast.Break())
                case ast.Continue():
                    res.extend(mk_phi_copies(prefix + (i,)))
                    res.append(ast.Continue())
                case ast.Return(None):
                    res.append(ast.Return(None))
                case ast.Return(_):
                    res.append(ast.Return(extract_expr(prefix + (i,))))
                case _:
                    idx = prefix + (i,)
                    instrs = blocks[idx]
                    assert len(instrs) == 1
                    res.append(instrs[0])
                    res.extend(mk_phi_copies(idx))
        return res

    body_ = process_stmts((), f.body)
    return replace_func_body(f, body_)


def get_phi_webs(phis):
    rep = {}
    phi_webs = {}
    def find(v):
        if v in rep:
            return rep[v]
        else:
            rep[v] = v
            phi_webs[v] = {v}
            return v
    def unify(vs):
        vs = list(vs)
        v = vs[0]
        v_ = find(v)
        for v2 in vs[1:]:
            v2_ = find(v2)
            if v_ != v2_:
                for vl in phi_webs[v2_]:
                    rep[vl] = v_
                    phi_webs[v_].add(vl)
                del phi_webs[v2_]
    for phi_funcs in phis.values():
        for phi in phi_funcs:
            unify(set(phi.uses.values()) | {phi.vdef})
    return phi_webs


def split_var_ranges2(f: ast.FunctionDef) -> ast.FunctionDef:
    cfg = CFGBuilder(IndexingLabelFactory(), PyASTBBlockFactory()).make_cfg(f.body)
    phi_funcs, renames = calc_SSA_info(cfg)

    phi_webs = get_phi_webs(phi_funcs)

    remap = {}
    for vs in phi_webs.values():
        v = min(vs, key=lambda v: (len(v), v))
        for v_ in vs:
            remap[v_] = v

    renames_ = {}
    for label in renames.keys():
        block_renames = []
        for vm1, vm2 in renames[label]:
            vm1_ = valmap(lambda v: remap.get(v, v), vm1)
            vm2_ = valmap(lambda v: remap.get(v, v), vm2)
            block_renames.append((vm1_, vm2_))
        renames_[label] = block_renames
    renames = renames_

    blocks = rename_blocks(cfg.blocks, renames)

    def extract_expr(idx):
        instrs = blocks[idx]
        assert len(instrs) == 1
        assert isinstance(instrs[0], ast.Expr)
        return instrs[0].value

    def extract_target(idx):
        instrs = blocks[idx]
        assert len(instrs) == 1
        assert isinstance(instrs[0], ast.Assign)
        assert len(instrs[0].targets) == 1
        return instrs[0].targets[0]

    def process_stmts(prefix, stmts):
        res = []
        for i, st in enumerate(stmts):
            match st:
                case ast.If(_, body, orelse):
                    test_ = extract_expr(prefix + (i, 'if-test'))
                    body_ = process_stmts(prefix + (i, 'if-then',), body)
                    if len(orelse) != 0:
                        orelse_ = process_stmts(prefix + (i, 'if-else',), orelse)
                    else:
                        orelse_ = []
                    res.append(ast.If(test_, body_, orelse_))
                case ast.While(_, body, []):
                    test_ = extract_expr(prefix + (i, 'while-test'))
                    body_ = process_stmts(prefix + (i, 'while-body',), body)
                    res.append(ast.While(test_, body_, []))
                case ast.For(_, _, body, []):
                    target = extract_target(prefix + (i, 'for-pre-body'))
                    iter_ = extract_expr(prefix + (i, 'for-pre-head'))
                    body_ = process_stmts(prefix + (i, 'for-body'), body)
                    res.append(ast.For(target, iter_, body_, []))
                case ast.Break():
                    res.append(ast.Break())
                case ast.Continue():
                    res.append(ast.Continue())
                case ast.Return(None):
                    res.append(ast.Return(None))
                case ast.Return(_):
                    res.append(ast.Return(extract_expr(prefix + (i,))))
                case _:
                    idx = prefix + (i,)
                    instrs = blocks[idx]
                    assert len(instrs) == 1
                    res.append(instrs[0])
        return res

    body_ = process_stmts((), f.body)
    return replace_func_body(f, body_)

