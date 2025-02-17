import networkx as nx

from dataclasses import dataclass, replace

from toolz.itertoolz import *
from typing import Iterable, TypeAlias, TypeVar

from . import fail
from .cfg import CFG


class RenamerFactory:
    def __init__(self, cfg, outer_defs):
        self.dom_helper = cfg.get_dom_helper()
        self.varCounts = {p: 1 for p in outer_defs}
        start = self.dom_helper.start
        self.rd = {v: (v, start, -1) for v in outer_defs}

    def fresh(self, v):
        cnt = self.varCounts.get(v, 0)
        self.varCounts[v] = cnt + 1
        return f'{v}_{cnt}'

    def dominates(self, i1, i2):
        b1, id1 = i1
        b2, id2 = i2
        if b1 == b2:
            return id2 >= id1
        else:
            return self.dom_helper.dominates(b1, b2)

    def update_reaching_def(self, ii, v):
        assert ii is not None
        r = self.rd.get(v)
        while not (r is None or self.dominates(r[1:], ii)):
            r = self.rd.get(r[0])
        if r is not None:
            self.rd[v] = r

    def mk_renamer(self, ii) -> 'Renamer':
        return Renamer(self, ii)

    def correct_name(self, v, v_):
        return v if v + '_0' == v_ else v_

    def new_def_name(self, ii, v):
        self.update_reaching_def(ii, v)
        v_ = self.fresh(v)
        if self.rd.get(v) is not None:
            self.rd[v_] = self.rd.get(v)
        self.rd[v] = (v_, *ii)
        return self.correct_name(v, v_)

    def new_use_name(self, ii, v):
        self.update_reaching_def(ii, v)
        res = self.rd.get(v)
        if res is not None:
            return self.correct_name(v, res[0])
        else:
            fail(f'trying to use {v} before definition')


class Renamer:
    def __init__(self, renamer_factory: RenamerFactory, ii):
        self.renamer_factory = renamer_factory
        self.ii = ii

    def new_def_name(self, v: str) -> str:
        return self.renamer_factory.new_def_name(self.ii, v)

    def rename_defs(self, defs: set[str]) -> dict[str, str]:
        return {v: self.new_def_name(v) for v in defs}

    def new_use_name(self, v: str) -> str:
        return self.renamer_factory.new_use_name(self.ii, v)

    def rename_uses(self, uses: set[str]) -> dict[str, str]:
        return {v: self.new_use_name(v) for v in uses}


@dataclass
class PhiFunc:
    vdef: str
    uses: dict[object, str]

    def __repr__(self) -> str:
        uses = ', '.join(map(str, self.uses.values()))
        return f'{self.vdef} := \u03d5({uses})'


def make_phi_funcs(cfg, live_vars):
    phis = {}
    for n, vs in phi_nodes(cfg, live_vars).items():
        preds = cfg.get_preds(n)
        phis[n] = [PhiFunc(v, {p: v for p in preds}) for v in vs]
    return phis


def _fixp(f):
    def g(xs):
        return xs if (xs_ := f(xs)) == xs else g(xs_)
    return g


N = TypeVar('N')
V: TypeAlias = str


def live_vars(cfg: CFG[N, V]) -> dict[N, set[V]]:
    def step(ins: dict[N, set[N]]):
        res = ins.copy()
        for n in cfg.get_nodes():
            out: set[V] = set()
            for s in cfg.get_succs(n):
                out |= ins.get(s, set())
            gen, kill = set(cfg.get_uses(n)), set(cfg.get_defs(n))
            in_ = (out - kill) | gen
            res[n] = res.get(n, set()) | in_
        return res
    return _fixp(step)({})


def phi_nodes(cfg: CFG[N,V], live_vs=None) -> dict[N, set[V]]:
    df = cfg.get_dom_helper().dom_frontier

    def get_df(ns: Iterable[N]) -> set[N]:
        return set(mapcat(lambda n: df.get(n, set()), ns))

    def get_idf(ns: Iterable[N]) -> set[N]:
        return _fixp(lambda xs: xs | get_df(xs))(get_df(ns))

    res: dict[N, set[V]] = {}
    for n in cfg.get_nodes():
        defs = set(cfg.get_defs(n))
        for m in get_idf({n}):
            defs_ = defs.intersection(live_vs.get(m)) if live_vs is not None else defs
            if defs_:
                res[m] = res.get(m, set()) | defs_
    return res


def calc_SSA_info(cfg):
    dom_helper = cfg.get_dom_helper()
    live_vs = live_vars(cfg)
    phis = make_phi_funcs(cfg, live_vs)
    renamer_factory = RenamerFactory(cfg, live_vs[cfg.entry])

    def rename_phi_def(phi, var_map):
        v = phi.vdef
        phi.vdef = var_map.get(v, v)

    def rename_phi_use(phi, bb, var_map):
        v = phi.uses[bb]
        phi.uses[bb] = var_map.get(v, v)

    def get_phis_by_def(bb):
        return phis.get(bb, [])

    def get_phis_by_use(bb):
        return filter(lambda phi: bb in phi.uses, concat(phis.values()))

    renames: dict[object, list[(dict[str, str], dict[str, str])]] = {}

    for bb in dom_helper.dom_tree_preorder():
        phi_def_ii = (bb, -1)
        var_renamer = renamer_factory.mk_renamer(phi_def_ii)
        for phi in get_phis_by_def(bb):
            rename_phi_def(phi, var_renamer.rename_defs({phi.vdef}))

        block = cfg.blocks[bb]
        block_renames: list[(dict[str, str], dict[str, str])] = []
        for idx in range(len(block)):
            defs, uses = block.get_instr_defs_uses(idx)
            var_renamer = renamer_factory.mk_renamer((bb, idx))
            uses_var_map = var_renamer.rename_uses(uses)
            defs_var_map = var_renamer.rename_defs(defs)
            block_renames.append((defs_var_map, uses_var_map))
        renames[bb] = block_renames

        phi_use_ii = (bb, len(block))
        var_renamer = renamer_factory.mk_renamer(phi_use_ii)
        for phi in get_phis_by_use(bb):
            rename_phi_use(phi, bb, var_renamer.rename_uses({phi.uses[bb]}))

    return phis, renames


def rename_blocks(blocks, renames):
    instrs = {}
    for label, bb in blocks.items():
        block_renames = renames[label]
        block_instrs = []
        for i in range(len(bb)):
            bb.rename_instr_uses(i, block_renames[i][1])
            bb.rename_instr_defs(i, block_renames[i][0])
            block_instrs.append(bb._instrs[i])
        instrs[label] = block_instrs
    return instrs


def make_SSA(cfg):
    phis, renames = calc_SSA_info(cfg)
    return rename_blocks(cfg.blocks, renames), phis
