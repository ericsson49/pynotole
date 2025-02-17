import unittest
from toolz.dicttoolz import keymap, merge

from .graph_utils import *


def get_base_var(phi: PhiFunc) -> str:
    vs = {phi.vdef} | {v for v in phi.uses.values() if v is not None}
    min_len = min([len(v) for v in vs])
    for pl in range(min_len, 0, -1):
        vs_ = {v[:pl] for v in vs}
        if len(vs_) == 1:
            return list(vs_)[0].rstrip('_')
    assert False, f'no common prefix found for: {vs}'

def update_places(places: dict[str, str], prefix: str) -> dict[str, str]:
    return keymap(lambda k: f'{prefix}.{k}', places)

def extract_places_from_instr(i: SimpleInstr) -> dict[str, str]:
    return {'vdef': i.vdef} | {f'use{i}': v for i, v in enumerate(i.uses) if isinstance(v, str)}

def extract_places_from_instr_list(instrs: list[SimpleInstr]) -> dict[str, str]:
    return merge(*[update_places(extract_places_from_instr(instr), str(i)) for i, instr in enumerate(instrs)])

def extract_places_from_blocks(blocks: dict[str, list[SimpleInstr]]) -> dict[str, str]:
    return merge(*[update_places(extract_places_from_instr_list(instrs), b) for b, instrs in blocks.items()])

def extract_places_from_phi(phi: PhiFunc) -> dict[str, str]:
    bv = get_base_var(phi)
    return update_places({'vdef': phi.vdef} | {f'use.{b}': v for b, v in phi.uses.items()}, bv)

def extract_places_from_phi_list(phis: list[PhiFunc]) -> dict[str, str]:
    return merge(*[extract_places_from_phi(phi) for phi in phis])

def extract_places_from_phis(phis: dict[str, list[PhiFunc]]) -> dict[str, str]:
    return merge(*[update_places(extract_places_from_phi_list(ps), b) for b, ps in phis.items()])

def match_places(self: unittest.TestCase, places1: dict[str, str], places2: dict[str, str]):
    self.assertEqual(places1.keys(), places2.keys())
    return {(places1[k], places2[k]) for k in places1.keys()}

def unify(self: unittest.TestCase, pairs):
    pairs = set(pairs)
    p1 = {a for a, _ in pairs}
    p2 = {b for _, b in pairs}
    self.assertEqual(len(pairs), len(p1))
    self.assertEqual(len(pairs), len(p2))
    return dict(pairs)

