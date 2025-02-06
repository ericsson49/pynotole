from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple, TypeVar

import ast
import astor


_T = TypeVar('_T')
_Strategy = Callable[[_T], _T|None]


class Rewriting(ABC):
    @classmethod
    @abstractmethod
    def compare_nodes(self, a, b) -> bool:
        ...

    @classmethod
    @abstractmethod
    def some(cls, rule: _Strategy) -> _Strategy:
        ...

    @classmethod
    @abstractmethod
    def all(cls, rule: _Strategy) -> _Strategy:
        ...

    @classmethod
    def _sanitize_rule(cls, rule: _Strategy) -> _Strategy:
        def r(t: _T) -> _T|None:
            res = rule(t)
            if res is not None and not cls.compare_nodes(t, res):
                return res
        return r

    @classmethod
    def repeat(cls, rule: _Strategy) -> _Strategy:
        rule_ = cls._sanitize_rule(rule)
        def f(t: _T) -> _T|None:
            curr, prev = rule_(t), None
            while curr is not None:
                curr, prev = rule_(curr), curr
            return prev
        return f

    @classmethod
    def plus(cls, r1: _Strategy, r2: _Strategy) -> _Strategy:
        def f(t: _T) -> _T|None:
            return res if (res := r1(t)) is not None else r2(t)
        return f

    @classmethod
    def seq(cls, r1: _Strategy, r2: _Strategy) -> _Strategy:
        def f(t: _T) -> _T|None:
            if (res := r1(t)) is not None:
                return r2(res)
        return f

    @classmethod
    def bottomup(cls, s: _Strategy) -> _Strategy:
        return cls.seq(cls.all(lambda t: cls.bottomup(s)(t)), s)

    @classmethod
    def topdown(cls, s: _Strategy) -> _Strategy:
        return cls.seq(s, cls.all(lambda t: cls.topdown(s)(t)))

    @classmethod
    def reduce(cls, rule: _Strategy) -> _Strategy:
        def x(t: _T) -> _T|None:
            return cls.plus(cls.some(x), rule)(t)
        return cls.repeat(x)


def rewrite(rule: _Strategy, t: _T) -> _T:
    return res if (res := rule(t)) is not None else t


def _flatten(elems):
    res = []
    for elem in elems:
        if isinstance(elem, list):
            res.extend(elem)
        else:
            res.append(elem)
    return res


class AstRewriting(Rewriting):
    @classmethod
    def compare_nodes(self, a, b) -> bool:
        return astor.dump_tree(a) == astor.dump_tree(b)

    @classmethod
    def _compare_seq(cls, ls1: list, ls2: list) -> bool:
        if len(ls1) != len(ls2):
            return False
        for e1, e2 in zip(ls1, ls2):
            if not cls.compare_nodes(e1, e2):
                return False
        return True

    @classmethod
    def some(cls, rule):
        def res(s):
            match s:
                case [stmt, *stmts]:
                    stmt_ = rule(stmt)
                    stmts_ = res(stmts)
                    if stmt_ is not None or stmts_ is not None:
                        r1 = stmt_ if stmt_ is not None else stmt
                        r2 = stmts_ if stmts_ is not None else stmts
                        stmts__ = _flatten([r1] + r2)
                        if not cls._compare_seq([stmt] + stmts, stmts__):
                            return stmts__
                case ast.If(test, body, orelse):
                    body_ = res(body)
                    orelse_ = res(orelse)
                    if body_ is not None or orelse_ is not None:
                        return ast.If(test=test,
                                      body=body_ if body_ is not None else body,
                                      orelse=orelse_ if orelse_ is not None else orelse)
                case ast.While(test, body, []):
                    body_ = res(body)
                    if body_ is not None:
                        return ast.While(test=test, body=body_, orelse=[])
                case ast.For(target, iter, body, []):
                    body_ = res(body)
                    if body_ is not None:
                        return ast.For(target, iter, body_, [])
        return res

    @classmethod
    def all(cls, rule):
        def f(s):
            match s:
                case [*stmts]:
                    stmts_ = list(map(rule, stmts))
                    if all(st is not None for st in stmts_):
                        return _flatten(stmts_)
                case ast.While(test, body, []):
                    body_ = f(body)
                    if body_ is not None:
                        return ast.While(test, body_, [])
                case ast.For(target, iter, body, []):
                    body_ = f(body)
                    if body_ is not None:
                        return ast.For(target, iter, body, [])
                case ast.If(test, body, orelse):
                    body_ = f(body)
                    orelse_ = f(orelse)
                    if body_ is not None or orelse_ is not None:
                        return ast.If(test,
                                      body_ if body_ is not None else body,
                                      orelse_ if orelse_ is not None else orelse)
                case _:
                    return s
        return f

