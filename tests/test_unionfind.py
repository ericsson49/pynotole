import unittest

from pynotole.unionfind import UnionFind

class UnionFindTestCase(unittest.TestCase):
    def test_find(self):
        uf = UnionFind()
        self.assertEqual('a', uf.find('a'))
        comp, = uf.components()
        self.assertEqual({'a'}, comp)
        self.assertEqual('b', uf.find('b'))
        comp1, comp2 = uf.components()
        self.assertEqual({frozenset(comp1), frozenset(comp2)}, {frozenset({'a'}), frozenset({'b'})})
        self.assertEqual('c', uf.find('c'))
        comp1, comp2, comp3 = uf.components()
        self.assertEqual({frozenset(comp1), frozenset(comp2), frozenset(comp3)},
                         {frozenset({'a'}), frozenset({'b'}), frozenset({'c'})})

    def test_union(self):
        uf = UnionFind()
        uf.union('a', 'b')
        comp, = uf.components()
        self.assertEqual(uf.find('a'), uf.find('b'))
        self.assertEqual(comp, {'a', 'b'})
        uf.union('c', 'd')
        comp1, comp2 = uf.components()
        self.assertEqual(uf.find('a'), uf.find('b'))
        self.assertEqual(uf.find('c'), uf.find('d'))
        self.assertNotEqual(uf.find('a'), uf.find('c'))
        self.assertEqual({frozenset(comp1), frozenset(comp2)}, {frozenset({'a', 'b'}), frozenset({'c', 'd'})})


if __name__ == '__main__':
    unittest.main()
