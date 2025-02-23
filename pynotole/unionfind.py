class UnionFind:
    def __init__(self):
        self._representative_of = {}
        self._components = {}

    def find(self, v):
        if v in self._representative_of:
            return self._representative_of[v]
        else:
            self._representative_of[v] = v
            self._components[v] = {v}
            return v

    def union(self, a, b):
        a_rep = self.find(a)
        b_rep = self.find(b)
        if a_rep != b_rep:
            for v in self._components[b_rep]:
                self._representative_of[v] = a_rep
                self._components[a_rep].add(v)
            del self._components[b_rep]

    def components(self):
        return self._components.values()

