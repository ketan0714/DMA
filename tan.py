import math
from collections import Counter
from itertools import combinations

# buys_computer dataset (14 rows): features = age, income, student, credit_rating; class = buys_computer
DATA = [
    ("<=30", "high", "no", "fair", "no"),
    ("<=30", "high", "no", "excellent", "no"),
    ("31-40", "high", "no", "fair", "yes"),
    (">40", "medium", "no", "fair", "yes"),
    (">40", "low", "yes", "fair", "yes"),
    (">40", "low", "yes", "excellent", "no"),
    ("31-40", "low", "yes", "excellent", "yes"),
    ("<=30", "medium", "no", "fair", "no"),
    ("<=30", "low", "yes", "fair", "yes"),
    (">40", "medium", "yes", "fair", "yes"),
    ("<=30", "medium", "yes", "excellent", "yes"),
    ("31-40", "medium", "no", "excellent", "yes"),
    ("31-40", "high", "yes", "fair", "yes"),
    (">40", "medium", "no", "excellent", "no"),
]
FEATURES = ["age", "income", "student", "credit_rating"]
X_NEW = {"age": "<=30", "income": "medium", "student": "yes", "credit_rating": "fair"}


class TAN:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def _cmi(self, rows, fi, fj, ci):
        """Conditional mutual information I(Xi; Xj | C)."""
        n = len(rows)
        cnt_ijc = Counter((r[fi], r[fj], r[ci]) for r in rows)
        cnt_ic = Counter((r[fi], r[ci]) for r in rows)
        cnt_jc = Counter((r[fj], r[ci]) for r in rows)
        cnt_c = Counter(r[ci] for r in rows)
        mi = 0.0
        for (a, b, c), nabc in cnt_ijc.items():
            p_abc = nabc / n
            p_ab_c = nabc / cnt_c[c]
            p_a_c = cnt_ic[(a, c)] / cnt_c[c]
            p_b_c = cnt_jc[(b, c)] / cnt_c[c]
            mi += p_abc * math.log(p_ab_c / (p_a_c * p_b_c))
        return mi

    def fit(self, rows, feats):
        self.feats = feats
        ci = len(feats)  # class column index
        self.classes = Counter(r[ci] for r in rows)
        self.n = len(rows)
        self.values = {f: sorted({r[i] for r in rows}) for i, f in enumerate(feats)}

        # 1-2. Maximum spanning tree over features, weighted by CMI (Prim's algorithm)
        w = {(i, j): self._cmi(rows, i, j, ci) for i, j in combinations(range(len(feats)), 2)}
        in_tree, edges = {0}, []
        while len(in_tree) < len(feats):
            i, j = max(((a, b) for (a, b) in w if (a in in_tree) != (b in in_tree)),
                       key=lambda e: w[e])
            edges.append((i, j))
            in_tree.update([i, j])

        # 3. Orient edges away from root feature 0
        self.parent = {0: None}
        for i, j in edges:
            p, ch = (i, j) if i in self.parent else (j, i)
            self.parent[ch] = p
        self.tree_edges = [(feats[p], feats[c]) for c, p in self.parent.items() if p is not None]

        # 4. Conditional probability tables: P(Xi | C, Parent(Xi))
        self.cpt = Counter()
        self.cpt_den = Counter()
        for r in rows:
            c = r[ci]
            for i, f in enumerate(feats):
                pv = r[self.parent[i]] if self.parent[i] is not None else None
                self.cpt[(f, r[i], c, pv)] += 1
                self.cpt_den[(f, c, pv)] += 1
        return self

    def score(self, x):
        """TAN score(C) = P(C) * prod_i P(x_i | C, Parent(x_i))."""
        scores = {}
        for c, nc in self.classes.items():
            s = nc / self.n
            for i, f in enumerate(self.feats):
                pidx = self.parent[i]
                pv = x[self.feats[pidx]] if pidx is not None else None
                k = len(self.values[f])
                num = self.cpt[(f, x[f], c, pv)] + self.alpha
                den = self.cpt_den[(f, c, pv)] + self.alpha * k
                s *= num / den if den else 0.0
            scores[c] = s
        return scores

    def posterior(self, x):
        s = self.score(x)
        total = sum(s.values())
        return {c: v / total for c, v in s.items()}


if __name__ == "__main__":
    tan = TAN(alpha=1).fit(DATA, FEATURES)  # alpha=1 avoids zero counts
    print("TAN tree edges:       ", tan.tree_edges)
    print("TAN scores:           ", {k: round(v, 6) for k, v in tan.score(X_NEW).items()})
    print("TAN posterior:        ", {k: round(v, 3) for k, v in tan.posterior(X_NEW).items()})
