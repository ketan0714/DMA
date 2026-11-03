from collections import Counter, defaultdict

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


class NaiveBayes:
    def __init__(self, alpha=0.0):
        self.alpha = alpha  # Laplace smoothing; 0 reproduces the slide numbers

    def fit(self, rows, feats):
        self.feats = feats
        self.classes = Counter(r[-1] for r in rows)
        self.n = len(rows)
        self.values = {f: sorted({r[i] for r in rows}) for i, f in enumerate(feats)}
        self.counts = defaultdict(int)  # (feature, value, class) -> count
        for r in rows:
            for i, f in enumerate(feats):
                self.counts[(f, r[i], r[-1])] += 1
        return self

    def score(self, x):
        """Unnormalized score(C) = P(C) * prod P(x_i | C)."""
        scores = {}
        for c, nc in self.classes.items():
            s = nc / self.n
            for f in self.feats:
                k = len(self.values[f])
                s *= (self.counts[(f, x[f], c)] + self.alpha) / (nc + self.alpha * k)
            scores[c] = s
        return scores

    def posterior(self, x):
        """Divide by P(X) = sum of scores to get P(C | X)."""
        s = self.score(x)
        total = sum(s.values())
        return {c: v / total for c, v in s.items()}


if __name__ == "__main__":
    nb = NaiveBayes().fit(DATA, FEATURES)
    print("Naive Bayes scores:   ", {k: round(v, 6) for k, v in nb.score(X_NEW).items()})
    print("Naive Bayes posterior:", {k: round(v, 3) for k, v in nb.posterior(X_NEW).items()})
