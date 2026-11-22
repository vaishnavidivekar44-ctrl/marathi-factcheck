"""Statistical tests for the Marathi fact-checker pilot (run: python stats_tests.py results_detail_v6.csv)."""
import sys
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm

path = sys.argv[1] if len(sys.argv) > 1 else "results_detail_v6.csv"
df = pd.read_csv(path, encoding="utf-8-sig")
df["rewrite"] = (df["system"] == "with_rewriting").astype(int)

# Paired layout: one row per (claim, form), columns baseline / with_rewriting
corr = df.pivot_table(index=["claim_id", "form"], columns="system", values="correct")
sim = df.pivot_table(index=["claim_id", "form"], columns="system", values="best_similarity")

print("Accuracy:", df.groupby("system")["correct"].mean().round(2).to_dict())

# 1. McNemar exact test (paired correct/incorrect)
b = int(((corr["baseline"] == 1) & (corr["with_rewriting"] == 0)).sum())   # baseline only
c = int(((corr["baseline"] == 0) & (corr["with_rewriting"] == 1)).sum())   # rewriting only
p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
print(f"McNemar exact: baseline-only={b}, rewriting-only={c}, p={p:.3f}")

# 2. Logistic regression: correct ~ rewrite + form
X = pd.get_dummies(df[["rewrite", "form"]], columns=["form"], drop_first=True).astype(float)
res = sm.Logit(df["correct"].astype(float), sm.add_constant(X)).fit(disp=0)
ci = np.exp(res.conf_int().loc["rewrite"])
print(f"Logistic regression: OR(rewrite)={np.exp(res.params['rewrite']):.2f} "
      f"(95% CI {ci[0]:.2f}-{ci[1]:.2f}), p={res.pvalues['rewrite']:.3f}")

# 3. Shapiro-Wilk on similarity (normality)
for s in ["baseline", "with_rewriting"]:
    w, p = stats.shapiro(sim[s]); print(f"Shapiro-Wilk {s}: W={w:.3f}, p={p:.4f}")

# 4. Kruskal-Wallis: similarity across the three input forms
for s in ["baseline", "with_rewriting"]:
    groups = [g["best_similarity"].values for _, g in df[df.system == s].groupby("form")]
    h, p = stats.kruskal(*groups); print(f"Kruskal-Wallis {s}: H={h:.2f}, p={p:.4f}")

# 5. Wilcoxon signed-rank: paired similarity
w, p = stats.wilcoxon(sim["with_rewriting"], sim["baseline"])
print(f"Wilcoxon signed-rank (similarity): W={w:.1f}, p={p:.3f}")

# 6. Claim-level sign test and bootstrap
claim = corr.groupby("claim_id").mean()
diff = claim["with_rewriting"] - claim["baseline"]
up, down = int((diff > 0).sum()), int((diff < 0).sum())
print(f"Claim-level sign test: improved={up}, worse={down}, p={stats.binomtest(min(up, down), up + down, 0.5).pvalue:.4f}")
rng = np.random.default_rng(42)
boot = [rng.choice(diff.values, len(diff)).mean() for _ in range(10000)]
print(f"Bootstrap mean gain={diff.mean():.2f}, 95% CI {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f}")
