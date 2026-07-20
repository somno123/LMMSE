import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

TRUE = {9: 9, 11: 11, 13: 13}
ANGLES = [0, 90, 180, 270]

# data[count][model] = [pred@0, @90, @180, @270]  (measured runs only)
data = {
    9:  {'Sonar': [8, 18, 10, 13], 'Sonnet': [11, 14, 10, 20], 'GPT': [6, 10, 8, 8]},
    11: {'Sonar': [9, 8, 10, 10],  'Sonnet': [14, 20, 13, 10], 'GPT': [10, 7, 8, 11]},
    13: {'Sonar': [10, 12, 13, 10], 'Sonnet': [14, 10, 20, 20], 'GPT': [10, 11, 13, 9]},
}
MODELS = ['Sonar', 'Sonnet', 'GPT']
GEMINI_UPRIGHT = {9: 7, 11: 11, 13: 11}   # measured 0-degree runs only

# |error| matrix: 9 cells (model x count) x 4 rotations
E = np.array([[abs(data[c][m][k] - TRUE[c]) for k in range(4)]
              for c in (9, 11, 13) for m in MODELS], float)

print("== Per-model summaries (12 configurations each) ==")
for m in MODELS:
    err = np.array([data[c][m][k] - TRUE[c] for c in (9, 11, 13) for k in range(4)], float)
    sd_rot = np.mean([np.std(data[c][m], ddof=1) for c in (9, 11, 13)])
    exact = np.mean([data[c][m][k] == TRUE[c] for c in (9, 11, 13) for k in range(4)])
    print(f"  {m:7s} MAE={np.abs(err).mean():.2f}  bias={err.mean():+.2f}  "
          f"SD_rot={sd_rot:.2f}  exact={exact*100:.0f}%")
gem_err = [GEMINI_UPRIGHT[c] - c for c in (9, 11, 13)]
print(f"  Gemini (upright only): predictions "
      f"{[GEMINI_UPRIGHT[c] for c in (9, 11, 13)]}, MAE={np.abs(gem_err).mean():.2f}, "
      f"exact={np.mean([e == 0 for e in gem_err])*100:.0f}%")

# 1) Friedman across rotations, blocked by cell (n=9)
chi2, p = stats.friedmanchisquare(*(E[:, k] for k in range(4)))
print(f"\nFriedman rotation main effect (9 blocks x 4 rotations): "
      f"chi2(3)={chi2:.2f}, p={p:.3f}")

# 2) Wilcoxon: upright vs mean-rotated, paired by cell (n=9)
W, pw = stats.wilcoxon(E[:, 0], E[:, 1:].mean(axis=1))
print(f"Wilcoxon upright vs rotated (n=9): W={W:.1f}, p={pw:.3f} "
      f"(means {E[:,0].mean():.2f} vs {E[:,1:].mean():.2f})")

# 3) model x rotation interaction (exploratory)
rows = [(m, ang, c, abs(data[c][m][k] - TRUE[c]))
        for c in (9, 11, 13) for m in MODELS for k, ang in enumerate(ANGLES)]
df = pd.DataFrame(rows, columns=['model', 'rotation', 'count', 'abserr'])
fit = smf.ols('abserr ~ C(model) * C(rotation)', data=df).fit()
print("\nTwo-way ANOVA on |error| (exploratory):")
print(sm.stats.anova_lm(fit, typ=2))
