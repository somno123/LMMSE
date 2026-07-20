import pandas as pd, numpy as np
from itertools import combinations
from scipy import stats

df = pd.read_csv('picture_ratings.csv')
MODELS = ['Sonar', 'Claude Sonnet 4.5', 'Grok 4.1', 'Gemini 3 Flash', 'GPT-5.2']
CRIT = ['Ph', 'G', 'S', 'W', 'Pr']

# 5 per-criterion means per model (over 2 pics x 3 judges x 5 rounds)
cm = df.pivot_table(index='criterion', columns='model', values='score').loc[CRIT, MODELS]
print("per-criterion means:\n", cm.round(3), "\n")
