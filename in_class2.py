# %%
import pandas as pd
import numpy as np
import sklearn as sk


# %%
salary_data = pd.read_csv("2025_salaries.csv", header=1)
salary_data.head()
# %%
stats = pd.read_csv("nba_2025.txt", sep =",")
stats.head()
# %%
df = pd.merge(stats, salary_data, on = "Player", how = "inner")
df.head()
# %%
# Duplicates
# Brackets creates new dataframe
duplicates = df[df.duplicated(subset="Player", keep = False)]
print(duplicates)
# %%
# after model:
# predict - model.predict(X)
# evaluate: model.score(X)