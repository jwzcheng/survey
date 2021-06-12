# %%
import pandas as pd

df = pd.read_csv('../data/california_housing_test.csv')

df.to_csv('tmp1.csv')
# %%
df.index.to_list()
# %%
df['index'] = df.index
# %%
