import pandas as pd

df = pd.read_excel("exports/training.xlsx")
print(df.head())

df['total_loss'] = df['Best train loss']+df['Best val loss']
df_sorted = df.sort_values('total_loss')

print(df.head())