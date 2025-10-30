import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
import numpy as np
import os 

file_path = r"Sensitive_test_IIII\Summary_table_Single_Parameter.xlsx"  
sheet_name = 0  

df = pd.read_excel(file_path, sheet_name=sheet_name)


df = df.rename(columns={
    'lam': 'Lam',
    'gamma': 'Gamma',
    'U': 'U',
    'CT MAPE (%)': 'Ct_MAPE'
})


df['Lam'] = pd.to_numeric(df['Lam'], errors='coerce')
df['Gamma'] = pd.to_numeric(df['Gamma'], errors='coerce')
df['U'] = pd.to_numeric(df['U'], errors='coerce')

df = df.query("(Lam == 0.3 or Lam == 0.7) and (Gamma == 0.3 or Gamma == 0.7)").copy()



df['Lam'] = df['Lam'].astype('category')
df['Gamma'] = df['Gamma'].astype('category')
df['U'] = df['U'].astype('category')


model = ols('Ct_MAPE ~ C(Lam) * C(Gamma) * C(U)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)


anova_export = anova_table.reset_index().rename(columns={'index': 'Source'})
anova_export['p-value'] = anova_export['PR(>F)'].apply(
    lambda p: '<0.001' if (not pd.isna(p) and p < 0.001) else f'{p:.4f}' if not pd.isna(p) else ''
)
anova_export['Significant'] = np.where(anova_table['PR(>F)'] < 0.05, 'Yes', 'No')
anova_export['mean_sq'] = anova_export['sum_sq'] / anova_export['df']


anova_export = anova_export[['Source', 'sum_sq', 'df', 'mean_sq', 'F', 'p-value', 'Significant']]


output_dir = os.path.join(os.path.dirname(file_path), "ANOVA_Results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ANOVA_Output.xlsx")

anova_export.to_excel(output_path, index=False)


