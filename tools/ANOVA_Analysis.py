import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
import numpy as np
import os 

# === 1️⃣ 读取 Excel 数据 ===
file_path = r"Sensitive_test_IIII\Summary_table_Single_Parameter.xlsx"  # ← 修改为你的文件路径
sheet_name = 0  # 或者填写具体sheet名，如 'Sheet1'

df = pd.read_excel(file_path, sheet_name=sheet_name)

# === 2️⃣ 重命名列，保证统一 ===
df = df.rename(columns={
    'lam': 'Lam',
    'gamma': 'Gamma',
    'U': 'U',
    'CT MAPE (%)': 'Ct_MAPE'
})
print(f"初始数据共有 {len(df)} 条记录")

# === 3️⃣ 类型转换 & 筛选 lam/gamma ===
df['Lam'] = pd.to_numeric(df['Lam'], errors='coerce')
df['Gamma'] = pd.to_numeric(df['Gamma'], errors='coerce')
df['U'] = pd.to_numeric(df['U'], errors='coerce')

df = df.query("(Lam == 0.3 or Lam == 0.7) and (Gamma == 0.3 or Gamma == 0.7)").copy()
print(f"筛选后共有 {len(df)} 条数据")

# === 4️⃣ 转换为分类变量 ===
df['Lam'] = df['Lam'].astype('category')
df['Gamma'] = df['Gamma'].astype('category')
df['U'] = df['U'].astype('category')

# === 5️⃣ 执行 ANOVA ===
model = ols('Ct_MAPE ~ C(Lam) * C(Gamma) * C(U)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\n=== 三因素 ANOVA 结果 ===")
print(anova_table)

# === 6️⃣ 构建导出表格 ===
anova_export = anova_table.reset_index().rename(columns={'index': 'Source'})
anova_export['p-value'] = anova_export['PR(>F)'].apply(
    lambda p: '<0.001' if (not pd.isna(p) and p < 0.001) else f'{p:.4f}' if not pd.isna(p) else ''
)
anova_export['Significant'] = np.where(anova_table['PR(>F)'] < 0.05, 'Yes', 'No')
anova_export['mean_sq'] = anova_export['sum_sq'] / anova_export['df']

# 仅保留关键列
anova_export = anova_export[['Source', 'sum_sq', 'df', 'mean_sq', 'F', 'p-value', 'Significant']]

# === 7️⃣ 输出到 Excel ===
output_dir = os.path.join(os.path.dirname(file_path), "ANOVA_Results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "ANOVA_Output.xlsx")

anova_export.to_excel(output_path, index=False)
print(f"\n✅ ANOVA 结果已保存到: {output_path}")

# === 8️⃣ 同时打印显著性摘要 ===
print("\n=== 显著性说明 ===")
for factor, p in anova_table['PR(>F)'].items():
    if pd.isna(p):
        continue
    if p < 0.05:
        print(f"✅ {factor} 显著影响 Ct MAPE (p = {p:.4f})")
    else:
        print(f"❌ {factor} 无显著影响 (p = {p:.4f})")