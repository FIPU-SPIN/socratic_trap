import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Učitaj
df = pd.read_csv('merged_annotations.csv', delimiter=';')
mapping = pd.read_csv('response_mapping.csv', delimiter=';')
df = df.merge(mapping, left_on='ID', right_on='response_id', how='left')

# Final label
def majority(row):
    vals = [int(row['annotator_1']), int(row['annotator_2']), int(row['annotator_3'])]
    valid = [v for v in vals if 0 <= v <= 2]
    return max(set(valid), key=valid.count) if len(valid) == 3 else (valid[0] if valid else -1)

df['final_label'] = df.apply(majority, axis=1)
df['is_strategic'] = (df['final_label'] == 2).astype(int)

# Izvuci domenu iz ID-a
df['domain'] = df['ID'].str[:3].map({
    'ALG': 'Algorithms',
    'PLP': 'Prog. Languages',
    'DB_': 'Databases',
    'NET': 'Networks',
    'OS_': 'OS'
})

# Za persuasiveness, uzmi prosjek tri anotatora
pers_cols = ['pers_1', 'pers_2', 'pers_3']
df['avg_persuasiveness'] = df[pers_cols].mean(axis=1)

strategic = df[df['final_label'] == 2]
# Za error_type, uzmi od prvog anotatora (ili većinu)
strategic['error_type'] = strategic['error_1']

# ========== 1. POSTOCI PO MODELU ==========
print("=" * 60)
print("1. STOPA STRATEŠKIH ZABLUDA PO MODELU")
print("=" * 60)
for m in sorted(df['model'].unique()):
    d = df[df['model'] == m]
    print(f"  {m}: {d['is_strategic'].mean()*100:.1f}%")

# ========== 2. POSTOCI PO DOMENI ==========
print("\n" + "=" * 60)
print("2. STOPA STRATEŠKIH ZABLUDA PO DOMENI")
print("=" * 60)
for d in sorted(df['domain'].unique()):
    data = df[df['domain'] == d]
    print(f"  {d}: {data['is_strategic'].mean()*100:.1f}%")

# ========== 3. ANOVA PO MODELIMA ==========
print("\n" + "=" * 60)
print("3. ONE-WAY ANOVA (modeli)")
print("=" * 60)
models = sorted(df['model'].unique())
groups = [df[df['model'] == m]['is_strategic'].values for m in models]
f_stat, p_val = stats.f_oneway(*groups)
print(f"  F = {f_stat:.4f}, p = {p_val:.4f}")

# ========== 4. TWO-WAY ANOVA (model × domena) ==========
print("\n" + "=" * 60)
print("4. TWO-WAY ANOVA (model × domena)")
print("=" * 60)
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('is_strategic ~ C(model) + C(domain) + C(model):C(domain)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# ========== 5. ANOVA ZA UVJERLJIVOST ==========
print("\n" + "=" * 60)
print("5. ANOVA ZA UVJERLJIVOST (samo klasa 2)")
print("=" * 60)
pers_groups = [strategic[strategic['model'] == m]['avg_persuasiveness'].dropna().values for m in models]
pers_groups = [g for g in pers_groups if len(g) > 0]
if len(pers_groups) >= 2:
    f_pers, p_pers = stats.f_oneway(*pers_groups)
    print(f"  F = {f_pers:.4f}, p = {p_pers:.4f}")
    for m in models:
        data = strategic[strategic['model'] == m]['avg_persuasiveness'].dropna()
        if len(data) > 0:
            print(f"  {m}: M = {data.mean():.2f} (SD = {data.std():.2f})")

# ========== 6. CHI-SQUARE (model × tip greške) ==========
print("\n" + "=" * 60)
print("6. CHI-SQUARE: Model × Tip greške")
print("=" * 60)
ct_model = pd.crosstab(strategic['model'], strategic['error_type'])
if ct_model.shape[1] >= 2:
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct_model)
    print(f"  χ²({dof}) = {chi2:.4f}, p = {p_chi:.4f}")
    print("\n  Kontingencijska tablica:")
    print(ct_model)

# ========== 7. CHI-SQUARE (domena × tip greške) ==========
print("\n" + "=" * 60)
print("7. CHI-SQUARE: Domena × Tip greške")
print("=" * 60)
ct_domain = pd.crosstab(strategic['domain'], strategic['error_type'])
if ct_domain.shape[1] >= 2:
    chi2_d, p_chi_d, dof_d, _ = stats.chi2_contingency(ct_domain)
    print(f"  χ²({dof_d}) = {chi2_d:.4f}, p = {p_chi_d:.4f}")
    print("\n  Kontingencijska tablica:")
    print(ct_domain)

# ========== 8. COHEN'S D ==========
print("\n" + "=" * 60)
print("8. COHEN'S D (veličina efekta između modela)")
print("=" * 60)
for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        g1 = df[df['model'] == m1]['is_strategic']
        g2 = df[df['model'] == m2]['is_strategic']
        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        if pooled_std > 0:
            d = (g1.mean() - g2.mean()) / pooled_std
            print(f"  {m1} vs {m2}: d = {d:.3f}")

# ========== 9. TUKEY HSD ==========
print("\n" + "=" * 60)
print("9. TUKEY HSD")
print("=" * 60)
tukey = pairwise_tukeyhsd(df['is_strategic'], df['model'], alpha=0.05)
print(tukey)