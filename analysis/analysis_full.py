"""
analysis_full.py
================
Statistička analiza anotacija za projekt SOCRATIC_TRAP.

Ulaz:  data/merged_annotations.xlsx
Izlaz: output/statistical_results.txt
       output/statistical_results.json  (za daljnju obradu)

Pokretanje:
    python analysis/analysis_full.py

Ovisnosti (vidi requirements.txt):
    pandas, numpy, scipy, openpyxl
"""

import os
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # SOCRATIC_TRAP/
DATA_PATH  = BASE_DIR / "data"  / "merged_annotations.xlsx"
OUT_DIR    = BASE_DIR / "output"
OUT_TXT    = OUT_DIR  / "statistical_results.txt"
OUT_JSON   = OUT_DIR  / "statistical_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# KNOWN MODELS
# ─────────────────────────────────────────────
KNOWN_MODELS = [
    "deepseek-r1_70b",
    "gemma4_31b",
    "llama3.3_70b",
    "llama4_16x17b",
    "ministral-3_14b",
    "mistral_7b",
    "qwen3_235b",
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_id(id_str: str):
    """Rastavlja ID u (domain, model, item)."""
    id_str = str(id_str)
    domain = id_str.split("_")[0]
    item   = id_str[-1]
    model  = next((m for m in KNOWN_MODELS if m in id_str), None)
    return domain, model, item


def majority_vote(row, cols):
    """Vraća vrijednost s najvećim brojem glasova (ignorira NaN)."""
    votes = [row[c] for c in cols if pd.notna(row[c])]
    if not votes:
        return np.nan
    vals, counts = np.unique(votes, return_counts=True)
    return vals[np.argmax(counts)]


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cohen's d s pooled SD."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / sp if sp > 0 else 0.0


def cohen_kappa_binary(a: np.ndarray, b: np.ndarray, categories=(0, 1, 2)) -> float:
    """Cohen's kappa za dva anotatora."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return np.nan
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in categories)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


def tukey_hsd(group_dict: dict, mse: float, n_total: int) -> list:
    """
    Vraća listu rječnika s rezultatima Tukey HSD za sve parove.
    group_dict: {name: np.array}
    """
    results = []
    k = len(group_dict)
    names = list(group_dict.keys())
    for m1, m2 in combinations(names, 2):
        g1, g2 = group_dict[m1], group_dict[m2]
        n1, n2 = len(g1), len(g2)
        diff = abs(g1.mean() - g2.mean())
        se   = np.sqrt(mse * (1 / n1 + 1 / n2) / 2)
        q    = diff / se if se > 0 else 0.0
        try:
            p_val = float(1 - studentized_range.cdf(q, k, n_total - k))
        except Exception:
            p_val = np.nan
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        results.append({
            "pair": f"{m1} vs {m2}",
            "mean_1": round(g1.mean(), 6),
            "mean_2": round(g2.mean(), 6),
            "diff":   round(diff,      6),
            "SE":     round(se,        6),
            "q":      round(q,         6),
            "p_adj":  round(p_val,     6),
            "sig":    sig,
        })
    return results


def welch_anova(*groups):
    """Welch's ANOVA (robusno na neujednačene varijance)."""
    k     = len(groups)
    n_j   = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    vars_ = np.array([g.var(ddof=1) if len(g) > 1 else 1e-9 for g in groups])
    w_j   = n_j / vars_
    W     = w_j.sum()
    gm_w  = (w_j * means).sum() / W
    numer = sum(w_j[i] * (means[i] - gm_w) ** 2 for i in range(k)) / (k - 1)
    lam   = sum((1 - w_j[i] / W) ** 2 / (n_j[i] - 1) for i in range(k)) * (3 / (k ** 2 - 1))
    F     = numer / (1 + (2 * (k - 2) / 3) * lam)
    df1   = k - 1
    df2   = 1 / (lam * 3 / (k ** 2 - 1)) if lam > 0 else 1e6
    p     = float(1 - stats.f.cdf(F, df1, df2))
    return float(F), int(df1), float(df2), p


def effect_size_label(val: float, metric: str = "eta2") -> str:
    if metric == "eta2":
        return "small" if val < 0.06 else "medium" if val < 0.14 else "large"
    if metric == "cramers_v":
        return "negligible" if val < 0.1 else "small" if val < 0.3 else "medium" if val < 0.5 else "large"
    if metric == "cohens_d":
        val = abs(val)
        return "small" if val < 0.5 else "medium" if val < 0.8 else "large"
    return "?"


# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────

print(f"[1/15] Učitavanje podataka: {DATA_PATH}")
df = pd.read_excel(DATA_PATH)
df[["domain", "model", "item"]] = df["ID"].apply(
    lambda x: pd.Series(parse_id(x))
)

# Izbaci retke gdje sva tri anotatora nedostaju
df = df.dropna(subset=["annotator_1", "annotator_2", "annotator_3"], how="all")

# Majority vote za klasu
df["majority_class"] = df.apply(
    lambda r: majority_vote(r, ["annotator_1", "annotator_2", "annotator_3"]), axis=1
)

# Je li strateška zabluda (klasa 2)?
df["is_strategic"] = (df["majority_class"] == 2.0).astype(float)

# Consensus tip greške (samo klasa 2)
df["error_type"] = df.apply(
    lambda r: majority_vote(r, ["error_1", "error_2", "error_3"]), axis=1
)

# Prosječna uvjerljivost
df["pers_mean"] = df[["pers_1", "pers_2", "pers_3"]].mean(axis=1, skipna=True)

# Podskupovi
df_all   = df.dropna(subset=["annotator_1", "annotator_2", "annotator_3"])  # kompletni trojci
df_class = df[df["item"] == "3"].copy()                                       # samo klasifikacijski redovi
df_cls2  = df_class[df_class["majority_class"] == 2.0].copy()                # samo klasa 2

print(f"    Ukupno redova: {len(df)}")
print(f"    Kompletnih trojci: {len(df_all)}")
print(f"    Klasifikacijskih redova (item=3): {len(df_class)}")
print(f"    Klasa 2 redova: {len(df_cls2)}")

# Grupirani podaci
model_groups  = {n: g["is_strategic"].dropna().values for n, g in df_class.groupby("model")}
domain_groups = {n: g["is_strategic"].dropna().values for n, g in df_class.groupby("domain")}

# JSON akumulator
J = {}

lines = []
SEP = "=" * 70

def h(title):
    lines.append("")
    lines.append(SEP)
    lines.append(title)
    lines.append(SEP)

# ─────────────────────────────────────────────
# 1. FLEISS' KAPPA
# ─────────────────────────────────────────────
print("[2/15] Fleiss' Kappa...")
h("1. FLEISS' KAPPA")

N_fk     = len(df_all)
k_ann    = 3
categories = [0.0, 1.0, 2.0]
n_cat    = len(categories)

matrix = np.zeros((N_fk, n_cat))
for i, (_, row) in enumerate(df_all.iterrows()):
    for j, cat in enumerate(categories):
        for ann in ["annotator_1", "annotator_2", "annotator_3"]:
            if row[ann] == cat:
                matrix[i, j] += 1

P_i    = np.sum(matrix * (matrix - 1), axis=1) / (k_ann * (k_ann - 1))
P_bar  = float(np.mean(P_i))
p_j    = np.sum(matrix, axis=0) / (N_fk * k_ann)
P_e    = float(np.sum(p_j ** 2))
kappa  = (P_bar - P_e) / (1 - P_e)
SE_k   = np.sqrt((P_bar * (1 - P_bar)) / (N_fk * (1 - P_e) ** 2))
z_k    = kappa / SE_k if SE_k > 0 else 0
p_k    = float(2 * (1 - stats.norm.cdf(abs(z_k))))

interp_map = [
    (0.0,  "Less than chance"),
    (0.20, "Slight"),
    (0.40, "Fair"),
    (0.60, "Moderate"),
    (0.80, "Substantial"),
    (1.01, "Almost perfect"),
]
interp = next(label for threshold, label in interp_map if kappa < threshold)

lines += [
    f"  N (kompletni trojci): {N_fk}",
    f"  Fleiss' κ = {kappa:.4f}",
    f"  SE = {SE_k:.4f}, z = {z_k:.4f}, p = {p_k:.6f}",
    f"  P_bar (observed) = {P_bar:.4f}, P_e (expected) = {P_e:.4f}",
    f"  Interpretacija: {interp}",
    "",
    f"  Proporcije po kategorijama:",
]
for j, cat in enumerate(categories):
    lines.append(f"    Klasa {int(cat)}: p_j = {p_j[j]:.4f}")

J["fleiss_kappa"] = {
    "N": N_fk, "kappa": round(kappa, 6), "SE": round(float(SE_k), 6),
    "z": round(float(z_k), 4), "p": round(p_k, 6),
    "P_bar": round(P_bar, 4), "P_e": round(P_e, 4),
    "interpretation": interp,
    "p_j": {f"class_{int(c)}": round(float(p_j[i]), 4) for i, c in enumerate(categories)},
}

# ─────────────────────────────────────────────
# 2. PAIRWISE COHEN'S KAPPA
# ─────────────────────────────────────────────
print("[3/15] Pairwise Cohen's Kappa...")
h("2. PAIRWISE COHEN'S KAPPA (inter-annotator)")

ann_arr = df_all[["annotator_1", "annotator_2", "annotator_3"]].values.astype(float)
pairs_ann = [("Ann1 vs Ann2", 0, 1), ("Ann1 vs Ann3", 0, 2), ("Ann2 vs Ann3", 1, 2)]

lines.append(f"\n  {'Par':<20} {'κ':>8} {'% slaganje':>12}")
lines.append("  " + "-" * 42)

kappas_pair = {}
for name, i, j in pairs_ann:
    k_val = cohen_kappa_binary(ann_arr[:, i], ann_arr[:, j])
    agr   = float(np.mean(ann_arr[:, i] == ann_arr[:, j]))
    kappas_pair[name] = {"kappa": round(k_val, 4), "agreement": round(agr, 4)}
    lines.append(f"  {name:<20} {k_val:>8.4f} {agr:>12.4f}")

mean_k = float(np.mean([v["kappa"] for v in kappas_pair.values()]))
lines.append(f"\n  Prosječni pairwise κ = {mean_k:.4f}")
J["pairwise_kappa"] = {"pairs": kappas_pair, "mean_kappa": round(mean_k, 4)}

# ─────────────────────────────────────────────
# 3. STOPA STRATEŠKIH ZABLUDA PO MODELU
# ─────────────────────────────────────────────
print("[4/15] Stopa strateških zabluda po modelu...")
h("3. STOPA STRATEŠKIH ZABLUDA PO MODELU")

model_rates = df_class.groupby("model")["is_strategic"].agg(
    rate="mean", n_strategic="sum", n_total="count"
).sort_values("rate", ascending=False)

lines.append(f"\n  {'Model':<25} {'Rate':>8} {'N_strat':>9} {'N_ukupno':>10}")
lines.append("  " + "-" * 55)
for model, row in model_rates.iterrows():
    lines.append(f"  {model:<25} {row['rate']:>8.4f} {int(row['n_strategic']):>9} {int(row['n_total']):>10}")

overall_rate = float(df_class["is_strategic"].mean())
lines.append(f"\n  Ukupna stopa: {overall_rate:.4f} ({overall_rate*100:.1f}%)")
J["strategic_fallacy_rates"] = {
    "by_model": model_rates.round(4).to_dict("index"),
    "overall":  round(overall_rate, 4),
}

# ─────────────────────────────────────────────
# 4. ONE-WAY ANOVA — MODEL
# ─────────────────────────────────────────────
print("[5/15] One-way ANOVA (model)...")
h("4. ONE-WAY ANOVA: Strateška zabluda ~ Model")

groups_m = [g for g in model_groups.values() if len(g) > 1]
f_m, p_m = stats.f_oneway(*groups_m)
gm_m     = df_class["is_strategic"].mean()
ss_b_m   = sum(len(g) * (g.mean() - gm_m) ** 2 for g in groups_m)
ss_t_m   = sum(((g - gm_m) ** 2).sum() for g in groups_m)
eta2_m   = ss_b_m / ss_t_m if ss_t_m > 0 else 0
k_m      = len(groups_m)
N_m      = sum(len(g) for g in groups_m)
mse_m    = sum(((g - g.mean()) ** 2).sum() for g in groups_m) / (N_m - k_m)

lines += [
    f"  F({k_m-1}, {N_m-k_m}) = {f_m:.4f}, p = {p_m:.6f}",
    f"  η² = {eta2_m:.4f}  [{effect_size_label(eta2_m, 'eta2')} efekt]",
    f"  MSE = {mse_m:.6f}",
]
J["anova_model"] = {
    "F": round(f_m, 4), "df_between": k_m-1, "df_within": N_m-k_m,
    "p": round(float(p_m), 6), "eta2": round(eta2_m, 4),
    "MSE": round(mse_m, 6),
}

# ─────────────────────────────────────────────
# 5. TUKEY HSD — MODEL
# ─────────────────────────────────────────────
print("[6/15] Tukey HSD (model)...")
h("5. TUKEY HSD POST-HOC: Model parovi")

tukey_m = tukey_hsd(model_groups, mse_m, N_m)
lines.append(f"\n  {'Par':<45} {'Diff':>8} {'SE':>8} {'q':>8} {'p_adj':>10} {'Sig':>5}")
lines.append("  " + "-" * 90)
for r in tukey_m:
    lines.append(f"  {r['pair']:<45} {r['diff']:>8.4f} {r['SE']:>8.4f} {r['q']:>8.4f} {r['p_adj']:>10.6f} {r['sig']:>5}")
J["tukey_model"] = tukey_m

# ─────────────────────────────────────────────
# 6. ONE-WAY ANOVA — DOMENA
# ─────────────────────────────────────────────
print("[7/15] ANOVA + post-hoc (domena)...")
h("6. ONE-WAY ANOVA + TUKEY HSD: Strateška zabluda ~ Domena")

groups_d = [g for g in domain_groups.values() if len(g) > 1]
f_d, p_d = stats.f_oneway(*groups_d)
gm_d     = df_class["is_strategic"].mean()
ss_b_d   = sum(len(g) * (g.mean() - gm_d) ** 2 for g in groups_d)
ss_t_d   = sum(((g - gm_d) ** 2).sum() for g in groups_d)
eta2_d   = ss_b_d / ss_t_d if ss_t_d > 0 else 0
k_d      = len(groups_d)
N_d      = sum(len(g) for g in groups_d)
mse_d    = sum(((g - g.mean()) ** 2).sum() for g in groups_d) / (N_d - k_d)

lines += [
    f"  F({k_d-1}, {N_d-k_d}) = {f_d:.4f}, p = {p_d:.6f}",
    f"  η² = {eta2_d:.4f}  [{effect_size_label(eta2_d, 'eta2')} efekt]",
    "",
    f"  {'Domena':<12} {'Mean':>8} {'SD':>8} {'N':>6}",
    "  " + "-" * 38,
]
for name, g in domain_groups.items():
    lines.append(f"  {name:<12} {g.mean():>8.4f} {g.std():>8.4f} {len(g):>6}")

lines.append(f"\n  Tukey HSD (domene):")
lines.append(f"  {'Par':<25} {'Diff':>8} {'q':>8} {'p_adj':>10} {'Sig':>5}")
lines.append("  " + "-" * 60)
tukey_d = tukey_hsd(domain_groups, mse_d, N_d)
for r in tukey_d:
    lines.append(f"  {r['pair']:<25} {r['diff']:>8.4f} {r['q']:>8.4f} {r['p_adj']:>10.6f} {r['sig']:>5}")

J["anova_domain"] = {
    "F": round(f_d, 4), "df_between": k_d-1, "df_within": N_d-k_d,
    "p": round(float(p_d), 6), "eta2": round(eta2_d, 4),
    "tukey": tukey_d,
}

# ─────────────────────────────────────────────
# 7. TWO-WAY ANOVA (model × domena)
# ─────────────────────────────────────────────
print("[8/15] Two-way ANOVA...")
h("7. TWO-WAY ANOVA: Strateška zabluda ~ Model × Domena")

df_2w       = df_class[["model", "domain", "is_strategic"]].dropna()
gm_2w       = df_2w["is_strategic"].mean()
ss_tot_2w   = ((df_2w["is_strategic"] - gm_2w) ** 2).sum()

m_means = df_2w.groupby("model")["is_strategic"].mean()
m_cnts  = df_2w.groupby("model")["is_strategic"].count()
d_means = df_2w.groupby("domain")["is_strategic"].mean()
d_cnts  = df_2w.groupby("domain")["is_strategic"].count()
c_means = df_2w.groupby(["model", "domain"])["is_strategic"].mean()
c_cnts  = df_2w.groupby(["model", "domain"])["is_strategic"].count()

ss_mod  = sum(m_cnts[m] * (m_means[m] - gm_2w) ** 2 for m in m_means.index)
ss_dom  = sum(d_cnts[d] * (d_means[d] - gm_2w) ** 2 for d in d_means.index)
ss_int  = sum(
    cnt * (c_means[m, d] - m_means[m] - d_means[d] + gm_2w) ** 2
    for (m, d), cnt in c_cnts.items()
    if m in m_means.index and d in d_means.index
)
ss_err_2w = ss_tot_2w - ss_mod - ss_dom - ss_int

n_mods = df_2w["model"].nunique()
n_doms = df_2w["domain"].nunique()
N_2w   = len(df_2w)
df_mod_2 = n_mods - 1
df_dom_2 = n_doms - 1
df_int_2 = df_mod_2 * df_dom_2
df_err_2 = N_2w - n_mods * n_doms

ms_mod_2 = ss_mod / df_mod_2
ms_dom_2 = ss_dom / df_dom_2
ms_int_2 = ss_int / df_int_2 if df_int_2 > 0 else 0
ms_err_2 = ss_err_2w / df_err_2 if df_err_2 > 0 else 1

F_mod_2  = ms_mod_2 / ms_err_2
F_dom_2  = ms_dom_2 / ms_err_2
F_int_2  = ms_int_2 / ms_err_2

p_mod_2  = float(1 - stats.f.cdf(F_mod_2, df_mod_2, df_err_2))
p_dom_2  = float(1 - stats.f.cdf(F_dom_2, df_dom_2, df_err_2))
p_int_2  = float(1 - stats.f.cdf(F_int_2, df_int_2, df_err_2))

eta2_mod_2 = ss_mod / ss_tot_2w
eta2_dom_2 = ss_dom / ss_tot_2w
eta2_int_2 = ss_int / ss_tot_2w

lines += [
    f"  N={N_2w}, modeli={n_mods}, domene={n_doms}",
    "",
    f"  {'Izvor':<20} {'SS':>10} {'df':>5} {'MS':>10} {'F':>10} {'p':>12}",
    "  " + "-" * 72,
    f"  {'Model':<20} {ss_mod:>10.4f} {df_mod_2:>5} {ms_mod_2:>10.4f} {F_mod_2:>10.4f} {p_mod_2:>12.6f}",
    f"  {'Domena':<20} {ss_dom:>10.4f} {df_dom_2:>5} {ms_dom_2:>10.4f} {F_dom_2:>10.4f} {p_dom_2:>12.6f}",
    f"  {'Model × Domena':<20} {ss_int:>10.4f} {df_int_2:>5} {ms_int_2:>10.4f} {F_int_2:>10.4f} {p_int_2:>12.6f}",
    f"  {'Greška':<20} {ss_err_2w:>10.4f} {df_err_2:>5} {ms_err_2:>10.4f}",
    f"  {'Ukupno':<20} {ss_tot_2w:>10.4f} {N_2w-1:>5}",
    "",
    f"  η²: Model={eta2_mod_2:.4f}, Domena={eta2_dom_2:.4f}, Interakcija={eta2_int_2:.4f}",
]
J["two_way_anova"] = {
    "model":       {"F": round(F_mod_2, 4), "df1": df_mod_2, "df2": df_err_2, "p": round(p_mod_2, 6), "eta2": round(eta2_mod_2, 4)},
    "domain":      {"F": round(F_dom_2, 4), "df1": df_dom_2, "df2": df_err_2, "p": round(p_dom_2, 6), "eta2": round(eta2_dom_2, 4)},
    "interaction": {"F": round(F_int_2, 4), "df1": df_int_2, "df2": df_err_2, "p": round(p_int_2, 6), "eta2": round(eta2_int_2, 4)},
}

# ─────────────────────────────────────────────
# 8. ANOVA ZA UVJERLJIVOST (klasa 2)
# ─────────────────────────────────────────────
print("[9/15] ANOVA za uvjerljivost...")
h("8. ANOVA: Uvjerljivost ~ Model (samo klasa 2)")

pers_groups = {n: g["pers_mean"].dropna().values for n, g in df_cls2.groupby("model")}
pers_valid  = [g for g in pers_groups.values() if len(g) > 1]

if len(pers_valid) > 1:
    f_pers, p_pers = stats.f_oneway(*pers_valid)
    gm_pers  = df_cls2["pers_mean"].dropna().mean()
    ss_b_p   = sum(len(g) * (g.mean() - gm_pers) ** 2 for g in pers_valid)
    ss_t_p   = sum(((g - gm_pers) ** 2).sum() for g in pers_valid)
    eta2_p   = ss_b_p / ss_t_p if ss_t_p > 0 else 0
    k_p      = len(pers_valid)
    N_p      = sum(len(g) for g in pers_valid)
    mse_p    = sum(((g - g.mean()) ** 2).sum() for g in pers_valid) / (N_p - k_p)

    lines += [
        f"  N klasa-2 = {len(df_cls2)}, valjanih ocjena = {int(df_cls2['pers_mean'].notna().sum())}",
        f"  F({k_p-1}, {N_p-k_p}) = {f_pers:.4f}, p = {p_pers:.6f}",
        f"  η² = {eta2_p:.4f}  [{effect_size_label(eta2_p, 'eta2')} efekt]",
        "",
        f"  {'Model':<25} {'Mean':>8} {'SD':>8} {'N':>6}",
        "  " + "-" * 53,
    ]
    for name, grp in df_cls2.groupby("model"):
        g = grp["pers_mean"].dropna()
        if len(g) > 0:
            lines.append(f"  {name:<25} {g.mean():>8.4f} {g.std():>8.4f} {len(g):>6}")

    lines.append(f"\n  Tukey HSD (uvjerljivost):")
    lines.append(f"  {'Par':<45} {'Diff':>8} {'q':>8} {'p_adj':>10} {'Sig':>5}")
    lines.append("  " + "-" * 90)
    tukey_pers = tukey_hsd(pers_groups, mse_p, N_p)
    for r in tukey_pers:
        lines.append(f"  {r['pair']:<45} {r['diff']:>8.4f} {r['q']:>8.4f} {r['p_adj']:>10.6f} {r['sig']:>5}")

    J["anova_persuasiveness"] = {
        "F": round(f_pers, 4), "df_between": k_p-1, "df_within": N_p-k_p,
        "p": round(float(p_pers), 6), "eta2": round(eta2_p, 4),
        "tukey": tukey_pers,
    }

# ─────────────────────────────────────────────
# 9. CHI-SQUARE: Model × Tip greške
# ─────────────────────────────────────────────
print("[10/15] Chi-square (model × tip greške)...")
h("9. CHI-SQUARE: Model × Tip greške")

df_err = df_cls2[df_cls2["error_type"].notna()].copy()
# Ukloni eventualne whitespace vrijednosti
df_err["error_type"] = df_err["error_type"].astype(str).str.strip()
df_err = df_err[df_err["error_type"] != "nan"]

ct_me = pd.crosstab(df_err["model"], df_err["error_type"])
chi2_me, p_me, dof_me, exp_me = stats.chi2_contingency(ct_me)
n_me = ct_me.values.sum()
cv_me = np.sqrt(chi2_me / (n_me * (min(ct_me.shape) - 1)))

lines += [
    "",
    "  Kontingencijska tablica (model × tip greške):",
    "  " + ct_me.to_string().replace("\n", "\n  "),
    "",
    f"  χ²({dof_me}) = {chi2_me:.4f}, p = {p_me:.6f}",
    f"  Cramér's V = {cv_me:.4f}  [{effect_size_label(cv_me, 'cramers_v')} efekt]",
]
J["chi2_model_error"] = {
    "chi2": round(chi2_me, 4), "df": dof_me,
    "p": round(float(p_me), 6), "cramers_v": round(cv_me, 4),
    "contingency": ct_me.to_dict(),
}

# ─────────────────────────────────────────────
# 10. CHI-SQUARE: Domena × Tip greške
# ─────────────────────────────────────────────
print("[11/15] Chi-square (domena × tip greške)...")
h("10. CHI-SQUARE: Domena × Tip greške")

ct_de = pd.crosstab(df_err["domain"], df_err["error_type"])
chi2_de, p_de, dof_de, exp_de = stats.chi2_contingency(ct_de)
n_de = ct_de.values.sum()
cv_de = np.sqrt(chi2_de / (n_de * (min(ct_de.shape) - 1)))

lines += [
    "",
    "  Kontingencijska tablica (domena × tip greške):",
    "  " + ct_de.to_string().replace("\n", "\n  "),
    "",
    f"  χ²({dof_de}) = {chi2_de:.4f}, p = {p_de:.6f}",
    f"  Cramér's V = {cv_de:.4f}  [{effect_size_label(cv_de, 'cramers_v')} efekt]",
]
J["chi2_domain_error"] = {
    "chi2": round(chi2_de, 4), "df": dof_de,
    "p": round(float(p_de), 6), "cramers_v": round(cv_de, 4),
    "contingency": ct_de.to_dict(),
}

# ─────────────────────────────────────────────
# 11. COHEN'S D — parovi modela
# ─────────────────────────────────────────────
print("[12/15] Cohen's d...")
h("11. COHEN'S D: Parovi modela (strateška zabluda)")

lines.append(f"\n  {'Par':<45} {'d':>8} {'|d|':>8} {'Veličina efekta':>18}")
lines.append("  " + "-" * 83)
cohens_d_results = []
for m1, m2 in combinations(list(model_groups.keys()), 2):
    g1, g2 = model_groups[m1], model_groups[m2]
    d = cohens_d(g1, g2)
    if np.isnan(d):
        continue
    size = effect_size_label(d, "cohens_d")
    cohens_d_results.append({"pair": f"{m1} vs {m2}", "d": round(d, 4), "abs_d": round(abs(d), 4), "size": size})
    lines.append(f"  {m1+' vs '+m2:<45} {d:>8.4f} {abs(d):>8.4f} {size:>18}")
J["cohens_d"] = cohens_d_results

# ─────────────────────────────────────────────
# 12. KRUSKAL-WALLIS
# ─────────────────────────────────────────────
print("[13/15] Kruskal-Wallis...")
h("12. KRUSKAL-WALLIS (neparametrijska alternativa ANOVA-i)")

h_m, p_kw_m = stats.kruskal(*groups_m)
h_d, p_kw_d = stats.kruskal(*groups_d)
lines += [
    f"  Po modelu:  H = {h_m:.4f}, p = {p_kw_m:.6f}",
    f"  Po domeni:  H = {h_d:.4f}, p = {p_kw_d:.6f}",
]
J["kruskal_wallis"] = {
    "by_model":  {"H": round(h_m, 4), "p": round(float(p_kw_m), 6)},
    "by_domain": {"H": round(h_d, 4), "p": round(float(p_kw_d), 6)},
}

# ─────────────────────────────────────────────
# 13. LEVENE'S TEST
# ─────────────────────────────────────────────
print("[14/15] Levene's test...")
h("13. LEVENE'S TEST (homogenost varijanci)")

lev_m, p_lev_m = stats.levene(*groups_m)
lev_d, p_lev_d = stats.levene(*groups_d)
lines += [
    f"  Modeli:  W = {lev_m:.4f}, p = {p_lev_m:.6f}  "
    f"→ {'Homogene varijance' if p_lev_m > 0.05 else 'NEHOMOGENE — preporuča se Welch ANOVA'}",
    f"  Domene:  W = {lev_d:.4f}, p = {p_lev_d:.6f}  "
    f"→ {'Homogene varijance' if p_lev_d > 0.05 else 'NEHOMOGENE — preporuča se Welch ANOVA'}",
]
J["levene"] = {
    "by_model":  {"W": round(lev_m, 4), "p": round(float(p_lev_m), 6), "homogeneous": bool(p_lev_m > 0.05)},
    "by_domain": {"W": round(lev_d, 4), "p": round(float(p_lev_d), 6), "homogeneous": bool(p_lev_d > 0.05)},
}

# ─────────────────────────────────────────────
# 14. WELCH ANOVA
# ─────────────────────────────────────────────
h("14. WELCH ANOVA (robusno na neujednačene varijance)")

F_wm, df1_wm, df2_wm, p_wm = welch_anova(*groups_m)
F_wd, df1_wd, df2_wd, p_wd = welch_anova(*groups_d)
lines += [
    f"  Po modelu: F({df1_wm}, {df2_wm:.1f}) = {F_wm:.4f}, p = {p_wm:.6f}",
    f"  Po domeni: F({df1_wd}, {df2_wd:.1f}) = {F_wd:.4f}, p = {p_wd:.6f}",
]
J["welch_anova"] = {
    "by_model":  {"F": round(F_wm, 4), "df1": df1_wm, "df2": round(df2_wm, 1), "p": round(p_wm, 6)},
    "by_domain": {"F": round(F_wd, 4), "df1": df1_wd, "df2": round(df2_wd, 1), "p": round(p_wd, 6)},
}

# ─────────────────────────────────────────────
# 15. DESKRIPTIVNA STATISTIKA
# ─────────────────────────────────────────────
print("[15/15] Deskriptivna statistika...")
h("15. DESKRIPTIVNA STATISTIKA")

pers_all = df_cls2["pers_mean"].dropna()
lines += [
    f"  Ukupno redova (nakon čišćenja):  {len(df)}",
    f"  Kompletnih trojci:               {len(df_all)}",
    f"  Klasifikacijski redovi (item=3): {len(df_class)}",
    f"  Klasa 2 redova:                  {len(df_cls2)}",
    "",
    f"  Raspodjela klasa (majority vote, item=3):",
]
for c, cnt in df_class["majority_class"].value_counts().sort_index().items():
    pct = cnt / len(df_class) * 100
    lines.append(f"    Klasa {int(c) if not pd.isna(c) else 'NA'}: {cnt} ({pct:.1f}%)")

lines += [
    "",
    f"  Uvjerljivost (klasa 2 only):",
    f"    Mean ± SD:      {pers_all.mean():.4f} ± {pers_all.std():.4f}",
    f"    Medijan [IQR]:  {pers_all.median():.4f} [{pers_all.quantile(0.25):.4f} – {pers_all.quantile(0.75):.4f}]",
    f"    Min/Max:        {pers_all.min():.1f} / {pers_all.max():.1f}",
    "",
    f"  Raspodjela tipova greške (klasa 2):",
]
for et, cnt in df_err["error_type"].value_counts().items():
    pct = cnt / len(df_err) * 100
    lines.append(f"    {et}: {cnt} ({pct:.1f}%)")

lines += ["", "  Stopa strateških zabluda po domeni:"]
for d, grp in df_class.groupby("domain"):
    rate = grp["is_strategic"].mean()
    lines.append(f"    {d}: {rate:.4f} ({rate*100:.1f}%)")

J["descriptives"] = {
    "n_total": len(df),
    "n_complete_triples": len(df_all),
    "n_class_rows": len(df_class),
    "n_class2_rows": len(df_cls2),
    "persuasiveness": {
        "mean": round(float(pers_all.mean()), 4),
        "sd":   round(float(pers_all.std()),  4),
        "median": round(float(pers_all.median()), 4),
        "q25": round(float(pers_all.quantile(0.25)), 4),
        "q75": round(float(pers_all.quantile(0.75)), 4),
        "min": float(pers_all.min()),
        "max": float(pers_all.max()),
    },
    "class_distribution": df_class["majority_class"].value_counts().to_dict(),
    "error_type_distribution": df_err["error_type"].value_counts().to_dict(),
    "strategic_rate_by_domain": {
        d: round(float(g["is_strategic"].mean()), 4)
        for d, g in df_class.groupby("domain")
    },
}

# ─────────────────────────────────────────────
# ZAPIS NA DISK
# ─────────────────────────────────────────────
header = [
    "STATISTIČKA ANALIZA — SOCRATIC_TRAP",
    f"Izvor: {DATA_PATH}",
    "=" * 70,
]
full_text = "\n".join(header + lines)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write(full_text)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(J, f, ensure_ascii=False, indent=2)

print(f"\n✓ TXT  → {OUT_TXT}")
print(f"✓ JSON → {OUT_JSON}")
print("\nDone.")