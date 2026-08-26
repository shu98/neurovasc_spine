import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import container
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import stats
import seaborn as sns
import tabulate 

sns.set_theme(style="whitegrid", rc={"axes.grid": False}, font_scale=1.1)

PLOT_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", 
    "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "black", "gold" 
]

def bootstrap_agg_price_ci(df, groupby_key, n_boot=1000, ci=95):
    records = []
    for keys, group in df.groupby([groupby_key, 'year']):
        estimates = []
        n = len(group)
        for _ in range(n_boot):
            sample = group.sample(n=n, replace=True)
            estimates.append(sample['Tot_Srvcs_Spend_adj'].sum() / sample['Tot_Srvcs'].sum())
        lower = np.percentile(estimates, (100 - ci) / 2)
        upper = np.percentile(estimates, 100 - (100 - ci) / 2)
        records.append({
            groupby_key: keys[0],
            'year': keys[1],
            'CI_lower_agg_price_adj': lower,
            'CI_upper_agg_price_adj': upper
        })
    return pd.DataFrame(records)

def compute_aagr(df, col, groupby_key='procedure', weight_col=None, start_year=2013, end_year=2023):
    """
    Compute unweighted and weighted AAGR for a given column.
    
    Parameters
    ----------
    df : pd.DataFrame
    col : str — column to compute AAGR for
    groupby_key : str — column to group by
    weight_col : str or None — column to use as weights for weighted AAGR.
                 If None, uses col itself as weights.
    start_year : int — start year (inclusive)
    end_year : int — end year (inclusive)
    
    Returns
    -------
    aagr : pd.DataFrame — per-group AAGR
    unweighted_aagr : float
    weighted_aagr : float
    """
    all_years = pd.DataFrame({
        groupby_key: np.repeat(df[groupby_key].unique(), end_year - start_year + 1),
        'year': np.tile(range(start_year, end_year + 1), df[groupby_key].nunique())
    })

    # Merge to fill missing years with NaN
    df_full = all_years.merge(
        df[[groupby_key, 'year', col] + ([weight_col] if weight_col and weight_col != col else [])],
        on=[groupby_key, 'year'],
        how='left'
    )

    aagr = (
        df_full.sort_values([groupby_key, 'year'])
        .groupby(groupby_key)
        .apply(lambda g: g[col].interpolate().pct_change().mean() * 100)
        .reset_index(name='AAGR_%')
    )

    weight_source = weight_col if weight_col else col
    weights = df.groupby(groupby_key)[weight_source].mean()

    aagr_weighted = aagr.set_index(groupby_key).join(weights.rename('_weight'))

    unweighted_aagr = aagr_weighted['AAGR_%'].mean()
    weighted_aagr = (
        (aagr_weighted['AAGR_%'] * aagr_weighted['_weight']).sum()
        / aagr_weighted['_weight'].sum()
    )

    return aagr, unweighted_aagr, weighted_aagr

def compute_cagr(df, col, groupby_key='procedure', years=[2013, 2023]):
    start_year = years[0]
    end_year = years[-1]

    # Use all available years, not just the ones in `years`
    filtered = (
        df[df['year'].between(start_year, end_year)]
        .groupby([groupby_key, 'year'])[col]
        .sum()
        .unstack('year')
    )

    def row_cagr(row):
        available = row.dropna()
        if len(available) < 2:
            return pd.Series({'CAGR_pct': np.nan, 'effective_start_year': np.nan, 'effective_end_year': np.nan})

        eff_start = available.index[available.index >= start_year].min() if any(available.index >= start_year) else np.nan
        eff_end = available.index[available.index <= end_year].max() if any(available.index <= end_year) else np.nan

        if pd.isna(eff_start) or pd.isna(eff_end) or eff_start == eff_end:
            return pd.Series({'CAGR_pct': np.nan, 'effective_start_year': eff_start, 'effective_end_year': eff_end})

        n_years = eff_end - eff_start
        cagr = ((row[eff_end] / row[eff_start]) ** (1 / n_years) - 1) * 100
        return pd.Series({'CAGR_pct': cagr, 'effective_start_year': int(eff_start), 'effective_end_year': int(eff_end)})

    cagr_df = filtered.apply(row_cagr, axis=1)
    return cagr_df.reset_index()

def plot_by_specialty(
    util_dfs, specialty_names, y_col, y_label, groupby_key='procedure', figsize=(25, 10), 
    format_as_dollar=False, title=None, legend_title='Procedure', legend_label_col=None
):
    n_procedures = max(df[groupby_key].nunique() for df in util_dfs)
    palette = sns.color_palette("tab10", n_procedures)

    fig, axs = plt.subplots(1, len(util_dfs), figsize=figsize)
    sns.set_theme(style="whitegrid", font_scale=1.4)

    for i, specialty in enumerate(util_dfs):
        grouped = specialty.groupby(groupby_key)
        for p_num, (label, df) in enumerate(grouped):
            df_sorted = df.sort_values('year')
            
            # Should be all the same, so just take the first row
            if legend_label_col is not None and legend_label_col in df_sorted:
                label = df_sorted.iloc[0][legend_label_col]
            
            axs[i].plot(
                df_sorted['year'], df_sorted[y_col],
                label=label, marker='o', markersize=7,
                linestyle='-', linewidth=2, color=palette[p_num]
            )

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].legend(title=legend_title, frameon=False, fontsize=12, title_fontsize=13)
        axs[i].set_xlabel("Year", fontsize=20)
        axs[i].set_ylabel(y_label, fontsize=20)
        axs[i].set_title(f"{specialty_names[i]}", fontsize=18, fontweight='bold', pad=12)
        axs[i].tick_params(axis='both', labelsize=13)
        axs[i].set_ylim(bottom=0)
        if format_as_dollar:
            axs[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    if title is not None:
        plt.suptitle(title, fontsize=22, fontweight='bold')

    plt.tight_layout()

def plot_by_specialty_with_ci(
    util_dfs, specialty_names, y_col, ci_lower_col, ci_upper_col, y_label, title_prefix, 
    groupby_key='procedure', figsize=(25, 8), title=None, legend_title='Procedure', legend_label_col=None
):
    n_procedures = max(df[groupby_key].nunique() for df in util_dfs)
    palette = sns.color_palette("tab10", n_procedures)

    fig, axs = plt.subplots(1, len(util_dfs), figsize=figsize)
    sns.set_theme(style="whitegrid", font_scale=1.4)

    for i, specialty in enumerate(util_dfs):
        grouped = specialty.groupby(groupby_key)
        for p_num, (label, df) in enumerate(grouped):
            df_sorted = df.sort_values('year')
            x = df_sorted['year']
            y = df_sorted[y_col]
            yerr = [y - df_sorted[ci_lower_col], df_sorted[ci_upper_col] - y]

            # Should be all the same, so just take the first row
            if legend_label_col is not None and legend_label_col in df_sorted:
                label = df_sorted.iloc[0][legend_label_col]

            axs[i].errorbar(
                x, y, yerr=yerr, label=label,
                marker='o', markersize=7, linestyle='-',
                linewidth=2, color=palette[p_num], elinewidth=1.5,
                # capsize=4, capthick=1.5,
            )

        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

        handles, labels = axs[i].get_legend_handles_labels()
        # Strip error bars from legend handles
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
        axs[i].legend(handles, labels, title=legend_title, frameon=False, fontsize=12, title_fontsize=13)
        # axs[i].legend(title=groupby_key, frameon=False, fontsize=12, title_fontsize=13)
        
        axs[i].set_xlabel("Year", fontsize=20)
        axs[i].set_ylabel(y_label, fontsize=20)
        axs[i].set_title(f"{specialty_names[i]}", fontsize=18, fontweight='bold', pad=12)
        axs[i].tick_params(axis='both', labelsize=13)
        axs[i].set_ylim(bottom=0)
        axs[i].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    if title is not None:
        plt.suptitle(title, fontsize=22, fontweight='bold')
    
    plt.tight_layout()

def compute_annual_change(util_dfs, specialty_names, col, change_col_name, groupby_key='procedure', start_year=2013, end_year=2023):
    """
    Compute year-over-year change for a given column, interpolating missing years.

    Parameters
    ----------
    util_dfs : list of pd.DataFrame
    specialty_names : list of str
    col : str — column to compute changes for
    change_col_name : str — name for the output change column
    groupby_key : str
    start_year : int
    end_year : int

    Returns
    -------
    merged_dfs : list of pd.DataFrame — each input df merged with the change column
    """
    x_years = list(range(start_year, end_year + 1))
    changes = {name: {} for name in specialty_names}

    for i, specialty in enumerate(util_dfs):
        for label, df in specialty.groupby(groupby_key):
            df_sorted = df.sort_values('year')
            pairs = dict(zip(df_sorted['year'], df_sorted[col]))
            y_new = [pairs.get(year, np.nan) for year in x_years]
            delta = pd.Series(y_new).interpolate().diff().to_numpy()
            changes[specialty_names[i]][label] = delta

    change_records = [
        [x_years[i], proc, delta[i]]
        for specialty, data in changes.items()
        for proc, delta in data.items()
        for i in range(len(delta))
    ]
    change_df = pd.DataFrame(change_records, columns=['year', groupby_key, change_col_name])

    merged_dfs = [
        df.merge(change_df, on=[groupby_key, 'year'])
        for df in util_dfs
    ]

    return merged_dfs