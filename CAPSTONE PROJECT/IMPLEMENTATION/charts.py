"""
Matplotlib Chart Generation Module for Graphical Analysis
=========================================================
Module 4: Generates static high-resolution visualization charts
and saves them to the static/charts/ directory for dashboard presentation.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server environment
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'charts')


def ensure_charts_dir():
    """Creates the charts directory if missing."""
    os.makedirs(CHARTS_DIR, exist_ok=True)


def generate_all_charts(df):
    """
    Renders all analytical charts from the predictions DataFrame.
    Returns:
        dict: Mapping of chart names to relative static URLs, or None if df is empty.
    """
    ensure_charts_dir()

    if df is None or len(df) == 0:
        return {
            'distribution_chart': None,
            'daily_trend_chart': None,
            'tod_breakdown_chart': None
        }

    # Custom modern styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    font_family = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']

    chart_paths = {}

    # 1. Distribution Donut / Bar Chart
    dist_file = os.path.join(CHARTS_DIR, 'distribution_chart.png')
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    
    categories = ['Low', 'Medium', 'High']
    colors = ['#10B981', '#F59E0B', '#EF4444']
    counts = [
        (df['Predicted_Category'] == 'Low').sum(),
        (df['Predicted_Category'] == 'Medium').sum(),
        (df['Predicted_Category'] == 'High').sum()
    ]

    total_pts = sum(counts)
    if total_pts > 0:
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=categories,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct*total_pts/100))})" if pct > 0 else '',
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
        )
        for t in texts:
            t.set_fontsize(11)
            t.set_fontweight('bold')
        for at in autotexts:
            at.set_fontsize(10)
            at.set_color('#1e293b')
            at.set_fontweight('bold')
        ax.set_title("Energy Prediction Category Distribution", fontsize=13, fontweight='bold', pad=15, color='#0f172a')
    else:
        ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', fontsize=12)

    plt.tight_layout()
    fig.savefig(dist_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    chart_paths['distribution_chart'] = 'charts/distribution_chart.png'

    # 2. Daily Prediction Trends (Stacked Bar)
    daily_file = os.path.join(CHARTS_DIR, 'daily_trend_chart.png')
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Extract date
    df_copy = df.copy()
    df_copy['Date'] = df_copy['Date_Time'].str.slice(0, 10)
    daily_agg = df_copy.groupby(['Date', 'Predicted_Category']).size().unstack(fill_value=0)
    for col in categories:
        if col not in daily_agg.columns:
            daily_agg[col] = 0
    daily_agg = daily_agg[categories]  # Order correctly
    daily_agg = daily_agg.tail(10)     # Most recent 10 days

    dates = daily_agg.index.tolist()
    x = np.arange(len(dates))
    width = 0.55

    bottom = np.zeros(len(dates))
    for i, cat in enumerate(categories):
        vals = daily_agg[cat].values
        ax.bar(x, vals, width, bottom=bottom, label=cat, color=colors[i], edgecolor='white', linewidth=1)
        bottom += vals

    ax.set_title("Daily Predictions Breakdown (Recent Days)", fontsize=13, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xticks(x)
    ax.set_xticklabels([d[5:] for d in dates], rotation=0, fontsize=10)
    ax.set_ylabel("Number of Predictions", fontsize=11, color='#334155')
    ax.set_xlabel("Date (MM-DD)", fontsize=11, color='#334155')
    ax.legend(title="Category", loc='upper left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(daily_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    chart_paths['daily_trend_chart'] = 'charts/daily_trend_chart.png'

    # 3. Consumption by Time of Day
    tod_file = os.path.join(CHARTS_DIR, 'tod_breakdown_chart.png')
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    tod_agg = df_copy.groupby(['Time_Of_Day', 'Predicted_Category']).size().unstack(fill_value=0)
    for col in categories:
        if col not in tod_agg.columns:
            tod_agg[col] = 0
    tod_agg = tod_agg.reindex(tod_order).fillna(0)

    x = np.arange(len(tod_order))
    bar_width = 0.25

    ax.bar(x - bar_width, tod_agg['Low'], bar_width, label='Low', color='#10B981', edgecolor='white')
    ax.bar(x, tod_agg['Medium'], bar_width, label='Medium', color='#F59E0B', edgecolor='white')
    ax.bar(x + bar_width, tod_agg['High'], bar_width, label='High', color='#EF4444', edgecolor='white')

    ax.set_title("Prediction Categories across Time of Day", fontsize=13, fontweight='bold', pad=15, color='#0f172a')
    ax.set_xticks(x)
    ax.set_xticklabels(tod_order, fontsize=10)
    ax.set_ylabel("Prediction Count", fontsize=11, color='#334155')
    ax.legend(title="Category", loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(tod_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    chart_paths['tod_breakdown_chart'] = 'charts/tod_breakdown_chart.png'

    return chart_paths
