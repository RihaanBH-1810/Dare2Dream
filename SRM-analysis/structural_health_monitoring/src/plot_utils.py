""".
 
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")

import itertools
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

# general plot configuration
SMALL_SIZE = 10
MEDIUM_SIZE = 16
LARGE_SIZE = 24
HUGE_SIZE = 32
FIG_SIZE = (22, 8)

def init_plotting():
    '''
    '''
    # Matplotlib
    plt.rc('figure', figsize=FIG_SIZE)        # default figure size
    plt.rc('figure', titlesize=HUGE_SIZE)     # fontsize of the figure title
    plt.rc('figure', titleweight='bold')      # weight of the figure title
    plt.rc('font', size=LARGE_SIZE)          # default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)      # fontsize of the axes title
    plt.rc('axes', titleweight='bold')        # weight of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LARGE_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LARGE_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LARGE_SIZE)    # legend fontsize
    plt.rc('legend', title_fontsize=LARGE_SIZE)    # legend fontsize
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300    
    
    
    # Seaborn
    sns.set(rc={"figure.figsize": FIG_SIZE,
                "figure.titlesize": HUGE_SIZE,
                "figure.titleweight": 'bold',
                "font.size": LARGE_SIZE,
                "axes.titlesize": LARGE_SIZE,
                "axes.titleweight": 'bold',
                "axes.labelsize": LARGE_SIZE,
                "xtick.labelsize": LARGE_SIZE,
                "ytick.labelsize": LARGE_SIZE,
                "legend.fontsize": LARGE_SIZE,
                "legend.title_fontsize": LARGE_SIZE,
                "figure.dpi": 300,
                "savefig.dpi": 300,
               })    

    
def plot_embedding_targets(X_embedded, y, alpha=1., palette=None):
    '''
    '''
    fig = plt.figure()
    
    cntr = Counter(y)
    if palette is None:
        palette = sns.color_palette("bright", len(cntr.keys()))

    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y,
                    legend='full', palette=palette, alpha=alpha)
    
    return fig



def plot_classwise_dist(df, x=None, label_col=None, palette=None):
    '''
    '''
    fig = plt.figure()
    
    if x is None:
        x = range(df.shape[1]-1)
    
    # last column contains the labels if not handled otherwise
    if label_col is None:
        label_col = df.columns[-1]
        
    # get all unique labels
    cntr = Counter(df[label_col])

    if palette is None:
        palette = sns.color_palette("bright", len(cntr.keys()))

    for i, (key, group) in enumerate(df.groupby(label_col)):
        # plot mean values
        lbl = f"{key:}_mean" if type(df[label_col].values[0])==str else f"{key:.2f}_mean"
        mean_spec = group.drop(columns=label_col).mean(axis=0)
        sns.lineplot(x=x, y=mean_spec, color=palette[i],
                     label=lbl)

        # plot std values
        std_spec = group.drop(columns=label_col).std(axis=0)
        lower_bound = mean_spec-std_spec
        upper_bound = mean_spec+std_spec
        lbl = f"{key:}_std" if type(df[label_col].values[0])==str else f"{key:.2f}_std"
        plt.fill_between(x, lower_bound, upper_bound,
                         color=palette[i], alpha=.1, label=lbl)
    
    return fig


def plot_classwise_kde(df, label_col, labels, palette=None, feature_idx=0, focus=-1, focus_lw=5):    
    '''
    '''
    fig = plt.figure()

    focus_label = 'None'

    x = df.columns[feature_idx]
    
    if focus>=0:
        focus_label = labels[focus]

    if palette is None:
        # get all unique labels
        cntr = Counter(df[label_col])
        palette = sns.color_palette("bright", len(cntr.keys()))

    for i, (key, group) in enumerate(df.groupby(label_col)):
        lw = None # default linewidth
        # plot focused kde with thicker line
        if key == focus_label:
            lw = focus_lw # focused linewidth
        
        lbl = f"{key:}" if type(df[label_col].values[0])==str else f"{key:.2f}_mean"
        sns.kdeplot(data=group, x=x, linewidth=lw, color=palette[i],
                          label=lbl)
        
    plt.legend()
    
    return fig


def plot_metrics(metrics: pd.DataFrame):
    '''
    '''
    fig = sns.boxplot(x="model", y="values", hue="metrics", data=metrics)
    
    return fig


def plot_metrics_bar(metrics: pd.DataFrame,
                     palette="Blues",
                     ci="sd",
                     errwidth=3.0,
                     capsize=.2):
    '''
    '''
    fig = sns.barplot(x="model",
                      y="values",
                      hue="metrics",
                      palette=palette,
                      data=metrics,
                      ci=ci,
                      errwidth=errwidth,
                      capsize=capsize)
    
    return fig


def get_metrics_summary(metrics: pd.DataFrame, summaries=['mean', 'std']):
    '''
    '''
    summary = metrics.groupby(['model', 'metrics'])['values'].describe()
    if summaries is not None:
        summary = summary[summaries]
    return summary

def get_sorted_metrics_summary(metrics: pd.DataFrame, summaries=['mean', 'std']):
    '''
    '''
    if summaries is not None:
        summary = summary[summaries]
    return summary


def plot_cv_indices(cv, X, y, ax, lw=10):
    """
    """
    if type(cv) is not list:
        cv = list(cv.split(X, y)).copy()
        
    n_splits = len(cv)
    
    # Generate the training/testing visualizations for each CV split
    for i, (train_idx, test_idx) in enumerate(cv):
        # translate indices to positions
        indices = np.array([np.nan] * len(X))
        indices[test_idx] = 1
        indices[train_idx] = 0

        ax.scatter(
            range(len(indices)),
            [i + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2,
        )

    cmap_data = plt.cm.Paired
    ax.scatter(
        range(len(X)), [i + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
    )

    yticklabels = list(range(n_splits)) + ["class"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2],
    )
    
    return ax
