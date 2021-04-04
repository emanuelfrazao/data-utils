import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_proportions_by_feature(df, feature, target, target_mean=None):
    
    # Get counts by feature and target
    df_local = df.groupby([feature, target]) \
        .count() \
        .iloc[:,:1]

    df_local = df_local.rename(columns = {df_local.columns[0]: 'count'}) \
        .reset_index() \
        .pivot_table(columns=target, values='count', index=feature)
    
    # Set up figure
    x_figsize = min(16, df[feature].unique().size * 2 + 2)
    
    fig, axes = plt.subplots(2,1, figsize = (x_figsize, 12))
    fig.subplots_adjust(top = 0.9, hspace=0.3)
    
    # Get axes' titles
    top_title = r"Proportion of $\it{{{0}}}$ values with respect to $\it{{{1}}}$'s categories." \
        .format(target.replace('_','\_'), feature.replace('_','\_'))
    bottom_title = r"Total count of $\it{{{0}}}$'s categories." \
        .format(feature.replace('_','\_'))
    
    # Draw top plot (proportions)
    df_local.apply(lambda x: x/sum(x), axis=1).plot.bar(stacked=True, ax=axes[0], rot=0)
    axes[0].set_ylabel('proportion')
    axes[0].set_title(top_title)
    
    if target_mean:
        # Draw targer mean
        axes[0].axhline(target_mean, color='green', ls='--')
    
    # Draw bottom plot (counts)
    df_local.apply(lambda x: sum(x), axis=1).plot.bar(ax=axes[1], rot=0)
    axes[1].set_ylabel('count')
    axes[1].set_title(bottom_title)




def plot_proportions_combined_by_features(df, first, second, target, target_mean=None):
    
    # Get counts by (first, second) and target
    def get_counts_inplace(df_, x, target):
        df_local = df_.groupby([x, target]) \
            .count() \
            .iloc[:,:1]

        df_local = df_local.rename(columns = {df_local.columns[0]: 'count'}) \
            .reset_index() \
            .pivot_table(columns=target, values='count', index=x)
        
        return df_local
        
    df_combined = df.join(df.apply(lambda x: (x[first], x[second]), axis=1).rename('combined'))
    
    df_local = get_counts_inplace(df_combined, 'combined', target)
    
    # Set up figure
    x_figsize = min(16, df_local.shape[0] * 2 + 2)
    
    fig, axes = plt.subplots(2,1, figsize = (x_figsize, 12))
    fig.subplots_adjust(top = 0.9, hspace=0.3)
    
    # Get axes' titles
    top_title = r"Proportion of $\it{{{0}}}$ values with respect to combined categories of $\it{{{1}}}$ and $\it{{{2}}}$." \
        .format(target.replace('_','\_'), first.replace('_','\_'), second.replace('_','\_'))
    bottom_title = r"Total count of combined categories of $\it{{{0}}}$ and $\it{{{1}}}$." \
        .format(first.replace('_','\_'), second.replace('_','\_'))
    
    # Draw top plot (proportions)
    df_local.apply(lambda x: x/sum(x), axis=1).plot.bar(stacked=True, ax=axes[0], rot=0)
    axes[0].set_ylabel('proportion')
    axes[0].set_xlabel(f'({first}, {second})')
    axes[0].set_title(top_title)
    
    if target_mean:
        # Draw target mean
        axes[0].axhline(target_mean, color='green', ls='--')
    
    # Draw bottom plot (counts)
    df_local.apply(lambda x: sum(x), axis=1).plot.bar(ax=axes[1], rot=0)
    axes[1].set_ylabel('count')
    axes[1].set_xlabel(f'({first}, {second})')
    axes[1].set_title(bottom_title)





def plot_proportions_by_bin(df, feature, target, nbins=21):
    
    def count_bins(feature, min_value, max_value, nbins=nbins):
        # Set up bins
        bins = np.linspace(np.floor(min_value), np.ceil(max_value), nbins)

        # Get the indexes in bins for each value
        indexes = np.searchsorted(bins, feature)

        # Count the values for each bin
        counts = np.zeros_like(bins)
        np.add.at(counts, indexes, 1)

        return pd.Series(data=counts, index=[int(x) for x in bins])

    # Get extremes
    min_value = df[feature].min()
    max_value = df[feature].max()
    
    # Get feature's values for each target class
    feature_values_by_class = {v: df[df[target] == v][feature].rename(str(v)) for v in sorted(df[target].unique())}
    
    feature_counts = {idx: count_bins(values, min_value, max_value).rename(str(idx)) for idx, values in feature_values_by_class.items()}
    
    df_local = pd.DataFrame(feature_counts)
    
    # Set up figure
    x_figsize = min(16, df[feature].unique().size * 2 + 2)
    
    fig, axes = plt.subplots(2,1, figsize = (x_figsize, 12))
    fig.subplots_adjust(top = 0.9, hspace=0.3)
    
    # Get axes' titles
    top_title = r"Proportion of $\it{{{0}}}$ values with respect to $\it{{{1}}}$'s bins." \
        .format(target.replace('_','\_'), feature.replace('_','\_'))
    bottom_title = r"Respective histogram of $\it{{{0}}}$." \
        .format(feature.replace('_','\_'))
    
    # Draw top plot (proportions)
    df_local.apply(lambda x: x/sum(x), axis=1).plot.bar(stacked=True, ax=axes[0], rot=0)
    axes[0].set_ylabel('proportion')
    axes[0].set_title(top_title)
    
    # Draw bottom plot (counts)
    df_local.apply(lambda x: sum(x), axis=1).plot.bar(ax=axes[1], rot=0)
    axes[1].set_ylabel('count')
    axes[1].set_title(bottom_title)






def plot_numerical_vs_categoricals(df, categoricals, numerical, ncols = 4):
    # Set up figure
    nrows = (len(categoricals)-1) // ncols + 1
    
    figsize = (ncols*4 + 1, nrows*4)
    
    fig, axes = plt.subplots(nrows, ncols, figsize = figsize, sharey=True)
    fig.subplots_adjust(top = 0.9, hspace=0.3, wspace=0.1)

    # Set figure's title
    figure_title = r'Confront $\it{{{0}}}$ with all categorical features'.format(numerical.replace('_','\_'))
    fig.suptitle(figure_title)

    # Set each subplot
    for idx, categorical in enumerate(categoricals):
        
        if nrows > 1:
            # Get indexes of the corresponding ax
            ax_row = idx // 4
            ax_col = idx % 4

            # Create the plot for the corresponding ax
            sns.boxplot(
                ax=axes[ax_row, ax_col], 
                data=df, 
                x=categorical, 
                y=numerical,
                order=sorted(list(df[categorical].unique()))
            )

            # Set axes subtitles and labels
            axes[ax_row,ax_col].set_title(categorical)

            axes[ax_row,ax_col].set_xlabel(None)
            
            if ax_col == 0:
                axes[ax_row,ax_col].set_ylabel(numerical)
            else:
                axes[ax_row,ax_col].set_ylabel(None)
            
            # Rotate overcrowded ticklabels
            if df[categorical].unique().size > 8:
                axes[ax_row,ax_col].set_xticklabels(axes[ax_row,ax_col].get_xticklabels(), rotation=45)
    