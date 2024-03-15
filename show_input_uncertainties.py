import pandas as pd
import numpy as np
import seaborn as sns
from viz.helper_outputprocessing import read_csv_file
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pcraster import *


cc_scenarios_labels = {'D': 'current climate',
                'G': 'low climate change',
                'Wp': 'high climate change'}

def process_dataframe_updated(dataframe):
    # Group by 'climvar' and 'cc_scenario'
    groups = dataframe.groupby(['climvar', 'cc_scenario'])

    # List to hold the processed dataframes
    processed_dfs = []

    for _, group in groups:
        # Drop rows where 'value' is 0
        group = group[group['value'] != 0]

        # Drop rows where the 'value' in the next row is larger
        group = group[group['value'].shift(-1) <= group['value']]

        # Append the processed group to the list
        processed_dfs.append(group)

    # Concatenate all processed groups into a single DataFrame
    return pd.concat(processed_dfs)




# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


def drought_frequency(d_filename):
    drought_info = read_csv_file(d_filename)

    # Apply the updated processing function to the DataFrame
    processed_df_updated = process_dataframe_updated(drought_info)
    processed_df_updated['year'] = processed_df_updated['year']/35
    processed_df_updated['value'] = processed_df_updated['value']

    # Identify unique values of 'cc_scenario'
    cc_scenarios = processed_df_updated['cc_scenario'].unique()

    # Number of unique cc_scenario values
    num_cc_scenarios = len(cc_scenarios)

    # First row: Histograms with bin centers at integers for each cc_scenario
    bin_edges = np.arange(-5, 190, 10)  # Bin edges from -0.5 to 19.5


    # Create a figure with two rows of subplots
    fig, axes = plt.subplots(nrows=2, ncols=num_cc_scenarios, sharey='row', figsize=(15, 7))

    # First row: Histograms for each cc_scenario
    for i, cc in enumerate(cc_scenarios):
        axes[0, i].hist(processed_df_updated[processed_df_updated['cc_scenario'] == cc]['value'], bins=bin_edges)
        axes[0, i].set_title(f'Frequency Droughts ({cc_scenarios_labels[cc]} scenario)')
        axes[0, i].set_xlabel('Drought Condition Length [days]')
        if i == 0:
            axes[0, i].set_ylabel('Frequency')

        # Set x-axis to show integers from 0 to 19
        axes[0, i].set_xlim(0, 190)
        axes[0, i].set_xticks(range(0, 190,10))
        axes[0, i].set_ylim(0, 520)
        # plt.xticks()
        axes[0, i].tick_params(axis='x', labelrotation=45)
        # axes[0, i].set_xticks(range(0, 200, 10))

    # Second row: Scatterplots for each cc_scenario
    for i, cc in enumerate(cc_scenarios):
        subset = processed_df_updated[processed_df_updated['cc_scenario'] == cc]
        # # Calculate trend line
        slope, intercept, _, _, _ = linregress(subset['year'], subset['value'])
        trend_line = slope * subset['year'] + intercept
        axes[1, i].plot(subset['year'], trend_line, color="red")  # Trend line in red

        sns.histplot(subset, x="year", y="value", ax=axes[1, i], binwidth=(5, 10))
        axes[1, i].set_title(f'Temporal distribution ({cc_scenarios_labels[cc]} scenario)')
        axes[1, i].set_xlabel('Years into the future')
        if i == 0:
            axes[1, i].set_ylabel('Drought Condition Length [days]')
        axes[1, i].set_ylim(0, 190)
        axes[1, i].set_yticks(range(0, 190,20))
    fig.suptitle('Drought Event Lenghts - Frequency and Temporal distribution', size=16)
    plt.tight_layout()
    fig.savefig(f'{inputfolder}/drought_occurences.png', dpi=300)

def get_flood_count_quantiles(df, output):
    df_lineplot = df[df.output == output]
    df_lineplot['year'] = df_lineplot['year'] / 35
    df_lineplot['count'] = (df_lineplot['value'] != 0).astype(int)

    df_lineplot['count'] = df_lineplot.groupby(['cc_scenario', 'climvar'])['count'].cumsum()
    return df_lineplot


def flood_frequency(f_filename):
    flood_info = read_csv_file(f_filename)
    df_lineplot = get_flood_count_quantiles(flood_info,  'floodtiming')
    grouped = df_lineplot.groupby(['cc_scenario', 'year'])['count'].quantile([.1, .5, .9])
    grouped = grouped.reset_index()

    # Unique cc_scenario values for creating subplots
    cc_scenarios = grouped['cc_scenario'].unique()

    # Creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(cc_scenarios), figsize=(15, 5), sharey=True)

    for i, cc in enumerate(cc_scenarios):
        subset = grouped[grouped['cc_scenario'] == cc]
        median_values = subset[subset['level_2'] == 0.5]
        lower_values = subset[subset['level_2'] == 0.1]
        upper_values = subset[subset['level_2'] == 0.9]

        axes[i].plot(median_values['year'], median_values['count'], label='Median')
        axes[i].fill_between(median_values['year'], lower_values['count'], upper_values['count'], alpha=0.3)
        axes[i].set_title(f'{cc_scenarios_labels[cc]} scenario')
        axes[i].set_xlabel('Years into the future')
        axes[i].grid()
        if i == 0:
            axes[i].set_ylabel('Count Flood events')
    fig.suptitle('Accumulated Flood counts (with uncertainty bands)', size=16)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'{inputfolder}/flood_occurence.png', dpi=300)

def flood_vs_multihazard(mh_flood, flood):
    flood_info = read_csv_file(flood)
    mh_info = read_csv_file(mh_flood)
    df_floodplot = get_flood_count_quantiles(flood_info,  'floodtiming')
    df_mhplot = get_flood_count_quantiles(mh_info,  'floodtiming')

    sh_floods = df_floodplot.groupby(['cc_scenario', 'year'])['count'].quantile([.1, .5, .9])
    sh_floods = sh_floods.reset_index()

    mh_floods = df_mhplot.groupby(['cc_scenario', 'year'])['count'].quantile([.1, .5, .9])
    mh_floods = mh_floods.reset_index()

    # Unique cc_scenario values for creating subplots
    cc_scenarios = mh_floods['cc_scenario'].unique()

    # Creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(cc_scenarios), figsize=(15, 5), sharey=True)

    for i, cc in enumerate(cc_scenarios):
        sh_subset = sh_floods[sh_floods['cc_scenario'] == cc]
        sh_median_values = sh_subset[sh_subset['level_2'] == 0.5]
        sh_lower_values = sh_subset[sh_subset['level_2'] == 0.1]
        sh_upper_values = sh_subset[sh_subset['level_2'] == 0.9]

        mh_subset = mh_floods[mh_floods['cc_scenario'] == cc]
        mh_median_values = mh_subset[mh_subset['level_2'] == 0.5]
        mh_lower_values = mh_subset[mh_subset['level_2'] == 0.1]
        mh_upper_values = mh_subset[mh_subset['level_2'] == 0.9]

        axes[i].plot(sh_median_values['year'], sh_median_values['count'], label='Single-Hazard Flood Median')
        axes[i].fill_between(sh_median_values['year'], sh_lower_values['count'], sh_upper_values['count'],
                             alpha=0.3)

        axes[i].plot(mh_median_values['year'], mh_median_values['count'], label='Multi-Hazard Flood Median')
        axes[i].fill_between(mh_median_values['year'], mh_lower_values['count'], mh_upper_values['count'],
                             alpha=0.3)
        axes[i].set_title(f'{cc_scenarios_labels[cc]} scenario')
        axes[i].set_xlabel('Years into the future')
        axes[i].grid()
        if i == 0:
            axes[i].set_ylabel('Count Flood events')
    fig.suptitle('Comparing Multi-Hazard and Single-Hazard Flood Event Counts (with uncertainty bands)', size=16)
    plt.legend()

    plt.tight_layout()
    fig.savefig(f'{inputfolder}/flood_occurence_comparison.png', dpi=300)

def count_waterlogging_events(mh_filename):
    waterlogging_info = read_csv_file(mh_filename)
    waterlogging_info = waterlogging_info[waterlogging_info.output == 'agri_areas_exposed_count']
    waterlogging_info['value'] = waterlogging_info['value'].apply(lambda x: x if x >= 2000 else 0)

    df_lineplot = get_flood_count_quantiles(waterlogging_info, 'agri_areas_exposed_count')
    grouped = df_lineplot.groupby(['cc_scenario', 'year'])['count'].quantile([.1, .5, .9])
    grouped = grouped.reset_index()

    # Unique cc_scenario values for creating subplots
    cc_scenarios = grouped['cc_scenario'].unique()

    # Creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(cc_scenarios), figsize=(15, 5), sharey=True)

    for i, cc in enumerate(cc_scenarios):
        subset = grouped[grouped['cc_scenario'] == cc]
        median_values = subset[subset['level_2'] == 0.5]
        lower_values = subset[subset['level_2'] == 0.1]
        upper_values = subset[subset['level_2'] == 0.9]

        axes[i].plot(median_values['year'], median_values['count'], label='Median')
        axes[i].fill_between(median_values['year'], lower_values['count'], upper_values['count'],
                             alpha=0.3)
        axes[i].set_title(f'({cc_scenarios_labels[cc]} scenario)')
        axes[i].set_xlabel('Years into the future')
        axes[i].grid()
        if i == 0:
            axes[i].set_ylabel('Count Waterlogging Events')
    fig.suptitle('Accumulated count of Waterlogging Events (with uncertainty bands)', size=16)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'{inputfolder}/waterlogging_occurence.png', dpi=300)


inputfolder = 'inputs_visual/009999'

filename = 'portfolio_00_00_00_00.csv.gz'
# droughts
d_filename = f'{inputfolder}/stage_1/drought_agr/{filename}'
f_filename = f'{inputfolder}/stage_1/flood_agr/{filename}'
mh_filename = f'{inputfolder}/stage_2/multihaz_agr/{filename}'

drought_frequency(d_filename)
flood_frequency(f_filename)
flood_vs_multihazard(mh_filename, f_filename)
count_waterlogging_events(mh_filename)

LandUse = readmap(r"{}".format('waasmodel_v6/inputs/maps/land_ini.pcr'))
agriculturemap = LandUse == 9  # Veerle Make map with only the agricultural areas
count_agri_area = pcr2numpy(agriculturemap,0).sum()
print(count_agri_area)