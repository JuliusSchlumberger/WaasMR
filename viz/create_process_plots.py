import numpy as np
import pandas as pd
import gzip
import matplotlib.pyplot as plt
from waasmodel_v6.inputs_waasmodel import ModelInputs
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

stages = {'flood_agr':1,
          'multihaz_agr':2,
          'multihaz_multisec': 3}

cc_scenario = 'Wp'
climvar = 4

def filter_numbers(numbers, sh_pair):
    if not numbers:
        return []

    # Start with the first number in the list
    filtered = [numbers[0]]
    threshold = ModelInputs(stage=stages[sh_pair]).yrs_btw_nmeas
    # Iterate over the rest of the list
    for num in numbers[1:]:
        if num - filtered[-1] > threshold:
            filtered.append(num)

    return filtered

def unique_event(sh_pair, realization, portfolio):
    # load & convert model output
    fname = f'model_outputs/{str(realization).zfill(6)}/stage_{stages[sh_pair]}/{sh_pair}/portfolio_{portfolio}.csv'
    with gzip.open(f'{fname}.gz', 'rb') as f:
        df = pd.read_csv(f)
    df.to_csv(fname)

    # Create figure with specified size
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
    axes = ax.ravel()

    # Select relevant values
    df_specific = df[(df.cc_scenario == cc_scenario) & (df.climvar == 4)]
    df_specific['actual_years'] = df_specific['year'] / 36

    # Discharge and Dike failures
    df_q = df_specific[df_specific.output == 'ClimQmax']
    df_fails = df_specific[df_specific.output == 'DikeFails']
    df_fails = df_fails[df_fails.value > 0]
    years_of_failure = df_fails.year


    df_fails = df_q[df_q.year.isin(years_of_failure)]

    # Plotting Discharge and Dike failures
    axes[0].plot(df_q.actual_years, df_q.value, label='Discharge', color='blue')
    axes[0].scatter(df_fails.actual_years, df_fails.value, color='red', label='Flood Events')
    axes[0].set_title('Discharge and Flood Events Over Time')
    axes[0].set_ylabel('Discharge [m3/s]')
    axes[0].legend()
    axes[0].grid(True)

    # Damage and ATP
    df_damage = df_specific[df_specific.output == 'DamAgr_f']
    df_atp = df_specific[df_specific.output == 'f_a_decision_value']

    df_threshold_avg = ModelInputs(stage=stages[sh_pair]).f_a_tp_cond
    df_threshold_extreme = ModelInputs(stage=stages[sh_pair]).f_a_tp_extreme_year

    df_implementations_extreme = df_damage[df_damage.value > df_threshold_extreme]
    df_extreme_times = df_implementations_extreme.actual_years.unique()

    df_implementations_avg = df_atp[df_atp.value > df_threshold_avg]
    df_avg_times = df_implementations_avg.actual_years.unique()

    # get periods in which no measures are implemented
    merged_timings = list(df_avg_times) + list(df_extreme_times)
    merged_timings = sorted(merged_timings)
    new_list = filter_numbers(merged_timings, sh_pair)
    df_avg_times_clean = [time for time in df_avg_times if time in new_list]
    df_extreme_times_clean = [time for time in df_extreme_times if time in new_list]

    df_impl_avg = df_implementations_avg[df_implementations_avg.actual_years.isin(df_avg_times_clean)]
    df_impl_extreme = df_implementations_extreme[df_implementations_extreme.actual_years.isin(df_extreme_times_clean)]


    # Plotting Damage, ATP, and thresholds
    axes[1].plot(df_damage.actual_years, df_damage.value, label='Annual Damage [MEur]', color='green')
    axes[1].plot(df_atp.actual_years, df_atp.value, label='Average Damage [MEur/15yr]', color='purple')
    axes[1].axhline(y=df_threshold_avg, color='orange', linestyle='--', label='ATP (Average)')
    axes[1].axhline(y=df_threshold_extreme, color='red', linestyle='--', label='ATP (Extreme)')

    axes[1].scatter(df_impl_avg.actual_years, df_impl_avg.value, color='red',marker='X', label='Measure Implementation')
    axes[1].scatter(df_impl_extreme.actual_years, df_impl_extreme.value, color='red',marker='X')

    for ts in new_list:
        axes[1].axvspan(ts, ts + ModelInputs(stage=stages[sh_pair]).yrs_btw_nmeas, color='grey', alpha=0.3)

    # Create a proxy artist for the legend
    highlight_patch = mpatches.Patch(color='grey', alpha=0.3, label='Years without new measures')

    # Get current handles and labels from the legend of axes[0]
    current_handles, current_labels = axes[1].get_legend_handles_labels()

    # Append the new handle and label to the existing ones
    current_handles.append(highlight_patch)
    current_labels.append('Years without new measures')

    # Reset the legend with the combined handles and labels
    axes[1].legend(handles=current_handles, labels=current_labels, loc='upper right')


    axes[1].set_title('Damage and Adaptation Tipping Points (ATP) Over Time')
    axes[1].set_ylabel('[MEur]')
    # axes[1].legend()
    axes[1].grid(True)

    # Additional formatting
    # fig.suptitle('Analysis of Climate Scenario and Decisions')
    axes[1].set_xlabel('Year')


    plt.tight_layout()
    plt.savefig(f'model_outputs/{str(realization).zfill(6)}/Process_plot_{sh_pair}_{portfolio}.png', dpi=300)

def across_uncertainty(sh_pair, realization, portfolio):
    # load & convert model output
    fname = f'model_outputs/{str(realization).zfill(6)}/stage_{stages[sh_pair]}/{sh_pair}/portfolio_{portfolio}.csv'
    with gzip.open(f'{fname}.gz', 'rb') as f:
        df = pd.read_csv(f)
    df.to_csv(fname)
    key_values = ['ClimQmax', 'DikeFails', 'DamAgr_f', 'f_a_decision_value']

    df_special = df[df.output.isin(key_values)]
    df_special['value'] = df_special['value'].astype(float)

    df_special['actual_years'] = df_special['year'] / 36

    # Discharge and Dike failures
    df_q = df_special.loc[df_special.output == 'ClimQmax']
    df_fails = df_special.loc[(df_special.output == 'DikeFails') & (df_special.value > 0)]

    df_fails.loc[:, 'value'] = 1

    # Damage and ATP
    df_damage = df_special[df_special.output == 'DamAgr_f']
    df_damage = df_damage[df_damage.value > 0]
    df_atp = df_special[df_special.output == 'f_a_decision_value']

    df_threshold_avg = ModelInputs(stage=stages[sh_pair]).f_a_tp_cond
    df_threshold_extreme = ModelInputs(stage=stages[sh_pair]).f_a_tp_extreme_year

    df_implementations_extreme = df_damage[df_damage.value > df_threshold_extreme]
    extr_grouped = df_implementations_extreme.groupby(['cc_scenario', 'climvar']).actual_years.unique()
    extr_grouped = extr_grouped.reset_index()
    # df_extreme_times = df_implementations_extreme.actual_years.unique()

    df_implementations_avg = df_atp[df_atp.value > df_threshold_avg]
    avg_grouped = df_implementations_avg.groupby(['cc_scenario', 'climvar']).actual_years.unique()
    avg_grouped = avg_grouped.reset_index()
    all_avg = []
    all_extreme = []
    # Convert to set for faster membership checking
    extr_grouped_set = extr_grouped.set_index(['cc_scenario', 'climvar'])['actual_years'].to_dict()

    for _, row in avg_grouped.iterrows():
        avg = set(row['actual_years'])
        avg = set(filter_numbers(list(avg), sh_pair))
        # Use .get to fetch the value or an empty set if the key does not exist
        extreme = set(extr_grouped_set.get((row['cc_scenario'], row['climvar']), []))
        extreme = set(filter_numbers(list(extreme), sh_pair))
        merged_timings = sorted(avg.union(extreme))  # union returns unique elements
        new_list = filter_numbers(merged_timings, sh_pair)
        new_set = set(new_list)
        # Intersection of sets
        df_avg_times_clean = avg.intersection(new_set)
        df_extreme_times_clean = extreme.intersection(new_set)
        all_avg.extend(df_avg_times_clean)
        all_extreme.extend(df_extreme_times_clean)

    # Convert back to lists if necessary
    all_avg = list(all_avg)
    all_extreme = list(all_extreme)

    df_all_avg = pd.DataFrame()
    df_all_avg['actual_years'] = all_avg
    df_all_avg['spec'] = 2

    df_all_extreme = pd.DataFrame()
    df_all_extreme['actual_years'] = all_extreme
    df_all_extreme['spec'] = 1


    # Figure
    fig, ax = plt.subplots(nrows=4, sharex=True, sharey=False, figsize=(15, 10),gridspec_kw={'height_ratios': [2, 1,2, 1]})
    axes = ax.ravel()

    # Main title
    fig.suptitle('Data Analysis of Model Outputs', fontsize=16, fontweight='bold')

    # Discharge and Dike failures
    sns.histplot(df_fails, x='actual_years', y='value', bins=(100, 30), color='red', ax=axes[0], cbar=True)
    sns.histplot(df_q, x='actual_years', y='value', bins=(100, 30), color='blue', ax=axes[0], cbar=True)
    axes[0].set_title('Discharge')
    axes[0].set_ylabel('Discharge [m3/s]')
    # axes[0].set_ylim([0, 25000])
    # Text for colorbars
    fig.text(0.70, 0.93, 'Density for Discharge', ha='center')
    fig.text(0.85, 0.93, 'Density for Flood Events', ha='center')

    # Discharge and Dike failures
    sns.histplot(df_fails, x='actual_years', y='value', bins=(100, 30), color='blue', ax=axes[1], cbar=True)
    sns.histplot(df_fails, x='actual_years', y='value', bins=(100, 30), color='red', ax=axes[1], cbar=True)
    axes[1].set_title('Flood Events')
    axes[1].set_ylabel('[-]')
    # axes[0].set_ylim([0, 25000])
    # Text for colorbars
    fig.text(0.70, 0.64, 'Density for Flood Events', ha='center')
    fig.text(0.85, 0.64, 'Density for Discharge', ha='center')

    # Damage and ATP
    sns.histplot(df_damage, x='actual_years', y='value', bins=(100, 30), color='purple', ax=axes[2], cbar=True)
    sns.histplot(df_atp, x='actual_years', y='value', bins=(100, 30), color='green', ax=axes[2], cbar=True)
    axes[2].set_title('Annual and Average Damages')
    axes[2].set_ylabel('[MEur]')
    # Text for colorbars
    fig.text(0.70, 0.478, 'Density for Average Damage', ha='center')
    fig.text(0.85, 0.478, 'Density for Annual Damage', ha='center')

    # Combined Data
    sns.histplot(df_all_extreme, x='actual_years', y='spec', bins=(100, 2), color='cyan', ax=axes[3], cbar=True)
    sns.histplot(df_all_avg, x='actual_years', y='spec', bins=(100, 2), color='magenta', ax=axes[3], cbar=True)
    axes[3].set_title('Implementation of measures based on the two ATP definitions')
    axes[3].set_ylabel('')
    # Text for colorbars
    # fig.text(0.92, 0.18, 'Density for Extreme\nDensity for Average', ha='center')
    fig.text(0.70, 0.188, 'Density for ATP (Average)', ha='center')
    fig.text(0.85, 0.188, 'Density for ATP (Extreme)', ha='center')

    # X-axis label for all subplots
    axes[-1].set_xlabel('Years')

    # Custom legends for the plots
    axes[0].legend(handles=[Line2D([0], [0], color='blue', lw=4, label='Discharge')])
    axes[1].legend(handles=[Line2D([0], [0], color='red', lw=4, label='Flood Events')])
    axes[2].legend(handles=[Line2D([0], [0], color='purple', lw=4, label='Damages (annual)'),
                            Line2D([0], [0], color='green', lw=4, label='Damages (average)')])
    axes[3].legend(handles=[Line2D([0], [0], color='cyan', lw=4, label='ATP (extreme)'),
                            Line2D([0], [0], color='magenta', lw=4, label='ATP (average)')])

    # Improve layout
    for ax in axes:
        ax.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for main title

    plt.tight_layout()
    plt.savefig(f'model_outputs/{str(realization).zfill(6)}/Process_Uncertainty_plot_{sh_pair}_{portfolio}.png', dpi=300)


realization = 8000
portfolio = '02_01_00_00'
# portfolio = '00_00_00_00'
sh_pairs = ['flood_agr','multihaz_agr','multihaz_multisec']
sh_pairs = ['multihaz_multisec']
for sh_pair in sh_pairs:
    unique_event(sh_pair=sh_pair, realization=realization, portfolio=portfolio)
    # across_uncertainty(sh_pair=sh_pair, realization=realization, portfolio=portfolio)