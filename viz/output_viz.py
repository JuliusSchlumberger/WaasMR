from waasmodel_v6.inputs_waasmodel import *
from waasmodel_v6.inputs_runclass import RunClassInputs
from viz.helper_outputprocessing import (create_subfolder, convert_files)
from matplotlib.tri import Triangulation
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

pathways_path = 'waasmodel_v6/inputs/'

pd.set_option('display.max_rows', None)  # or a specific large number like 500
pd.set_option('display.max_columns', None)  # or a specific large number like 50
pd.set_option('display.width', None)  # or a specific width like 1000
pd.set_option('display.max_colwidth', None)  # or a specific value like 500

# Given data in the form of lists (as represented from a table)
method_names = [
    "d_resilient_crops", "d_rain_irrigation", "d_gw_irrigation", "d_riv_irrigation",
    "d_soilm_practice", "d_multimodal_transport", "d_medium_ships", "d_small_ships",
    "d_dredging", "f_resilient_crops", "f_ditches", "f_local_support", "f_dike_elevation_s",
    "f_dike_elevation_l", "f_maintenance", "f_room_for_river", "f_wet_proofing_houses",
    "f_local_protect", "f_awareness_campaign", "no_measure"
]
portfolio_list = {'flood_agr': ['fa_p'],
                          'drought_agr': ['da_p'],
                          'flood_urb': ['fu_p'],
                          'drought_shp': ['ds_p'],
                          'multihaz_agr': ['fa_p', 'da_p'],
                          'multihaz_urb': ['fu_p'],
                          'multihaz_multisec': ['fa_p', 'da_p', 'fu_p', 'ds_p']}

names = [
    "Drought resilient crops", "Rainwater irrigation", "Groundwater irrigation", "River irrigation",
    "Soil moisture practice", "Multi-modal transport subsidies", "Fleet of medium size ships",
    "Fleet of small size ships", "River dredging", "Flood resilient crops", "Ditch system",
    "Local support conservation scheme", "Small dike elevation increase", "Large dike elevation increase",
    "Dike maintenance", "Room for the River", "Flood proof houses", "Local protection", "Awareness campaign", "no_measure"
]

# Create dictionary with method_name as keys and Name as values
method_dict = dict(zip(method_names, names))

measure_logos = {measure: f'viz/logos/{measure}.png' for measure in ModelInputs(stage=1).measure_numbers.keys()}


# extra functions
def insert_linebreak(s, max_length=20):
    if len(s) <= max_length:
        return s

    # Find the nearest space before or at max_length
    break_point = max_length
    while s[break_point] != ' ' and break_point > 0:
        break_point -= 1

    if break_point == 0:
        return s  # No space found; return the original string

    # Insert line break
    return s[:break_point] + '\n' + s[break_point + 1:]


def add_logo_and_text(ax, img_path, x, y, text):
    arr_img = plt.imread(img_path)
    imagebox = OffsetImage(arr_img, zoom=0.05)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False, xycoords=("axes fraction", "axes fraction"),
                        boxcoords="axes fraction", box_alignment=(0, 0.5))
    ax.add_artist(ab)
    text = insert_linebreak((text))

    ax.text(x + 0.05, y, text, transform=ax.transAxes, verticalalignment='center', fontsize=16)


title_list = {'fa_p': 'Agr. flood',
                                   'da_p': 'Agr. drought',
                                   'fu_p': 'Urban flood',
                                   'ds_p': 'Ship drought'}


class OutputPocessing():
    def __init__(self, stages, sectors_list, no_realizations, realization, model_output_dir, outputprocess_folder):
        runclass = RunClassInputs()
        self.keys_dict = runclass.keys_dict
        model_inputs = ModelInputs(stage=1)
        self.realization_numbers = model_inputs.measure_numbers

        self.stages = stages
        self.sectors_list = sectors_list
        self.realization = realization
        self.no_realizations = no_realizations

        self.model_output_dir = model_output_dir

        self.stages = {'flood_agr': 1,
                       'drought_agr': 1,
                       'flood_urb': 1,
                       'drought_shp': 1,
                       'multihaz_agr': 2,
                       'multihaz_urb': 2,
                       'multihaz_multisec': 3
                       }
        self.revenue_ref = 970.5    # Mio EUR
        self.cc_scenarios = ['D', 'G', 'Wp']
        self.sectors = {1: ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp'],
                        2: ['multihaz_agr', 'multihaz_urb'],
                        3: ['multihaz_multisec']}
        self.optimization = {'DamUrb_tot': 'min',
                             'DamAgr_f_tot': 'min',
                             'DamAgr_d_tot': 'min',
                             'DamShp_tot': 'min',
                             'invest_f_a': 'min',
                             'invest_f_u': 'min',
                             'invest_d_a': 'min',
                             'invest_d_s': 'min',
                             'invest_multihaz_agr': 'min',
                             'invest_multihaz_urb': 'min',
                             'invest_multihaz_multisec': 'min',
                             'maint_f_a': 'min',
                             'maint_f_u': 'min',
                             'maint_d_a': 'min',
                             'maint_d_s': 'min',
                             'maint_multihaz_agr': 'min',
                             'maint_multihaz_urb': 'min',
                             'maint_multihaz_multisec': 'min',
                             'revenue_agr': 'max',
                             'pathways_list_f_a': 'min',
                             'pathways_list_d_a': 'min',
                             'pathways_list_f_u': 'min',
                             'pathways_list_d_s': 'min',}

        self.tp_cond = {'flood_agr': [model_inputs.f_a_tp_cond],
                        'drought_agr': [model_inputs.d_a_tp_cond],
                        'flood_urb': [model_inputs.f_u_tp_cond],
                        'drought_shp': [model_inputs.d_s_tp_cond],
                        'multihaz_agr': [model_inputs.f_a_tp_cond, model_inputs.d_a_tp_cond],
                        'multihaz_urb': [model_inputs.f_u_tp_cond],
                        'multihaz_multisec': [model_inputs.f_a_tp_cond, model_inputs.d_a_tp_cond,
                                              model_inputs.f_u_tp_cond, model_inputs.d_s_tp_cond]
                        }
        self.measure_dict = {
            'flood_urb': [['no_measure']],
            'flood_agr': [['no_measure']],
            'drought_shp': [['no_measure']],
            'drought_agr': [['no_measure']],
            'multihaz_agr': [['no_measure']],
            'multihaz_urb': [['no_measure']],
            'multihaz_multisec': [['no_measure']]}


        self.sector_names = {'flood_agr': ['flood_agr'],
                             'drought_agr': ['drought_agr'],
                             'flood_urb': ['flood_urb'],
                             'drought_shp': ['drought_shp'],
                             'multihaz_agr': ['flood_agr', 'drought_agr'],
                             'multihaz_urb': ['flood_urb'],
                             'multihaz_multisec': ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp']
                             }

        self.process_folder = outputprocess_folder
        self.viz_folder = create_subfolder(self.process_folder, 'figures')

    def run_processing(self, difference_groups, robustness_heatmap=False, interactionplot=False,timingplot=False,  convert_files_switch=False):
        if convert_files_switch:
            convert_files(folder=self.process_folder)
        if robustness_heatmap:
            for sector in self.sectors_list:
                robustness_df = pd.read_csv(f'{self.process_folder}/all_robustness_mean_per_pathway_all_stages.csv')
                self.heatmap_allobjectives(robustness_df=robustness_df,sector=sector)
        if interactionplot:
            self.plot_measure_interaction_effects(interaction_stages=['multihaz_agr', 'multihaz_multisec']) # ['multihaz_agr', 'multihaz_multisec']
        if timingplot:
            for sector in self.sectors_list:
                print('sector', sector)
                timing_df = pd.read_csv(f'{self.process_folder}/combined_timings_stage_{self.stages[sector]}.csv')
                self.create_timingcomparison_plots(df_timings=timing_df, difference_groups=difference_groups)

    def heatmap_allobjectives(self, robustness_df, sector):
        ylabel_list = {'fa_p': 'flood_agr',
                       'da_p': 'drought_agr',
                       'fu_p': 'flood_urb',
                       'ds_p': 'drought_shp'}

        replacement_dict = {
            'DamAgr_f_tot': 'Struct. Damage\n[MEur]',
            'invest_f_a': 'Cost\n [MEur]',
            'maint_f_a': 'Maintenance\nCost [MEur]',
            'DamAgr_d_tot': 'Crop Loss\n[MEur]',
            'revenue_agr': 'Crop Loss\n[MEur]',
            'invest_d_a': 'Cost\n [MEur]',
            'maint_d_a': 'Maintenance\nCost [MEur]',
            'DamUrb_tot': 'Damage\n [MEur]',
            'invest_f_u': 'Cost\n [MEur]',
            'maint_f_u': 'Maintenance\nCost [MEur]',
            'DamShp_tot': 'Damage \n[MEur]',
            'invest_d_s': 'Cost\n [MEur]',
            'maint_d_s': 'Maintenance\nCost [MEur]',
            'fa_p': 'Agricultural flood pathways alternatives',
            'da_p': 'Agricultural drought pathways alternatives',
            'fu_p': 'Urban flood pathways alternatives',
            'ds_p': 'Shipping drought pathways alternatives',
            'total_system': 'needs_local_replacement',
            'value_aggregated': 'needs_local_replacement',
            'cc_scenario': 'Climate Change Scenario',
            'fa_p_obj': 'Agricultural flood objectives (aggregated)',
            'da_p_obj': 'Agricultural drought objectives (aggregated)',
            'fu_p_obj': 'Urban flood objectives (aggregated)',
            'ds_p_obj': 'Shipping drought objectives (aggregated)',
            'pathways_list_d_a': 'Number\nmeasures',
            'pathways_list_f_a': 'Number\nmeasures',
            'pathways_list_f_u': 'Number\nmeasures',
            'pathways_list_d_s': 'Number\nmeasures'

            }

        sector_objectives = {'fa_p': ['pathways_list_f_a', 'invest_f_a', 'DamAgr_f_tot', 'revenue_agr', ],
                             'da_p': ['pathways_list_d_a', 'invest_d_a', 'revenue_agr'],
                             'fu_p': ['pathways_list_f_u', 'invest_f_u', 'DamUrb_tot'],
                             'ds_p': ['pathways_list_d_s', 'invest_d_s', 'DamShp_tot'],
                             'multihaz_multisec': ['DamAgr_f_tot', 'revenue_agr',  'invest_f_a',
                                                   # 'DamAgr_d_tot',
                                                   'invest_d_a', 'DamUrb_tot',
                                                   'invest_f_u', 'DamShp_tot', 'invest_d_s',
                                                   'pathways_list_f_a', 'pathways_list_d_a', 'pathways_list_f_u',
                                                   'pathways_list_d_s'],
                             'multihaz_agr': ['DamAgr_f_tot',  'invest_f_a','revenue_agr', #'DamAgr_d_tot',
                                               'invest_d_a', 'pathways_list_f_a', 'pathways_list_d_a'],
                             'multhaz_urb': ['DamUrb_tot', 'invest_f_u', 'pathways_list_f_u']}

        cc_scenario_names = {'D': 'current', 'G': '2°C', 'Wp': '4°C'}

        number_pathways = {1: {'fa_p': 9,
                           'da_p': 11,
                           'fu_p': 11,
                           'ds_p': 10},
                           2: {'fa_p': 11,
                               'da_p': 11,
                               'fu_p': 11,
                               'ds_p': 10},
                           3: {'fa_p': 15,
                               'da_p': 11,
                               'fu_p': 15,
                               'ds_p': 10},
        }

        heatmap = create_subfolder(mainfolder=self.viz_folder, subfolder='heatmap_robustness')

        # Check rows where 'output' is 'revenue_agr' and subtract 'remaining_rev' from 'robustness'
        robustness_df.loc[robustness_df['output'] == 'revenue_agr', 'robustness'] = self.revenue_ref - robustness_df[
            'robustness']

        for sh_portfolios in portfolio_list[sector]:  # for each sh-pathway list, do the following
            df_sector = robustness_df[robustness_df['sector_hazard'] == sector]
            df_shportfolio = df_sector[df_sector[sh_portfolios] != 999]
            reordered_list = [sh_portfolios]
            rem_list = [sh for sh in portfolio_list[sector] if sh != sh_portfolios]
            reordered_list = np.append(reordered_list, rem_list)

            baseline_df = df_shportfolio[df_shportfolio[sh_portfolios] == 0][['cc_scenario', 'output', 'robustness']]
            baseline_df = baseline_df.rename(columns={'robustness': 'baseline_robustness'})

            # Step 2: Join the baseline values with the original DataFrame
            df = df_shportfolio.merge(baseline_df, on=['cc_scenario', 'output'], how='left')

            # Step 3: Subtract the baseline from robustness and store in a new column
            df['subtracted'] = df['robustness'] - df['baseline_robustness']
            # Set 'subtracted' to original 'robustness' for rows where sh_portfolios is 0
            df.loc[df[sh_portfolios] == 0, 'subtracted'] = df.loc[df[sh_portfolios] == 0, 'robustness']

            groupings = [obj for obj in sector_objectives[sh_portfolios] if obj.startswith('pathways')]
            mask = df['output'].isin(groupings)
            df.loc[mask, 'subtracted'] = (df['robustness'] - df['baseline_robustness']).abs()

            # Dropping the baseline_robustness column for clarity
            df = df.drop('baseline_robustness', axis=1)

            if len(reordered_list) == 4:
                reordered_dict = {0: reordered_list[:2],
                                  1: reordered_list[2:]}
            else:
                reordered_dict = {0: reordered_list[:2]}
            sh_list = []
            for group in range(len(reordered_dict)):
                fig = plt.figure(figsize=(20, 12))

                container_list = []
                for number_plots in range(np.minimum(len(portfolio_list[sector]), 2)):
                    # print(width_ratio_dict[group], len(sector_objectives[reordered_dict[group][number_plots]]))
                    print(reordered_dict[group][number_plots], number_pathways[self.stages[sector]][reordered_dict[group][0]])
                    all_objectives = gridspec.GridSpec(2, len(sector_objectives[reordered_dict[group][number_plots]]),height_ratios=[1,np.maximum(number_pathways[self.stages[sector]][reordered_dict[group][0]], number_pathways[self.stages[sector]][reordered_dict[group][-1]])])
                    container_list.append(all_objectives)
                # print(container_list)

                top_list = []
                bottom_list = []
                for cix, c in enumerate(container_list):
                    for i in range(len(sector_objectives[reordered_dict[group][cix]])):
                        top = fig.add_subplot(c[0, i])   # baseline
                        bottom = fig.add_subplot(c[1, i])   # real heatmap

                        # Move x-axis to the top
                        top.xaxis.set_ticks_position('top')
                        top.xaxis.set_label_position('top')

                        bottom.xaxis.set_ticks_position('top')
                        bottom.set_xticks([])

                        top_list.append(top)
                        bottom_list.append(bottom)

                if len(reordered_dict[0]) > 1:
                    if len(sector_objectives[reordered_dict[group][0]]) > len(
                            sector_objectives[reordered_dict[group][1]]):
                        positions = [0, 0.58]
                        width = [0.54, 0.36]
                    elif len(sector_objectives[reordered_dict[group][0]]) < len(
                            sector_objectives[reordered_dict[group][1]]):
                        positions = [0, .40]
                        width = [0.36, .54]
                        print(positions)
                    else:
                        positions = [0, 0.49]
                        width = [0.45, 0.45]
                else:
                    positions = [0]
                if self.stages[sector] == 3:
                    hspace_new = 0.04
                else:
                    hspace_new = 0.04
                for number_plots in range(np.minimum(len(portfolio_list[sector]), 2)):
                    if np.minimum(len(portfolio_list[sector]), 2) == 1:
                        container_list[number_plots].update(left=positions[number_plots], right=.94)
                        container_list[number_plots].update(wspace=0.07, hspace=hspace_new)
                    else:
                        container_list[number_plots].update(left=positions[number_plots],
                                                            right=positions[number_plots] + width[number_plots])
                        container_list[number_plots].update(wspace=0.07, hspace=hspace_new)

                cbar_ax = fig.add_axes([.95, .11, .03, .6])
                ax_tick = 0

                df_test = df.drop(['r10', 'r90'], axis=1)

                for sh_pair in reordered_dict[group]:
                    sh_list = np.append(sh_list, sh_pair)
                    keys_list_full = sector_objectives[sh_pair]

                    # normalize same unit objectives
                    # List of output groups
                    df_test['normalized'] = 1
                    # print(df_test)

                    output_groups = [[obj for obj in sector_objectives[sh_pair] if not obj.startswith('pathways')],
                                     [obj for obj in sector_objectives[sh_pair] if obj.startswith('pathways')]]

                    if sh_pair == sh_portfolios:
                        # Normalize for each group
                        for groupings in output_groups:
                            mask = df_test['output'].isin(groupings)
                            # remove_baseline = df_test[df_test[sh_pair] != 0]
                            mask_temp = df_test['output'].isin(groupings)
                            max_value = np.maximum(df_test.loc[mask_temp, 'subtracted'].max(),
                                                   np.abs(df_test.loc[mask_temp, 'subtracted'].min()))

                            if len(groupings) > 1:

                                # df_test.loc[mask, 'normalized'] = (df_test.loc[mask, 'subtracted'] - df_test.loc[
                                #     mask, 'subtracted'].min()) / (
                                #                                           df_test.loc[mask, 'subtracted'].max() -
                                #                                           df_test.loc[mask, 'subtracted'].min())
                                df_test.loc[mask, 'normalized'] = (df_test.loc[mask, 'subtracted']) / max_value

                            else:
                                df_test.loc[mask, 'normalized'] = - (df_test.loc[mask, 'subtracted']) / max_value

                    else:
                        for groupings in output_groups:
                            mask = df_test['output'].isin(groupings)
                            # remove_baseline = df_test[df_test[sh_pair] != 0]
                            mask_temp = df_test['output'].isin(groupings)
                            max_value = np.maximum(df_test.loc[mask_temp, 'subtracted'].max(),
                                                   np.abs(df_test.loc[mask_temp, 'subtracted'].min()))

                            if len(groupings) > 1:

                                # df_test.loc[mask, 'normalized'] = (df_test.loc[mask, 'subtracted'] - df_test.loc[
                                #     mask, 'subtracted'].min()) / (
                                #                                           df_test.loc[mask, 'subtracted'].max() -
                                #                                           df_test.loc[mask, 'subtracted'].min())
                                df_test.loc[mask, 'normalized'] = (df_test.loc[mask, 'subtracted']) / max_value

                            else:
                                df_test.loc[mask, 'normalized'] = - (df_test.loc[mask, 'subtracted']) / max_value


                    #
                    #     # Find baseline values for each cc_scenario
                    #     baseline_values = df_test[df_test[sh_portfolios] == 0].groupby(['cc_scenario', 'output'])[
                    #         'robustness'].first()
                    #
                    #     # Normalize around the baseline
                    #     for scenario, baseline in baseline_values.iteritems():
                    #         mask = (df_test['cc_scenario'] == scenario[0]) & (df_test['output'] == scenario[1])
                    #         df_test.loc[mask, 'normalized'] = - (df_test.loc[mask, 'robustness'] - baseline) / (
                    #                 1.3 * baseline - 0.7 * baseline)
                    #
                    #         df_test.loc[mask, 'normalized'] = (df_test.loc[mask, 'subtracted']) / (
                    #             np.maximum(df_test.loc[mask, 'subtracted'].max(),
                    #                        np.abs(df_test.loc[mask, 'subtracted'].min())))
                    #
                    # # df_out = df_test[df_test[sh_portfolios] != 0]
                    # df_out = df_test.copy()
                    #
                    #
                    # if sh_pair == sh_portfolios:
                    #     # calulate max values
                    #     remove_baseline = df_test[df_test[sh_pair] != 0]
                    #     max_value = np.maximum(remove_baseline.loc[:, 'subtracted'].max(), np.abs(remove_baseline.loc[:, 'subtracted'].min()))
                    #     # Normalize for each group
                    #     df_test.loc[:, 'normalized'] = (df_test.loc[:, 'subtracted']) / max_value
                    #     print(max_value)
                    # else:   # Find baseline values for each cc_scenario
                    #     # calulate max values
                    #     remove_baseline = df_test[df_test[sh_portfolios] != 0]
                    #     max_value = np.maximum(remove_baseline.loc[:, 'subtracted'].max(),
                    #                            np.abs(remove_baseline.loc[:, 'subtracted'].min()))
                    #     # Normalize for each group
                    #     df_test.loc[:, 'normalized'] = (df_test.loc[:, 'subtracted']) / max_value
                    #     print(max_value)

                        # baseline_values = df_test[df_test[sh_portfolios] == 0].groupby(['cc_scenario', 'output'])[
                        #     'robustness'].first()
                        #
                        # # Normalize around the baseline
                        # for scenario, baseline in baseline_values.iteritems():
                        #     mask = (df_test['cc_scenario'] == scenario[0]) & (df_test['output'] == scenario[1])
                        #     df_test.loc[mask, 'normalized'] = - (df_test.loc[mask, 'robustness'] - baseline) / (
                        #             1.3 * baseline - 0.7 * baseline)
                        #
                        #     df_test.loc[mask, 'normalized'] = (df_test.loc[mask, 'subtracted'])/ (np.maximum(df_test.loc[mask, 'subtracted'].max(),
                        #                 np.abs(df_test.loc[mask, 'subtracted'].min())))

                    df_out = df_test.copy()

                    for key in keys_list_full:
                        if key.startswith('pathway') and sh_pair == sh_portfolios:
                            my_cmap = plt.cm.colors.ListedColormap(['grey', 'grey'])
                        elif sh_pair == sh_portfolios:
                            my_cmap = 'RdYlGn_r'
                        else:
                            my_cmap = 'RdYlGn_r'

                        # Create an all-white colormap
                        all_white_cmap = plt.cm.colors.ListedColormap(['grey', 'grey'])
                        df_key = df_out[df_out['output'] == key]

                        # create pivot tables from long lists for current stage data (colored plots)
                        other_stage_mean = df_key.pivot(columns=sh_portfolios, index='cc_scenario',
                                                        values=['robustness', 'normalized', 'subtracted']).reset_index()
                        # Drop the unnecessary "cc_scenario" column
                        other_stage_mean.set_index('cc_scenario', inplace=True)
                        other_stage_mean.index.name = None
                        other_stage_mean.columns.name = None

                        other_stage_mean = other_stage_mean.reindex(sorted(other_stage_mean.columns), axis=1)
                        other_stage_mean = other_stage_mean.fillna(0)
                        other_stage_mean = other_stage_mean.rename(index=cc_scenario_names)

                        if not key.startswith('pathway') and not key.startswith('main'):
                            other_stage_mean['subtracted'] = other_stage_mean['subtracted'].astype(int)
                            my_fmt = "d"
                        else:
                            other_stage_mean['subtracted'] = other_stage_mean['subtracted'].round(1)
                            my_fmt = ".1f"

                        top_plot_n = other_stage_mean['normalized'].loc[:,[0]]
                        top_plot_s = other_stage_mean['subtracted'].loc[:,[0]]

                        bottom_plot_n = other_stage_mean['normalized'].drop(0, axis=1)
                        bottom_plot_s = other_stage_mean['subtracted'].drop(0, axis=1)

                        # Convert column names to integers
                        bottom_plot_n.columns = bottom_plot_n.columns.astype(int)
                        sns.heatmap(top_plot_n.T,
                                    annot=top_plot_s.T, fmt=my_fmt,
                                    linewidth=.5, ax=top_list[ax_tick], cmap=all_white_cmap, vmin=-1, vmax=1,
                                    cbar=False, annot_kws={
                                "size": 48 / np.sqrt(number_pathways[self.stages[sector]][sh_pair]-2)}, )
                        sns.heatmap(bottom_plot_n.astype(np.float32).T,
                                    annot=bottom_plot_s.T, fmt=my_fmt,
                                    linewidth=.5, ax=bottom_list[ax_tick], cmap=my_cmap, vmin=-1, vmax=1,
                                    cbar=ax_tick == 1, cbar_ax=cbar_ax if ax_tick else None, annot_kws={
                                "size": 48 / np.sqrt(len(bottom_plot_n.astype(np.float32).T))}, )

                        xticklabels = top_list[ax_tick].get_xticklabels()
                        bottom_list[ax_tick].set_xticks([])
                        for label in xticklabels:
                            label.set_fontsize(18)

                        # rotate & update the ytick labels on the second subplot
                        if ax_tick == 1:
                            # Update color bar
                            cbar = bottom_list[ax_tick].collections[0].colorbar
                            cbar.ax.invert_yaxis()
                            cbar.set_ticks([1, -1])
                            cbar.set_ticklabels(['Lowest','Highest'])
                            cbar.ax.tick_params(labelsize=18)
                            cbar.ax.yaxis.set_label_position('right')
                            cbar.ax.yaxis.label.set_horizontalalignment('center')

                        if ax_tick == 0:
                            # get the used pathway alterantives
                            pathway_df = pd.read_csv(
                                f'{pathways_path}/stage{self.stages[sector]}_portfolios_{ylabel_list[sh_portfolios]}.txt',
                                header=None)
                            bottom_list[ax_tick].set_ylabel(f'')
                            top_list[ax_tick].set_ylabel(f'')

                            for tick in bottom_list[ax_tick].get_yticklabels():
                                tick.set_rotation(0)
                            for tick in top_list[ax_tick].get_yticklabels():
                                tick.set_rotation(0)

                        else:
                            bottom_list[ax_tick].tick_params(axis='y', which='both', labelleft=False)
                            bottom_list[ax_tick].set_ylabel('')

                            top_list[ax_tick].tick_params(axis='y', which='both', labelleft=False)
                            top_list[ax_tick].set_ylabel('')

                        # Set key title per subplot
                        bottom_list[ax_tick].set_title(f'', y=1.12, fontsize=20)
                        top_list[ax_tick].set_title(f'{replacement_dict[key]}', y=1.5, fontsize=20)

                        # add icons to yticks
                        if ax_tick == 0:

                            # top plot
                            y_labels = top_list[ax_tick].get_yticklabels()
                            y_labels[0] = 'Baseline cost and damages\n(no measures implemented)'
                            y_labels[0] = 'Baseline cost and damages'
                            top_list[ax_tick].set_yticklabels(y_labels, fontsize=20)

                            # Get the y-tick positions
                            y_ticks = bottom_list[ax_tick].get_yticks()
                            y_labels = bottom_list[ax_tick].get_yticklabels()
                            bottom_list[ax_tick].set_yticklabels(y_labels, fontsize=20)

                            # add icons to y-axis
                            for idx, (y_tick, label) in enumerate(zip(y_ticks, y_labels)):
                                # Get all values from the row with index 1
                                row_values = pathway_df.loc[int(idx) + 1].tolist()

                                if sector == 'drought_shp':
                                    # Filter out nan from the list
                                    row_values = [x for x in row_values if x == x]
                                if self.stages[sector] == 1:
                                    if len(sector_objectives[sh_pair]) == 4:
                                        dist = -.73
                                        next_button_location = 0.17
                                    else:
                                        dist = -.53
                                        next_button_location = 0.12
                                elif self.stages[sector] == 2:
                                    if sh_pair == 'fa_p' :
                                        dist = -1.3
                                        next_button_location = 0.3
                                    elif sh_pair == 'da_p':
                                        dist = -1.4
                                        next_button_location = 0.3
                                    else:
                                        dist = -.55
                                        next_button_location = 0.12
                                else:
                                    if len(sector_objectives[reordered_dict[group][0]]) + len(sector_objectives[reordered_dict[group][1]]) == 6:
                                        dist = -1.2
                                        next_button_location = 0.25
                                    elif sh_pair == 'da_p':
                                        dist = -1.35
                                        next_button_location = 0.3
                                    elif sh_pair == 'ds_p':
                                        dist = -1.35
                                        next_button_location = 0.3
                                    elif sh_pair == 'fa_p':
                                        dist = -1.35
                                        next_button_location = 0.3
                                    elif sh_pair == 'fu_p':
                                        dist = -1.4
                                        next_button_location = 0.3

                                for measure in row_values:
                                    if measure not in ['no_measure', np.nan, 'nan', np.NaN, '']:
                                        # Get the logo filename for the current column from the dictionary
                                        img_path = measure_logos[measure]
                                        arr_img = plt.imread(img_path)

                                        imagebox = OffsetImage(arr_img, zoom=0.05)
                                        imagebox.image.axes = bottom_list[ax_tick]

                                        ab = AnnotationBbox(imagebox, (dist, y_tick),
                                                            xybox=(0, 0),  # Offset to place the logos slightly above
                                                            xycoords=("axes fraction", "data"),
                                                            boxcoords="offset points",
                                                            box_alignment=(0, .5),  # Align logos to the bottom
                                                            bboxprops={"edgecolor": "none"}, frameon=False)

                                        bottom_list[ax_tick].add_artist(ab)
                                        dist += next_button_location
                            # position_y = number_pathways[self.stages[sector]][reordered_dict[group][number_plots]]

                            if self.stages[sector] == 1:

                                if sh_pair == 'fa_p':
                                    position_y = 0.78
                                    x_position = -0.09
                                else:
                                    position_y = 0.80
                                    x_position = -0.1
                            elif self.stages[sector] == 2:
                                if sh_pair == 'fa_p':
                                    position_y = 0.805
                                    x_position = -0.09
                                else:
                                    position_y = 0.805
                                    x_position = -0.1
                            else:
                                if sh_pair == 'fu_p' and reordered_dict[group][0] == 'da_p':
                                    x_position = -0.1
                                    position_y = 0.81
                                elif sh_pair == 'fa_p':
                                    x_position = -0.1
                                    position_y = 0.82
                                elif sh_pair == 'fu_p':
                                    x_position = -0.1
                                    position_y = 0.82
                                else:
                                    x_position = -0.1
                                    position_y = 0.81
                            fig.text(x_position, position_y, #(position_y - 1) / position_y - 0.1,
                                     f'{title_list[reordered_list[0]]} pathways ', ha='center', fontsize=22,
                                     weight='bold')

                        ax_tick += 1
                # add a legend
                # Create a bounding box (legend box)
                rect = Rectangle((0, 0.1), 0.94, -.16, transform=fig.transFigure, facecolor='none', ec='black',
                                 zorder=5)
                fig.patches.extend([rect])
                fig.text(0.13, .07, f'Legend: Pathways measures', ha='center', fontsize=22, weight='bold')

                legend_ax_dim = [0.02, 0.06, 0.95, 0.3]  # [left, bottom, width, height] in figure coordinates
                legend_ax = fig.add_axes(legend_ax_dim, frame_on=False)  # Adjust the position and size as needed
                legend_ax.axis('off')  # Turn off the axis

                # Cbar title
                fig.text(0.95, 0.74, 'Performance\nRobustness', size=22)

                # Manually position logos and text within the box
                unique_values = pd.unique(pathway_df.values.ravel())
                unique_values = [x for x in unique_values if x == x]

                positions = [(0, -.1), (0.23, -.1), (0.46, -.1), (0.69, -.1),
                             (0, -0.3), (0.23, -0.3), (0.46, -0.3), (0.69, -0.3),
                             (0, -0.5), (0.23, -0.5), (0.46, -0.5), (0.69, -0.5)]
                for idx, name in enumerate(unique_values):
                    if name not in ['no_measure', np.nan, 'nan', np.NaN, '']:
                        img_path = measure_logos[name]
                        label_text = f'{method_dict[name]}'
                        add_logo_and_text(legend_ax, img_path, positions[idx - 1][0], positions[idx - 1][1], label_text)

                # Create a single Axes object for each gridspec to set the title
                if len(portfolio_list[sector]) == 1:
                    positions = [0.5]
                else:
                    positions = [0.3, 0.73]

                for x in range(len(positions)):
                    fig.text(positions[x], 0.97, f'{title_list[reordered_dict[group][x]]} objectives', ha='center',
                             fontsize=22, weight='bold')

                # Save fig
                if len(portfolio_list[sector]) >= 2:
                    fig.savefig(
                        f'{heatmap}/heatmap_statistics_{sector}_{sh_portfolios}_stage_{self.stages[sector]}_{reordered_dict[group][0]}_{reordered_dict[group][1]}.png',
                        dpi=300, bbox_inches='tight')
                else:
                    fig.savefig(
                        f'{heatmap}/heatmap_statistics_{sector}_{sh_portfolios}_stage_{self.stages[sector]}_{reordered_dict[group][0]}.png',
                        dpi=300, bbox_inches='tight')

                plt.clf()
                plt.close()


    def plot_measure_interaction_effects(self,interaction_stages):
        heatmap = create_subfolder(mainfolder=self.viz_folder, subfolder='heatmap_interaction')

        search_keys = {'fa_p': ['DamAgr_f_tot'],
                       'da_p': ['revenue_agr'],
                       'fu_p': ['DamUrb_tot'],
                       'ds_p': ['DamShp_tot']}
        # search_keys = {'fa_p': ['revenue_agr'],
        #                'da_p': ['revenue_agr'],
        #                'fu_p': ['DamUrb_tot'],
        #                'ds_p': ['DamShp_tot']}
        sh_pairs = {'fa_p': 'flood_agriculture',
                    'da_p': 'drought_agriculture',
                    'fu_p': 'flood_urban',
                    'ds_p': 'drought_shipping'}

        cc_scenario_names = {'D': 'current', 'G': '2°C', 'Wp': '4°C'}

        for interaction_stage in interaction_stages:
            # interaction_df = f'{self.process_folder}/{interaction_stage}_robustness_pw_pair.csv'
            interaction_df = f'{self.process_folder}/all_robustness_mean_interaction_per_pathway_{interaction_stage}.csv'
            robustness_df = pd.read_csv(interaction_df)

            pairs = {'multihaz_agr': ['fa_p', 'da_p'],
                     'multihaz_multisec': ['fa_p', 'da_p', 'fu_p', 'ds_p']}
            pathways_list = pairs[interaction_stage]
            start_tick = 1

            for pair1 in pairs[interaction_stage]:

                # pathways_list.remove(pair1)  # avoid duplication of the process
                for no_interactions in range(len(pathways_list)-start_tick):
                    # pair1 = pairs[interaction_stage][0]
                    pair2 = pathways_list[no_interactions+start_tick]
                    for i in range(len(search_keys[pair1])):
                        searchkeys = [search_keys[pair1][i], search_keys[pair2][i]]


                        df_pivots = {}

                        df_relevant = robustness_df[(robustness_df[pair1] != 999) & (robustness_df[pair2] != 999)]

                        # Set up the figure and subplots
                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                        fig.subplots_adjust(wspace=0.05)
                        ax_tick = 0
                        for cc_scenario in self.cc_scenarios:
                            # print(cc_scenario)
                            search_tick = 0
                            df_cc_scenario = df_relevant[df_relevant.cc_scenario == cc_scenario]
                            for key in searchkeys:
                                df_key = df_cc_scenario[df_cc_scenario.output == key]

                                df_pivot = df_key.pivot(index=pair1, columns=pair2, values='robustness')
                                # print(df_pivot)

                                if search_tick == 0:
                                    if self.optimization[search_keys[pair1][i]] == 'min':
                                        df_pivots[pair1] = 1/df_pivot.div(df_pivot.iloc[:, 0], axis=0)
                                    elif self.optimization[search_keys[pair1][i]] == 'max':
                                        df_pivots[pair1] = df_pivot.div(df_pivot.iloc[:, 0], axis=0)

                                else:
                                    if self.optimization[search_keys[pair2][i]] == 'min':
                                        df_pivots[pair2] = 1/df_pivot.div(df_pivot.iloc[0,:], axis=1)
                                    elif self.optimization[search_keys[pair2][i]] == 'max':
                                        df_pivots[pair2] = df_pivot.div(df_pivot.iloc[0,:], axis=1)
                                search_tick += 1

                            arr1to2 = df_pivots[pair1].values
                            arr2to1 = df_pivots[pair2].values
                            arr1to2 = np.append(arr1to2, -np.ones((len(arr1to2[:, 1]), 1)), axis=1)
                            arr1to2 = np.append(arr1to2, -np.ones((1, len(arr1to2[1, :]))), axis=0)
                            arr2to1 = np.append(arr2to1, -np.ones((len(arr2to1[:, 1]), 1)), axis=1)
                            arr2to1 = np.append(arr2to1, -np.ones((1, len(arr2to1[1, :]))), axis=0)
                            M = len(arr1to2[0, :]) - 1
                            N = len(arr1to2[:, 1]) - 1
                            x = np.arange(M + 1)
                            y = np.arange(N + 1)
                            xs, ys = np.meshgrid(x, y)

                            zh1 = arr1to2
                            zh2 = arr2to1

                            zh1 = zh1[:-1, :-1].ravel()
                            zh2 = zh2[:-1, :-1].ravel()
                            # max_val = np.maximum(np.amax(zh1), np.amax(zh2)) * 1.1
                            # min_val = np.minimum(np.amin(zh1), np.amin(zh2)) * 0.9

                            triangles1 = [(i + j * (M + 1), i + 1 + j * (M + 1), i + (j + 1) * (M + 1) + 1) for j in
                                          range(N) for i in range(M)]
                            triangles2 = [(i + 1 + j * (M + 1) - 1, i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1)) for j
                                          in range(N) for i in range(M)]

                            triang1 = Triangulation(xs.ravel(), ys.ravel(), triangles1)
                            triang2 = Triangulation(xs.ravel(), ys.ravel(), triangles2)

                            my_cmap = plt.get_cmap('RdYlGn')
                            # my_cmap = ListedColormap(sns.color_palette('RdYlGn', 5))

                            # define the string labels for x and y axis
                            xlabels = df_pivots[pair1].columns.tolist()
                            ylabels = df_pivots[pair1].index.tolist()

                            # set tick locations to be the middle point between each pair of adjacent x or y values
                            xticks = np.insert(np.diff(x) / 2 + x[:-1], 0, x[0] - 0.5)
                            yticks = np.insert(np.diff(y) / 2 + y[:-1], 0, y[0] - 0.5)

                            # plot heatmap with centered axis labels and string labels
                            ax = axs[ax_tick]
                            # Adding a specifier for each subplot
                            ax.set_title(f"{cc_scenario_names[cc_scenario]} scenario")

                            # bottom triangles for arr1to2 (for stage 2: combined of drought_agr & flood_agr effects relative to flood_agr)
                            img1 = ax.tripcolor(triang1, zh1, cmap=my_cmap, vmax=1.3, vmin=0.7, edgecolors='w',
                                                 linewidths=1.2)
                            img2 = ax.tripcolor(triang2, zh2, cmap=my_cmap, vmax=1.3, vmin=0.7, edgecolors='w',
                                                 linewidths=1.2)

                            # plt.tick_params(axis='x', which='both', labelbottom=False, labeltop=True)
                            ax.set_xticks(xticks[1:], xlabels)

                            ax.set_xlim(x[0], x[-1])
                            ax.set_ylim(y[0], y[-1])

                            # Show ylabels only for the utmost left figure
                            if ax_tick > 0:
                                ax.set_yticks(yticks[1:], ylabels)
                                ax.set_yticklabels([])
                            else:
                                ax.set_yticks(yticks[1:], ylabels)
                                ax.set_ylabel(f'potential pathway for {sh_pairs[pair1]}')
                            if ax_tick == 1:
                                ax.set_xlabel(f'potential pathway for {sh_pairs[pair2]}')

                            ax_tick += 1

                        # Adjust the third subplot to make space for the colorbar
                        fig.subplots_adjust(right=0.94)
                        cbar_ax = fig.add_axes([0.96, 0.15, 0.03, 0.7])

                        # Show colorbar only for the third subplot
                        plt.colorbar(img1, cax=cbar_ax,  label='Interaction effect factor')

                        fig.savefig(
                            f'{heatmap}/heatmap_interactions_{interaction_stage}_{search_keys[pair1][i]}_{search_keys[pair2][i]}.png',
                            dpi=300,
                            bbox_inches='tight')
                        print('figure saved')

                start_tick += 1


    def create_timingcomparison_plots(self, df_timings, difference_groups ):
        color_list = [
                (255 / 255, 215 / 255, 0 / 255),
                (250 / 255, 135 / 255, 117 / 255),
                (205 / 255, 52 / 255, 181 / 255),
                (157 / 255, 2 / 255, 215 / 255),
                (0 / 255, 0 / 255, 255 / 255)
            ]
        no_interaction = difference_groups['no_interaction']
        with_interaction = difference_groups['with_interaction']

        # Create a ListedColormap from the color list
        my_cmap = ListedColormap(color_list)

        difference_plot = {}

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.08)

        timing_plots = create_subfolder(self.viz_folder, 'sequencing_measures')

        df_sector = df_timings[df_timings.sectorname == no_interaction[0]]

        df_clean = df_sector[df_sector['timing'].notna()]

        df_pivot = df_clean.pivot(index=[no_interaction[0],'portfolio', 'cc_scenario', 'climvar','realization'], columns='no_measure', values='timing').reset_index()

        # sort based on implemented measure columns
        not_columns = ['no_measure', no_interaction[0], 'portfolio', 'cc_scenario', 'climvar', 'realization']
        relevant_columns = [column_name for column_name in df_pivot.columns if column_name not in not_columns]
        sorted_df = df_pivot.groupby([no_interaction[0]]).apply(lambda x: x.sort_values(relevant_columns, ascending=True)).reset_index(
            drop=True)

        # make sure that all have the same number of columns
        for additional_columns in range(len(relevant_columns),6):
            sorted_df[additional_columns] = 100
        sorted_df = sorted_df.replace(np.NaN, 100)

        sorted_df['ylabel'] = no_interaction[0] + ' | ' + sorted_df['cc_scenario'] + ' | ' + sorted_df[
            'climvar'].astype(str)

        df_new = sorted_df.copy()
        df_original = sorted_df.copy()
        for x in range(2,6):
            df_new.loc[:, x] = sorted_df.loc[:, x] - sorted_df.loc[:, x-1]
        # per portfolio
        ax_tick = 0
        for portfol in [no_interaction[1], with_interaction[1]]:
            portfol_df = df_new[df_new.portfolio == portfol]
            portfol_df_original = df_original[df_original.portfolio == portfol]

            portfol_df.set_index('ylabel').loc[:, [1,2,3,4,5]].plot(kind='barh',ax=axs[ax_tick], stacked=True, width=0.95, color=my_cmap.colors,legend=False)
            axs[ax_tick].set_yticklabels(['' for x in portfol_df.ylabel])
            axs[ax_tick].set_xlabel("Years")
            axs[ax_tick].xaxis.grid()
            if ax_tick == 0:
                axs[ax_tick].set_ylabel(f'Model Realizations (n={len(portfol_df.ylabel)})')
            else:
                axs[ax_tick].set_ylabel(f'')
            axs[ax_tick].set_title(f'{no_interaction[0]}: {portfol}')
            axs[ax_tick].set_xlim([0,100])
            axs[ax_tick].set_xticks(np.arange(0, 101, 10))

            # store for difference_groups
            if portfol == difference_groups['no_interaction'][1]:
                difference_plot['no_interaction'] = portfol_df_original.loc[:,
                                                     [1, 2, 3, 4, 5]]
                print(portfol_df_original)
            if portfol == difference_groups['with_interaction'][1]:
                difference_plot['with_interaction'] = portfol_df_original.loc[:,
                                                    [1, 2, 3, 4, 5]]
                print(portfol_df_original)
            ax_tick += 1

        # create_comparison plot:

        differences = difference_plot['with_interaction'].values - difference_plot['no_interaction'].values
        differences_df = pd.DataFrame(columns=[1,2,3,4,5], data=differences)
        differences_df.plot(kind='barh', stacked=True, width=0.95, ax=axs[ax_tick], color=my_cmap.colors, legend=False)
        axs[ax_tick].set_yticklabels(['' for x in portfol_df.ylabel])
        axs[ax_tick].set_xlabel("Change implementation timing (with interaction)")
        axs[ax_tick].xaxis.grid()
        # axs[ax_tick].legend(['current condition', '1st measure', '2nd measure', '3rd measure', '4th measure'],
        #             loc='upper right', fontsize=16)
        axs[ax_tick].set_title(f'Shift timing')
        axs[ax_tick].set_xlim([-50, 50])
        axs[ax_tick].set_xticks(np.arange(-50, 51, 10))
        # plt.tight_layout()

        # Adjust figure to make space for the legend
        fig.subplots_adjust(bottom=0.2)

        # Add a figure-level legend
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, ['current condition', '1st measure', '2nd measure', '3rd measure', '4th measure'], loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0))

        fig.savefig(f'{timing_plots}/stacked_bar_00_comparison_{no_interaction[0]}_{difference_groups["with_interaction"][1]}.png', dpi=300, bbox_inches='tight')

