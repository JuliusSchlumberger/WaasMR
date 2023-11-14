import os
import numpy as np
import pandas as pd


def load_files(outputfolder):
    return [f'{outputfolder}/{f}' for f in os.listdir(outputfolder) if f.startswith('robustness') and os.path.isfile(os.path.join(outputfolder, f))]

def get_average_robustness_sector_pathway(cc_scenarios, df, stage, sector):
    df_means = []

    portfolio_list = {'flood_agr': ['fa_p'],
                      'drought_agr': ['da_p'],
                      'flood_urb': ['fu_p'],
                      'drought_shp': ['ds_p'],
                      'multihaz_agr': ['fa_p', 'da_p'],
                      'multihaz_urb': ['fu_p'],
                      'multihaz_multisec': ['fa_p', 'da_p', 'fu_p', 'ds_p']}
    portfolios_names = ['fa_p', 'da_p', 'fu_p', 'ds_p']

    df[portfolios_names] = df.portfolio.str.split('_', expand=True)
    df[portfolios_names] = df[portfolios_names].astype(int)

    # for sector in df.sector_hazard.unique():
    df_sector = df[df['sector_hazard'] == sector]
    for cc_scenario in cc_scenarios:
        df_climate = df_sector[df_sector['cc_scenario'] == cc_scenario]
        keys = df_climate.output.unique()
        for sh_portfolios in portfolio_list[sector]:  # for each sh-pathway list, do the following
            rem_list = [x for x in portfolios_names if x not in sh_portfolios]
            print(sh_portfolios, rem_list)

            for pathway in df_climate[sh_portfolios].unique():
                temp = df_climate[df_climate[sh_portfolios] == pathway]

                for key in keys:
                    temp2 = temp[temp['output'] == key]
                    rob_val = temp2.robustness.mean()
                    r10_val = 0
                    r90_val = 0

                    if np.isnan(rob_val):
                        rob_val = 0
                        print(key, pathway, 'alarm')
                    if np.isnan(r10_val):
                        r10_val = 0
                    if np.isnan(r90_val):
                        r90_val = 0

                    df_means.append({
                        'robustness': rob_val,
                        'r10': r10_val,
                        'r90': r90_val,
                        'output': key,
                        'stage': stage,
                        'sector_hazard': sector,
                        'cc_scenario': cc_scenario,
                        sh_portfolios: pathway,
                        rem_list[0]: 999,
                        rem_list[1]: 999,
                        rem_list[2]: 999,
                    })
    df_means = pd.DataFrame(df_means)
    return df_means

def get_average_robustness_sector_interaction(cc_scenarios, df, stage, sector):
    df_means = []

    portfolio_list = {'flood_agr': ['fa_p'],
                      'drought_agr': ['da_p'],
                      'flood_urb': ['fu_p'],
                      'drought_shp': ['ds_p'],
                      'multihaz_agr': ['fa_p', 'da_p'],
                      'multihaz_urb': ['fu_p'],
                      'multihaz_multisec': ['fa_p', 'da_p', 'fu_p', 'ds_p']}
    portfolios_names = ['fa_p', 'da_p', 'fu_p', 'ds_p']

    df[portfolios_names] = df.portfolio.str.split('_', expand=True)
    df[portfolios_names] = df[portfolios_names].astype(int)

    # for sector in df.sector_hazard.unique():
    df_sector = df[df['sector_hazard'] == sector]
    for cc_scenario in cc_scenarios:
        df_climate = df_sector[df_sector['cc_scenario'] == cc_scenario]
        keys = df_climate.output.unique()
        interaction_list = portfolio_list[sector]
        for sh_portfolios in interaction_list:  # for each sh-pathway list, do the following
            interaction_list = [x for x in interaction_list if x not in sh_portfolios]
            rem_list = [x for x in portfolios_names if x not in sh_portfolios]
            print(sh_portfolios, rem_list)

            for pathway in df_climate[sh_portfolios].unique():
                for interacting_sector in interaction_list:
                    last_list = [x for x in rem_list if x not in interacting_sector]
                    for interacting_pathway in df_climate[interacting_sector].unique():
                        subset_mainpathway = df_climate[df_climate[sh_portfolios] == pathway]
                        subset_both_pathways = subset_mainpathway[subset_mainpathway[interacting_sector] == interacting_pathway]

                        for key in keys:
                            temp2 = subset_both_pathways[subset_both_pathways['output'] == key]
                            rob_val = temp2.robustness.mean()
                            r10_val = 0
                            r90_val = 0

                            if np.isnan(rob_val):
                                rob_val = 0
                                print(key, pathway, 'alarm')
                            if np.isnan(r10_val):
                                r10_val = 0
                            if np.isnan(r90_val):
                                r90_val = 0

                            df_means.append({
                                'robustness': rob_val,
                                'r10': r10_val,
                                'r90': r90_val,
                                'output': key,
                                'stage': stage,
                                'sector_hazard': sector,
                                'cc_scenario': cc_scenario,
                                sh_portfolios: pathway,
                                interacting_sector: interacting_pathway,
                                last_list[0]: 999,
                                last_list[1]: 999,
                            })
    df_means = pd.DataFrame(df_means)
    return df_means

def run_combiner(realization, sectors_list, stages_list):

    cc_scenarios = ['D', 'G', 'Wp']
    addrobustness = True
    combine_means_stages = []
    # combine all robustness values per stage
    for stage in stages_list:
        outputfolder = f'model_outputs/{str(realization).zfill(6)}/stage_{stage}'
        robustness_output = f'model_outputs/{str(realization).zfill(6)}/combined_robustness_stage_{stage}.csv'
        if addrobustness:
            files = load_files(outputfolder=outputfolder)
            print(files)
            robustness_df = pd.read_csv(files[0])
            # robustness_df.to_csv(robustness_output, index=False)
            for file in files[1:]:
                add_robustness = pd.read_csv(file)
                robustness_df = pd.concat([robustness_df,add_robustness], ignore_index=True)
            robustness_df.to_csv(robustness_output, index=False)
        all_robustness_df = pd.read_csv(robustness_output)
        # get average robustness values across various pathways combinations per sector-hazard pair per stage
        for sector in sectors_list[stage]:
            df_means = get_average_robustness_sector_pathway(cc_scenarios=cc_scenarios, df=all_robustness_df, stage=stage,
                                              sector=sector)
            # print(df_means)
            meanfolder = f'model_outputs/{str(realization).zfill(6)}/all_robustness_mean_per_pathway_{sector}.csv'
            combine_means_stages.append(meanfolder)
            df_means.to_csv(meanfolder)
            if sector in ['multihaz_agr', 'multihaz_multisec']:
                df_mean_interaction = get_average_robustness_sector_interaction(cc_scenarios=cc_scenarios, df=all_robustness_df, stage=stage,
                                                  sector=sector)
                interaction_meanfolder = f'model_outputs/{str(realization).zfill(6)}/all_robustness_mean_interaction_per_pathway_{sector}.csv'
                df_mean_interaction.to_csv(interaction_meanfolder)
            # combine the mean robustness values across all stages
    # combine all mean robustness values for all stages
    means_output = f'model_outputs/{str(realization).zfill(6)}/all_robustness_mean_per_pathway_all_stages.csv'
    robustness_df = pd.read_csv(combine_means_stages[0])
    robustness_df.to_csv(means_output, index=False)
    print(combine_means_stages)
    for stage in combine_means_stages[1:]:
        add_robustness = pd.read_csv(stage)
        robustness_df = pd.concat([robustness_df, add_robustness], ignore_index=True)
        robustness_df.to_csv(means_output, index=False)


realization = 5000
sectors_list = ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp', 'multihaz_agr', 'multihaz_urb',
                'multihaz_multisec']
sectors_list = ['multihaz_agr']



