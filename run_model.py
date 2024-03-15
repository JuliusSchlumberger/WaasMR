from waasmodel_v6.RunClass import WrapperRunModel
import numpy as np

def specific_run(sector, stage, first_scenario, num_scenarios,analysis,test_all, outfolder, measure_list, portfolio_numbers):
    """To run a single combination of pathways combinations"""
    for i, tick in enumerate(zip(measure_list,realization_numbers)):
        runmodel = WrapperRunModel(hazard_sector_pair=sector, first_scenario=first_scenario,
                                           last_scenario=first_scenario + num_scenarios,
                                           first_timestep=0, last_timestep=3600, stage=stage,
                                           portfolio_numbers=portfolio_numbers,
                                           realization=tick[1],
                                           analysis=analysis,
                                           test_all=test_all)
        runmodel.wrapper_runner(outfolder=outfolder, measures=tick[0])


def normal_run(realization_ini, sector, stage, pathway_combinations, start_value, end_value):
    """To run the Waas-MR model in stress-test mode."""
    pathway_combinations_relevant = pathway_combinations[start_value:end_value]
    sectors = ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp', 'multihaz_agr', 'multihaz_urb',
               'multihaz_multisec']
    sector = sectors[sector]

    for combination in pathway_combinations_relevant:
        portfolio_numbers_eff = {
            'flood_agr': combination[0],
            'drought_agr': combination[1],
            'flood_urb': combination[2],
            'drought_shp': combination[3]
        }

        runmodel = WrapperRunModel(hazard_sector_pair=sector, first_scenario=1,
                                   last_scenario=31,
                                   first_timestep=0, last_timestep=3600, stage=stage,
                                   portfolio_numbers=portfolio_numbers_eff, realization=realization_ini,
                                   analysis='out_only', test_all=False)
        runmodel.wrapper_runner(outfolder='model_outputs', measures=['no_measure'])


# SPECIFIC RUN #
sector = 'multihaz_agr' # options are: 'flood_agr', 'drought_agr', 'flood_urb', 'drought_shp', 'multihaz_agr', 'multihaz_urb', 'multihaz_multisec'
first_scenario = 1  # scenarios from 1 to 31
stage = 2
num_scenarios = 30
analysis = 'analysis'  # 'out_only', 'analysis' to specify which outputfiles are generated (see inputs_runclass.py)
test_all = False  # if False, only discharge levels not leading to damages will be neglected (speed up computation)
outfolder = 'inputs_visual'  # name of folder (within current folder) created to store the outputs
measure_list = [['no_measure']]
realization_numbers = [9999]
portfolio_numbers = {
                'flood_agr': 0,
                'flood_urb': 0,
                'drought_agr': 0,
                'drought_shp': 0}

specific_run(sector=sector, stage=stage, first_scenario=first_scenario, num_scenarios=num_scenarios,analysis=analysis,
             test_all=test_all, outfolder=outfolder, measure_list=measure_list, portfolio_numbers=portfolio_numbers)


# NORMAL RUN #
realization = 7000
sector = 0  # 0: flood_agr, 1: drought_agr, 2: flood_urb, 3: drought_shp, 4: multihaz_agr, 5: multihaz_urb, 6: multihaz_multisec
stage= 1    # 1: single sector, single hazard, 2: multihazard, 3: multi-risk


fname_combos = f'PotentialPathways/0_allcombinations_stage{stage}_sector_{sector}_realization_{realization}.txt'
pathway_combinations = np.loadtxt(fname_combos, delimiter=',', dtype=int)

start_value = 6 # which potential pathways combinations to test.
end_value = len(pathway_combinations)

#
# normal_run(realization_ini=realization, sector=sector,stage=stage,
#            pathway_combinations=pathway_combinations, start_value=start_value, end_value=end_value)