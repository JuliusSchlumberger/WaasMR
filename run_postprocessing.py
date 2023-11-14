from waasmodel_v6.RunClass import WrapperRunModel
import numpy as np
from postprocessing.local_check_completeness import run_checker
from postprocessing.create_list_of_files import list_of_files
from postprocessing.local_outputs_combineoutputs_new import run_processing
from postprocessing.local_combine_robustnessvalues import run_combiner



realization = 5000
sectors_list = ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp', 'multihaz_agr', 'multihaz_urb',
                'multihaz_multisec']
sectors_list = ['flood_agr']
mainfolder = 'model_outputs'
start_val = 0
end_val = -1

for sector in sectors_list:
    # Check completeness
    run_checker(mainfolder=mainfolder, realization=realization,
                measure_realizations=[0], sector=sector)

    # create list of outputfiles
    list_of_files(mainfolder=mainfolder,realization=realization, measure_realizations=[0],
                   sector=sector)

    # create endvalues & robustness values
    run_processing(mainfolder='model_outputs', realization=realization, no_realizations=1,
                   sectors_list=sectors_list, start_val=start_val, end_val=end_val,
                   endvalues=True, robustvalues=True)

# combine all processed data
sectors_list = {1: ['flood_agr', 'drought_agr','flood_urb', 'drought_shp'],
                    2: ['multihaz_agr'],
                    3: ['multihaz_multisec']}
stages_list = [1]

run_combiner(realization=realization, sectors_list=sectors_list, stages_list=stages_list)
