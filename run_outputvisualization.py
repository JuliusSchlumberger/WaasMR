from viz.output_viz import OutputPocessing


realization = 5000
stages = [1,2,3]
outfolder = 'model_outputs'  # name of folder (within current folder) created to store the outputs
no_realizations = 1

sectors_list = {1: ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp'],
                2: ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp', 'multihaz_agr', 'multihaz_urb'],
                3: ['flood_agr', 'drought_agr', 'flood_urb', 'drought_shp', 'multihaz_agr', 'multihaz_urb',
                    'multihaz_multisec']}
sectors_list = {1: ['flood_urb'],
                2: ['multihaz_agr'],
                3: ['multihaz_multisec']}

sectors_list = sectors_list[max(stages)]
print(sectors_list, max(stages))

robustness_heatmap = True
interactionplot = False
convert_files_switch = False
timingplot = False

difference_groups = {'no_interaction': ['f_a', '06_00_00_00' ],
                             'with_interaction': ['f_a', '06_03_00_00' ]}

tester = OutputPocessing(stages=stages, no_realizations=no_realizations, realization=realization,
                         model_output_dir='model_outputs/',
                         outputprocess_folder=f'model_outputs/{str(realization).zfill(6)}',
                         sectors_list=sectors_list)
tester.run_processing(convert_files_switch=convert_files_switch,
                      robustness_heatmap=robustness_heatmap, interactionplot=interactionplot, timingplot=timingplot, difference_groups=difference_groups)