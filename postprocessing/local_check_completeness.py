import itertools
import os


def run_checker(mainfolder, realization, measure_realizations, sector):
    num_realizations_dict = {
        'flood_agr': {
            'fa_p': 9,
            'da_p': 0,
            'fu_p': 0,
            'ds_p': 0
        },
        'drought_agr': {
            'fa_p': 0,
            'da_p': 11,
            'fu_p': 0,
            'ds_p': 0
        },
        'flood_urb': {
            'fa_p': 0,
            'da_p': 0,
            'fu_p': 11,
            'ds_p': 0
        },
        'drought_shp': {
            'fa_p': 0,
            'da_p': 0,
            'fu_p': 0,
            'ds_p': 10
        },
        'multihaz_agr': {
            'fa_p': 12,
            'da_p': 11,
            'fu_p': 0,
            'ds_p': 0
        },
        'multihaz_urb': {
            'fa_p': 0,
            'da_p': 0,
            'fu_p': 15,
            'ds_p': 0
        },
        'multihaz_multisec': {
            'fa_p': 15,
            'da_p': 11,
            'fu_p': 15,
            'ds_p': 10
        }
    }

    num_realizations = num_realizations_dict[sector]

    fa_p = range(num_realizations['fa_p'])
    fu_p = range(num_realizations['fu_p'])
    da_p = range(num_realizations['da_p'])
    ds_p = range(num_realizations['ds_p'])

    combinations = list(itertools.product(fa_p, da_p, fu_p, ds_p))
    result = check_all_outputs(mainfolder, realization, measure_realizations, sector)
    computed_combinations = [tuple(map(int, x.split('_'))) for x in result]
    missing_combinations = [x for x in combinations if x not in computed_combinations]
    print("missing: ", missing_combinations)

    with open(f'{mainfolder}/{str(realization).zfill(6)}/missing_combinations_{sector}.txt', "w") as file:
        if missing_combinations == []:
            file.write('no_combs_missing')
        else:
            file.write('\n'.join('%s,%s,%s,%s' % x for x in missing_combinations))


def check_all_outputs(mainfolder, realization, measure_realizations, sector):
    '''combines all outputs into one big dataframe of long-version'''
    stages = {'flood_agr': 1,
              'drought_agr': 1,
              'flood_urb': 1,
              'drought_shp': 1,
              'multihaz_agr': 2,
              'multihaz_urb': 2,
              'multihaz_multisec': 3
              }
    letter_ranges = {
        'flood_agr': [49, 60],
        'drought_agr': [51, 62],
        'flood_urb': [49, 60],
        'drought_shp': [51, 62],
        'multihaz_agr': [52, 63],
        'multihaz_urb': [52, 63],
        'multihaz_multisec': [57, 68]
    }

    folder_paths = []
    for reali in measure_realizations:
        folder_paths.append(f'{mainfolder}/{str(realization + reali).zfill(6)}/stage_{stages[sector]}/{sector}/')

    file_list = []
    for folder_path in folder_paths:
        # identify all files in all subfolders of a certain realization (load all stages and sector output files)
        file_list.append(find_csvfiles_in_folder(folder_path=folder_path))

    flattened_list = [element[letter_ranges[sector][0]:letter_ranges[sector][1]] for sublist in file_list for element in
                      sublist]
    # unique_list = list(set(flattened_list))
    print("tested pathways (combinations):", flattened_list)
    return flattened_list


def find_csvfiles_in_folder(folder_path):
    file_paths = []
    # walk through the directory tree and add file paths to the list
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.gz'):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths
