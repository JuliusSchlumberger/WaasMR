import numpy as np
import os


def list_of_files(sector, realization,mainfolder, measure_realizations):
    stages = {
        'flood_agr': 1,
        'drought_agr': 1,
        'flood_urb': 1,
        'drought_shp': 1,
        'multihaz_agr': 2,
        'multihaz_urb': 2,
        'multihaz_multisec': 3
    }

    folder_paths = []
    for reali in measure_realizations:
        folder_paths.append(f'{mainfolder}/{str(realization + reali).zfill(6)}/stage_{stages[sector]}/{sector}')
    print(folder_paths)
    file_list = []
    for folder_path in folder_paths:
        # identify all files in all subfolders of a certain realization (load all stages and sector output files)
        file_list.append(find_csvfiles_in_folder(folder_path=folder_path))
    print(len(file_list))
    flattened_list = [element for sublist in file_list for element in sublist]
    np.savetxt(f'{mainfolder}/{str(realization).zfill(6)}/all_files_{sector}.txt', flattened_list, fmt='%s', delimiter=',')


def find_csvfiles_in_folder(folder_path):
    file_paths = []
    # walk through the directory tree and add file paths to the list
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.gz'):
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths



