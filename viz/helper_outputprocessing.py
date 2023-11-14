import numpy as np
import os
import pandas as pd
import itertools
import gzip

def read_csv_file(file):
    # read the compressed file back into a DataFrame
    with gzip.open(file, 'rb') as f:
        df = pd.read_csv(f)
    return df

def convert_files(folder):
    filenames = ['endvalues.csv', 'timing.csv', 'all_outputs.csv']
    # compress the text-based file with gzip
    for filename in filenames:
        df = read_csv_file(file=f'{folder}/{filename}.gz')
        df.to_csv(f'{folder}/{filename}')

def create_subfolder(mainfolder, subfolder):
    subfolder = os.path.join(mainfolder, subfolder)
    try:
        os.mkdir(subfolder)
    except OSError as error:
        pass
    return subfolder

def call_function(func_name, *args, **kwargs):
    output = globals()[func_name](*args, **kwargs)
    return output


def createAllCombinations(no_portfolio_f_a, no_portfolio_d_a, no_portfolio_f_u, no_portfolio_d_s, sector):
    portfolio_numbers = {}
    temp_fa = np.array(no_portfolio_f_a).astype(str)
    temp_fa = np.array([i.zfill(2) for i in temp_fa])
    portfolio_numbers['flood_agr'] = np.array([i + '_00_00_00' for i in temp_fa])
    temp_da = np.array(no_portfolio_d_a).astype(str)
    temp_da = np.array([i.zfill(2) for i in temp_da])
    portfolio_numbers['drought_agr'] = np.array(['00_' + i + '_00_00' for i in temp_da])
    temp_fu = np.array(no_portfolio_f_u).astype(str)
    temp_fu = np.array([i.zfill(2) for i in temp_fu])
    portfolio_numbers['flood_urb'] = np.array(['00_00_' + i + '_00' for i in temp_fu])
    temp_ds = np.array(no_portfolio_d_s).astype(str)
    temp_ds = np.array([i.zfill(2) for i in temp_ds])
    portfolio_numbers['drought_shp'] = np.array(['00_00_00_' + i for i in temp_ds])

    if sector == 'multihaz_agr':
        combinations_s2 = itertools.product(temp_fa, temp_da, np.array(['00']), np.array(['00']))
        portfolio_numbers['multihaz_agr'] = ['_'.join(combination) for combination in combinations_s2]
    elif sector == 'multihaz_urb':
        portfolio_numbers['multihaz_urb'] = np.array(['00_00_' + i + '_00' for i in temp_fu])


    combinations_s3 = itertools.product(temp_fa, temp_da, temp_fu, temp_ds)

    # Concatenate the strings from each combination while preserving the order

    portfolio_numbers['multihaz_multisec'] = ['_'.join(combination) for combination in combinations_s3]
    return portfolio_numbers

def create_folder(outfolder, folder_paths):
    notebook_dir = os.getcwd()
    foldername = outfolder
    if len(folder_paths) > 1:
        realization_names = str(folder_paths[0]) + '_to_' + str(folder_paths[-1])
    else:
        realization_names = str(folder_paths[0])

    outputfolder_dir = os.path.join(notebook_dir, foldername)
    try:
        os.mkdir(outputfolder_dir)
    except OSError as error:
        pass
    outputfolder_dir = os.path.join(outputfolder_dir, realization_names)
    try:
        os.mkdir(outputfolder_dir)
    except OSError as error:
        pass
    plots_dir = os.path.join(outputfolder_dir, 'plots')
    try:
        os.mkdir(plots_dir)
    except OSError as error:
        pass
    return outputfolder_dir, plots_dir

def robustness_Voudouris(data):
    # Sort the data in ascending order
    sorted_data = np.sort(data)

    # Calculate the robustness parameters
    n = len(sorted_data)
    r_10 = sorted_data[int(round(0.1 * n))]
    r_90 = sorted_data[int(round(0.9 * n))]
    r_50 = sorted_data[int(round(0.9 * n))]

    if r_90 == r_10:
        skewness = 0
    else:
        skewness = ((r_90 + r_10)/2 - r_50)/((r_90 - r_10)/2)

    return np.round(skewness,2)

def robustness_Kwakkel(data, optimization='minimization'):
    mean = np.mean(data)
    std = np.std(data)

    if optimization=='min':
        robustness = (mean + 1)*(std + 1)
    elif optimization=='max':
        robustness = (mean + 1) / (std + 1)
    else:
        print('error! optimization type not correctly specified!')
    out = np.round(robustness,2)
    return out

def robustness_mean(data, optimization='minimization'):
    mean = np.mean(data)
    out = np.round(mean,2)
    return out

def calculate_robustnessvalues(data, optimization='default'):
    mean = np.round(np.mean(data),2)
    # kwakkel = robustness_Kwakkel(data, optimization)
    # voudouris = robustness_Voudouris(data)

    val90 = np.quantile(data, 0.9, method='closest_observation')
    val10 = np.quantile(data, 0.1, method='closest_observation')

    # # find the values in the list closest to the 90th and 10th quantiles
    # val90 = min(data, key=lambda x: abs(x - q90))
    # val10 = min(data, key=lambda x: abs(x - q10))

    return [mean, val10, val90]
