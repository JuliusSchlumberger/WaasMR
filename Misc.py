import pandas as pd
import numpy as np
from pcraster import *
from itertools import permutations
from itertools import combinations
import pickle
# from helperfunctions import sequence_pairs
import matplotlib.pyplot as plt
import gzip

def read_csv_file(file):
    # read the compressed file back into a DataFrame
    with gzip.open(file, 'rb') as f:
        df = pd.read_csv(f)
    return df

fname = f'model_outputs/005000/portfolio_00_06_00_00.csv'

df = read_csv_file(f'{fname}.gz')
df.to_csv(fname)
print(df)
print(error)

create_ints_forMeasures = False
#region translate measure strings into integer sequences
sectors = ['flood_agr', 'flood_urb', 'drought_agr', 'drought_shp']
measures_sectors = ['measure_agr_f', 'measure_urb_f', 'measure_agr_dr','measure_shp_dr']
measures_dict = {}
measures_dict['measure_agr_dr'] = ['no_measure', 'd_resilient_crops', 'd_rain_irrigation', 'd_gw_irrigation', 'd_riv_irrigation']
measures_dict['measure_agr_f'] = ['no_measure','f_resilient_crops', 'f_ditches', 'f_dike_elevation_s', 'f_dike_elevation_l', 'f_room_for_river']
measures_dict['measure_urb_f'] = ['no_measure','f_dike_elevation_s', 'f_dike_elevation_l', 'f_room_for_river', 'f_wet_proofing_houses', 'f_local_protect', 'f_noconstrucarea']
measures_dict['measure_shp_dr'] = ['no_measure','d_multimodal_transport', 'd_medium_ships', 'd_small_ships', 'd_dredging']
measure_combos = ['inputs/measure_storylines_flood_agr.txt', 'inputs/measure_storylines_flood_urb.txt', 'inputs/measure_storylines_drought_agr.txt', 'inputs/measure_storylines_drought_shp.txt']

if create_ints_forMeasures:
    for sector in range(4):
        measure_sequences = pd.read_csv(measure_combos[sector], names=['1','2', '3', '4'], delim_whitespace=True, dtype='str')
        print(measure_sequences)
        number_idx = len(measures_dict[measures_sectors[sector]])

        for measure in range(number_idx):
            print(measure)
            measure_sequences.replace(to_replace=measures_dict[measures_sectors[sector]][measure], value=str(measure), inplace=True)
        measure_sequences.replace(to_replace=np.NaN, value=0, inplace=True)
        # measure_sequences[0,:] = ['0', '0', '0', '0']
        measure_sequences.astype('int')
        print(measure_sequences)
        measure_sequences.to_csv('outputs/parallel_processed/00_sequences_as_ints_'+sectors[sector] + '.txt', header=False, index=False )

#endregion

batch_file = False
#region Create list of potential pathways
if batch_file:
    for runfile in range(1,5):
        pathways = [17, 32, 17, 19]
        multiprocesses = np.zeros((32, 9)).astype('str')
        multiprocesses[:, 0] = 'python'
        multiprocesses[:, 1] = 'run_file_parallel_snell_tp'+str(runfile)+'.py'
        multiprocesses[:, 8] = '&'
        for j in [1,2]:
            for i in range(4):
                multiprocesses[i*8:(8+i*8):, 2] = str(i)  # hazard/sector
                if j == 1:
                    multiprocesses[i * 8:(4 + i * 8), 3] = '1'  # min_realization
                    multiprocesses[i * 8:(4 + i * 8), 4] = '4'  # max_realization
                    multiprocesses[(4 + i * 8):(8 + i * 8), 3] = '4'  # min_realization
                    multiprocesses[(4 + i * 8):(8 + i * 8), 4] = '7'  # max_realization
                elif j == 2:
                    multiprocesses[i * 8:(4 + i * 8), 3] = '7'  # min_realization
                    multiprocesses[i * 8:(4 + i * 8), 4] = '10'  # max_realization
                    multiprocesses[(4 + i * 8):(8 + i * 8), 3] = '10'  # min_realization
                    multiprocesses[(4 + i * 8):(8 + i * 8), 4] = '14'  # max_realization
                multiprocesses[:,5] = str(runfile)   # Run number for subfolder creation

                # for storylines
                # scen range 1 to 25
                multiprocesses[0 + 8 * i, 6] = str(int(0 * pathways[i]/4))  # min_seq
                multiprocesses[0 + 8 * i, 7] = str(int(1 * pathways[i]/4))  # max_seq
                multiprocesses[1 + 8 * i, 6] = str(int(1 * pathways[i]/4))  # min_seq
                multiprocesses[1 + 8 * i, 7] = str(int(2 * pathways[i]/4))  # min_seq
                multiprocesses[2 + 8 * i, 6] = str(int(2 * pathways[i]/4))  # min_seq
                multiprocesses[2 + 8 * i, 7] = str(int(3 * pathways[i]/4))  # min_seq
                multiprocesses[3 + 8 * i, 6] = str(int(3 * pathways[i]/4))  # min_seq
                multiprocesses[3 + 8 * i, 7] = str(int(4 * pathways[i]/4))  # min_seq
                # scen range 26 to 51
                multiprocesses[4 + 8 * i, 6] = str(int(0 * pathways[i]/4))  # min_seq
                multiprocesses[4 + 8 * i, 7] = str(int(1 * pathways[i]/4))  # max_seq
                multiprocesses[5 + 8 * i, 6] = str(int(1 * pathways[i]/4))  # min_seq
                multiprocesses[5 + 8 * i, 7] = str(int(2 * pathways[i]/4))  # min_seq
                multiprocesses[6 + 8 * i, 6] = str(int(2 * pathways[i]/4))  # min_seq
                multiprocesses[6 + 8 * i, 7] = str(int(3 * pathways[i]/4))  # min_seq
                multiprocesses[7 + 8 * i, 6] = str(int(3 * pathways[i]/4))  # min_seq
                multiprocesses[7 + 8 * i, 7] = str(int(4 * pathways[i]/4))  # min_seq
            np.savetxt('batch_file_inputs_'+str(j)+'_tp'+str(runfile)+'_.txt', multiprocesses, fmt='%s')
#endregion

createmeasuresfiles = False
#region Create Measure files
if createmeasuresfiles:
    f_measures_agri_list = ['f_resilient_crops','f_ditches', 'f_lease_land','f_dike_elevation_s',
                            'f_dike_elevation_l', 'f_room_for_river']
    # f_measures_agri_list = ['f_resilient_crops', 'f_ditches', 'f_dike_elevation_s',
    #                         'f_dike_elevation_l', 'f_room_for_river']
    f_measures_urb_list = ['f_dike_elevation_s','f_dike_elevation_l', 'f_room_for_river',
                           'f_wet_proofing_houses', 'f_local_protect', 'f_noconstrucarea']
    d_measures_agri_list = ['d_resilient_crops', 'd_rain_irrigation', 'd_gw_irrigation', 'd_riv_irrigation']
    d_measures_ship_list = ['d_multimodal_transport', 'd_medium_ships', 'd_small_ships', 'd_dredging']
    names = ['flood_agr', 'flood_urb', 'drought_agr', 'drought_shp']

    lists = [f_measures_agri_list, f_measures_urb_list, d_measures_agri_list, d_measures_ship_list]
    number_decisions = 4

    for i in range(len(lists)):
        combis = list()
        combis_num = list()
        storys = list()
        # combis = list(itertools.combinations(lists[i],number_decisions))
        num_values = np.arange(1,len(lists[i])+1).astype('str')
        # Random selection of pathways
        for n in range(len(lists[i]) + 1):
            temp = np.minimum(n, number_decisions)
            combis += list(permutations(lists[i], int(temp)))
            combis_num += list(permutations(num_values, int(temp)))
        np.savetxt('inputs/measure_sequences_'+str(names[i])+'.txt', combis, fmt="%s")
        print(len(combis_num))
        with open('inputs/measure_storylines2_' + str(names[i]) + '.txt', "wb") as fp:  # Pickling
            pickle.dump(combis, fp)
        # print(combis)
        tester = np.zeros((len(combis_num),number_decisions+1))
        tester[:,-1] = 1

        for j in range(1, len(combis_num)):
            if combis_num[j] != ():
                tester[j,:len(combis_num[j])] = combis_num[j]
        np.savetxt('outputs/parallel_processed/00_pathways_numeric_'+str(names[i]) + '.txt', tester)
        #     combined_list.append(''.join(combis_num[j]))
        # print(np.array(combined_list).astype('int'))
#endregion

#region Create input timeseries from Rhine model
run = False
names = ['Neerslag_01mm_sum_decade', 'Verdamping_01mm_sum_decade', 'qlobith_gemdecade', 'qlobith_maxdecade', 'NeerslagVerdamping_01mm_sumdecade']
list_names = np.zeros((30,1)).astype('str')
list_names[:10,0] = 'D'
list_names[10:20,0] = 'G'
list_names[20:30,0] = 'Wp'
scenarios = range(1,11)
scenarios = np.append(scenarios, range(21,31))
scenarios = np.append(scenarios, range(41,51))
print(scenarios)

scenarios_names = np.tile(np.array(range(1,11)).reshape(-1,1), (3,1)).reshape(-1,1)
print(scenarios_names)
print(list_names)
title_names = []
for s, i in zip(list_names, scenarios_names):
    print(s, i)
    # print(s[0], str(i))
    title_names.append(s[0] + str(i[0]))
print(title_names)

if run:
    tick = 0
    output_precip = np.zeros((3600,len(scenarios)))
    output_evapo = output_precip
    gem_q = np.zeros((3600, len(scenarios)))
    max_q = gem_q
    for i in scenarios:
        print(i)
        input = np.loadtxt('inputs\Rhinemodel_inputs_forReference/' + names[4] + '.'+str(i))
        output_precip[:,tick] = input[:,0] * 0.1 / 1000
        tick += 1
    # print(output_precip, title_names)
    tester = pd.DataFrame(output_precip, columns=title_names)
    tester.to_csv('inputs/precipitation_scenarios_new.tss', index=False)

    tick = 0
    for i in scenarios:
        input = np.loadtxt('inputs\Rhinemodel_inputs_forReference/' + names[4] + '.' + str(i))
        output_evapo[:,tick] = input[:,1] * 0.1 / 1000
        print(output_evapo[:,tick])
        gem_q[:, tick] = np.loadtxt('inputs\Rhinemodel_inputs_forReference\qlobith_gemdecade.' + str(i))[:, 1]
        max_q[:, tick] = np.loadtxt('inputs\Rhinemodel_inputs_forReference\qlobith_maxdecade.' + str(i))[:, 1]
        tick += 1
    tester = pd.DataFrame(max_q, columns=title_names)
    tester.to_csv('inputs/discharge_max_decade_new.tss', index=False)
    tester = pd.DataFrame(gem_q, columns=title_names)
    tester.to_csv('inputs/discharge_mean_decade_new.tss', index=False)
    tester = pd.DataFrame(output_evapo, columns=title_names)
    tester.to_csv('inputs/evaporation_scenarios_new.tss', index=False)

    # np.savetxt('inputs/discharge_max_decade_new.tss', max_q, header=",".join(title_names), delimiter=",")
    # np.savetxt('inputs/discharge_mean_decade_new.tss', gem_q, header=",".join(title_names), delimiter=",")
    # np.savetxt('inputs/evaporation_scenarios_new.tss', output_evapo, header=",".join(title_names), delimiter=",")

#endregion

#region created new randomreeks
run = False
if run:
    r = np.loadtxt('inputs/archive/randomreeks2z45_Rhinemodel_new.tss', skiprows=4)
    r[:,1] = r[:,1]
    for i in range(len(r[:,1])):
        if r[i,1] > 1:
            r[i,1] = 1
    np.savetxt('inputs/randoms_new.txt', r)
#endregion
run = True
#region dike elevation
if run:
    Case = readmap(r"{}".format('inputs/maps/case.pcr'))  # map of zeros
    LandUse = readmap(r"{}".format('inputs/maps/land_ini.pcr'))  # land use see landuse_code.txt
    DEM = readmap(r"{}".format('inputs/archive/dem_ini.pcr'))
    LandUse = ifthenelse(LandUse==18, DEM + 0.5 ,DEM)
    report(LandUse,'inputs/maps/dem_ini.pcr')
#endregion

run = False
#region Create urban growth scenarios
if run:
    # assumed that urban growth only updates every 20 years
    urb_scen_low = np.random.normal(loc=2, scale=10, size=(5, 15)).round()  # 10 scenarios for low urban growth
    urb_scen_mid = np.random.normal(loc=70, scale=150, size=(5, 20)).round()  # 10 scenarios for low urban growth
    urb_scen_high = np.random.normal(loc=130, scale=80, size=(5, 15)).round()  # 10 scenarios for low urban growth

    urb_scens = np.append(urb_scen_low,urb_scen_mid, axis=1)
    urb_scens = np.append(urb_scens, urb_scen_high, axis=1)
    np.savetxt('inputs/urb_scenarios.tss', urb_scens)

#endregion

fitpfCurve = False
#region fit Pf table to curve
if fitpfCurve:
    PfveenTbl = np.loadtxt('inputs/pf_peat.txt', delimiter='\t', skiprows=1)
    curve = np.polyfit(PfveenTbl[:-1,0],PfveenTbl[:-1,1], deg=3)
    xvalues = np.linspace(0,100,100)
    function_fit = np.poly1d(curve)
    print(curve)
    # print(PfveenTbl[0,:])
    plt.plot(PfveenTbl[:,0],PfveenTbl[:,1],label='original')
    plt.plot(xvalues, function_fit(xvalues), label='fit')
    plt.legend()
    plt.show()
#endregion



#endregion

# Create separate DamFact Tables
def adjust_crop_res(inputtabl):
    inputtabl.loc[inputtabl.dam_curve == 2, 'damfact'] = inputtabl.loc[
                                                               inputtabl.dam_curve == 2, 'damfact'] - 0.1
    inputtabl.loc[inputtabl.damfact < 0, 'damfact'] = 0
    return inputtabl

# Create separate DamFact Tables
def adjust_crop_res_drought(inputtabl):
    inputtabl.loc[inputtabl.dam_curve == 2, 'damfact'] = inputtabl.loc[
                                                               inputtabl.dam_curve == 2, 'damfact'] + 0.1
    inputtabl.loc[inputtabl.damfact > 1, 'damfact'] = 1
    return inputtabl

def adjust_floodprone(inputtabl):
    inputtabl.loc[inputtabl.dam_curve == 1, 'damfact'] = inputtabl.loc[
                                                               inputtabl.dam_curve == 1, 'damfact'] * 0.9
    return inputtabl

def adjust_locprotect(inputtabl):
    inputtabl.loc[0:8, 'damfact'] = 0
    return inputtabl
#
sepDamFact = False
if sepDamFact:
    damfactTbl = pd.read_csv('inputs/damfact.tbl', delim_whitespace=True, names=['range', 'dam_curve', 'damfact'])
    print(damfactTbl)

    # flood resilient crops
    inputtabl = damfactTbl
    flood_res = adjust_crop_res(inputtabl=inputtabl)
    flood_res.to_csv('inputs/measures/damfact_fres.tbl', header=False, sep='\t')

    inputtabl = pd.read_csv('inputs/measures/damfact_fres.tbl', delim_whitespace=True, names=['range', 'dam_curve', 'damfact'])
    flood_res_floodprone = adjust_floodprone(inputtabl)
    flood_res_floodprone.to_csv('inputs/measures/damfact_fres_floodprone.tbl', header=False, sep='\t')

    inputtabl = pd.read_csv('inputs/measures/damfact_fres.tbl', delim_whitespace=True,
                            names=['range', 'dam_curve', 'damfact'])
    flood_res_locprotect = adjust_locprotect(inputtabl)
    flood_res_locprotect.to_csv('inputs/measures/damfact_fres_locprotect.tbl', header=False, sep='\t')

    inputtabl = pd.read_csv('inputs/measures/damfact_fres_floodprone.tbl', delim_whitespace=True,
                            names=['range', 'dam_curve', 'damfact'])
    flood_res_floodprone_locprotect = adjust_locprotect(inputtabl)
    flood_res_floodprone_locprotect.to_csv('inputs/measures/damfact_fres_floodprone_locprotect.tbl', header=False, sep='\t')

    # floodprone
    inputtabl = damfactTbl
    floodprone = adjust_floodprone(inputtabl)
    floodprone.to_csv('inputs/measures/damfact_floodprone.tbl', header=False, sep='\t')

    floodprone_locprotect = adjust_locprotect(floodprone)
    floodprone_locprotect.to_csv('inputs/measures/damfact_floodprone_locprotect.tbl', header=False, sep='\t')

    # local protect
    inputtabl = pd.read_csv('inputs/damfact.tbl', delim_whitespace=True, names=['range', 'dam_curve', 'damfact'])
    locprotect = adjust_locprotect(inputtabl)
    locprotect.to_csv('inputs/measures/damfact_locprotect.tbl', header=False, sep='\t')


sepDamFact_dres = False
if sepDamFact_dres:
    damfactTbl = pd.read_csv('inputs/damfact.tbl', delim_whitespace=True, names=['range', 'dam_curve', 'damfact'])
    print(damfactTbl)

    # drought resilient crops
    inputtabl = damfactTbl
    drought_res = adjust_crop_res_drought(inputtabl=inputtabl)
    drought_res.to_csv('inputs/measures/damfact_dres.tbl', header=False, sep='\t')

    inputtabl = pd.read_csv('inputs/measures/damfact_dres.tbl', delim_whitespace=True, names=['range', 'dam_curve', 'damfact'])
    drought_res_floodprone = adjust_floodprone(inputtabl)
    drought_res_floodprone.to_csv('inputs/measures/damfact_dres_floodprone.tbl', header=False, sep='\t')

    inputtabl = pd.read_csv('inputs/measures/damfact_dres.tbl', delim_whitespace=True,
                            names=['range', 'dam_curve', 'damfact'])
    drought_res_locprotect = adjust_locprotect(inputtabl)
    drought_res_locprotect.to_csv('inputs/measures/damfact_dres_locprotect.tbl', header=False, sep='\t')

    inputtabl = pd.read_csv('inputs/measures/damfact_dres_floodprone.tbl', delim_whitespace=True,
                            names=['range', 'dam_curve', 'damfact'])
    drought_res_floodprone_locprotect = adjust_locprotect(inputtabl)
    drought_res_floodprone_locprotect.to_csv('inputs/measures/damfact_dres_floodprone_locprotect.tbl', header=False, sep='\t')
