import matplotlib.pyplot as plt
import pandas as pd

from helperfunctions import *
from funcs_viz_new import *
import os

# create outputfolder
notebook_dir = os.getcwd()
vizfolder = 'inputs/data_plots/'
inputfolder = 'inputs/'
inputmaps = inputfolder + '/maps'


# Maps
plotmaps = False
inputs_flood_agr = False
inputs_drought_agr = True

def calc_rootvol(soilm_tm1,dr=0.2):
    rootvol = np.minimum(100.0, np.maximum(0.0, (
            soilm_tm1 / dr) * 100.0))
    return(rootvol)

def calc_pf_fit(rootvol):
    """Fit from input table (see Misc.py for the fit). Polyonimal fit 3rd degree."""
    pf = np.maximum(0, -2.46285709 * 10 ** (-5) * rootvol ** 3 + 3.53904855 * 10 ** (
        -3) * rootvol ** 2 - 2.02754270 * 10 ** (-1) * rootvol ** 1 + 6.89184020)
    return pf


if plotmaps:
    LandUse = readmap(r"{}".format(inputmaps + '/land_ini.pcr'))  # land use see landuse_code.txt
    fig = plot_advanced(LandUse,True,names=['urban', 'recreation', 'industry', 'nature', 'agriculture',
                                            'greenhouses', 'infrastructure', 'river', 'dikes'],
                        colourscheme='RdYlBu',title='land use in Waas region')
    fig.savefig(vizfolder + '\maps_LandUse.png', dpi=300)

    DEM = readmap(r"{}".format(inputmaps + '/dem_ini.pcr'))
    fig = plot_advanced(DEM,False, colourscheme='RdYlBu', label='DEM [m]', title='original DEM in Waas region')
    fig.savefig(vizfolder + '\maps_DEM.png', dpi=300)

    DijkRtot = readmap(r"{}".format(inputmaps + '/dikerings_nominal.pcr'))  # dike rings nominal with, river included
    fig = plot_advanced(DijkRtot, True, colourscheme='Accent', names=['0','1','2','3','4','5'], title='Dike rings inlcuding river plane' )
    fig.savefig(vizfolder + '\maps_DijkRtot.png', dpi=300)

    PrimDijk = readmap(
        r"{}".format(inputmaps + '/primdikes.pcr'))  # 1,2,3: North dike; 4,5: South Dike along river
    fig = plot_advanced(PrimDijk, True, colourscheme='Accent', names=['0','2','3','4','5'], title='Dike rings inlcuding river plane' )
    fig.savefig(vizfolder + '\maps_DPrimDijk.png', dpi=300)
if inputs_flood_agr:
    ClimQmax = pd.read_csv(inputfolder + '/discharge_max_decade_new.tss')
    ClimQmax = ClimQmax.values
    ClimQmax_dict = {}
    ClimQmax_dict['no climate change'] = ClimQmax[:,:10]
    ClimQmax_dict['medium climate change'] = ClimQmax[:,10:20]
    ClimQmax_dict['high climate change'] = ClimQmax[:, 20:30]
    create_frequency_histograms(ClimQmax_dict, xlabel='discharge in 10-day period [m3/s]', folder=vizfolder,
                                filename='fa_discharge_max_decade')

    ClimQmax = pd.read_csv(inputfolder + '/discharge_mean_decade_new.tss')
    ClimQmax = ClimQmax.values
    ClimQmax_dict = {}
    ClimQmax_dict['no climate change'] = ClimQmax[:, :10]
    ClimQmax_dict['medium climate change'] = ClimQmax[:, 10:20]
    ClimQmax_dict['high climate change'] = ClimQmax[:, 20:30]
    create_frequency_histograms(ClimQmax_dict, xlabel='discharge in 10-day period [m3/s]', folder=vizfolder,
                                filename='fa_discharge_mean_decade')

    # Frequencies per timesteps
    a = np.empty(3600)
    b = range(36)
    ind = np.arange(len(a))
    np.put(a, ind, b)
    print(len(a))
    no_climate = np.zeros((1,2))
    temp = np.zeros((3600,2))
    for i in range(10):
        temp[:,0] = ClimQmax[:,i]
        temp[:,1] = a
        no_climate = np.append(no_climate, temp, axis=0)

    medium_climate = np.zeros((1, 2))
    for i in range(10,20):
        temp[:, 0] = ClimQmax[:, i]
        temp[:, 1] = a
        medium_climate = np.append(medium_climate, temp, axis=0)

    high_climate = np.zeros((1, 2))
    for i in range(20,30):
        temp[:, 0] = ClimQmax[:, i]
        temp[:, 1] = a
        high_climate = np.append(high_climate, temp, axis=0)

    fig, ax = plt.subplots(nrows=3, sharex=True)
    fig.subplots_adjust(hspace=0.5)

    ax = ax.ravel()

    no_climate_pd = pd.DataFrame(no_climate,columns=['values', 'timesteps'])
    # no_climate_pd = no_climate_pd[no_climate_pd.values > 8500]
    no_climate_pd['timesteps'] = no_climate_pd['timesteps'].astype('int')
    no_climate_pd.boxplot(by='timesteps', ax=ax[0])
    medium_climate_pd = pd.DataFrame(medium_climate, columns=['values', 'timesteps'])
    # medium_climate_pd = medium_climate_pd[medium_climate_pd.values > 8500]
    medium_climate_pd['timesteps'] = medium_climate_pd['timesteps'].astype('int')
    medium_climate_pd.boxplot(by='timesteps', ax=ax[1])
    high_climate_pd = pd.DataFrame(high_climate, columns=['values', 'timesteps'])
    # high_climate_pd = high_climate_pd[high_climate_pd.values > 8500]
    high_climate_pd['timesteps'] = high_climate_pd['timesteps'].astype('int')
    high_climate_pd.boxplot(by='timesteps', ax=ax[2])

    vmax = np.maximum(np.array(np.max(no_climate_pd.values[:, 0])),
        np.array(np.max(medium_climate_pd.values[:, 0])),
        np.array(np.max(high_climate_pd.values[:, 0]))) * (1 + 0.1)
    vmin = 8500

    ax[0].set_title('no climate change', size=9)
    ax[0].set_ylim([vmin, vmax])

    # ax[1].legend_ = None
    ax[1].set_title('medium climate change', size=9)
    ax[1].set_ylim([vmin, vmax])

    # ax[2].legend_ = None
    ax[2].set_title('high climate change', size=9)
    ax[2].set_ylim([vmin, vmax])
    fig.text(0.02, 0.5, 'discharge [m3/s]', va='center', rotation='vertical')
    fig.savefig(vizfolder + '/fa_seasonal_Qdistribution.png', dpi=300, bbox_inches='tight')

    plt.clf()

    FragTbl = pd.read_table(inputfolder + '/FragTab01lsm.tbl', names=['bins', 'probability'])
    FragTbl['bins'] = np.linspace(-2.95, 5.9, len(FragTbl['bins']))
    FragTbl_plot = FragTbl.plot(kind='line', x='bins', title='Fragility Curve Dike', fontsize=12, logy=True)
    FragTbl_plot.set_xlabel('difference dike crest - water level [m]')
    FragTbl_plot.set_ylabel('Failure probability [-]')
    plt.savefig(vizfolder + '/fa_FragTbl_dike.png', dpi=300, bbox_inches='tight')

    DamFact = pd.read_table(inputfolder + '\Rhinemodel_inputs_forReference/damfact_forvisualization.tbl',
                                names=['min_depth', 'max_depth', 'land use type', 'damage factor'])
    fig, axes = plt.subplots(nrows=1, ncols=1)
    for i in range(1, 10, 1):
        DamFact_group = DamFact.loc[DamFact['land use type'] == i]
        land_use_types = ['urban', 'agricultural', 'industry', 'empty', 'infrastructure', 'empty', 'greenhouses',
                          'empty', 'empty']
        if land_use_types[i - 1] == 'empty':
            continue
        DamFact_group.plot(kind='line', x='min_depth', y='damage factor',
                           title='Damage factor for different land use types', ax=axes,
                           legend=DamFact_group['land use type'].unique(), label='type_' + land_use_types[i - 1])
    axes.set_xlabel('inundation depth [m]')
    axes.set_ylabel('damage factor')
    fig.savefig(vizfolder + '/fa_DamFact.png', dpi=300, bbox_inches='tight')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
    MaxDamTbl = pd.read_table(inputfolder + '/maxdam2000.tbl', names=['LU class', 'max damage [Mio EUR/ha]'])
    MaxDamTbl['LU class'] = ['empty', 'urban', 'empty', 'empty', 'empty', 'recreation', 'industry', 'empty', 'nature',
                             'agriculture', 'empty', 'greenhouses', 'empty', 'infrastructure', 'river', 'empty',
                              'dikes']

    RelevantMaxDam = MaxDamTbl[MaxDamTbl['LU class'] != 'empty']

    RelevantMaxDam.plot(kind = 'bar', x = 'LU class', y = 'max damage [Mio EUR/ha]', title = 'max. flood damages per LU class',
                        ax = axes, fontsize = 12, logy = True)
    axes.set_ylabel('max damage [Mio EUR/ha]')
    plt.subplots_adjust(bottom=0.25)
    fig.savefig(vizfolder + '/fa_MaxDam.png', dpi=300, bbox_inches='tight')
if inputs_drought_agr:
    evapo_scenarios = pd.read_csv(inputfolder + '/evaporation_scenarios_new.tss')  # reference evaporation according to Makkink [m/10d] at a given timestep
    evapo_scenarios = evapo_scenarios.values
    evapo_dict = {}
    evapo_dict['no climate change'] = evapo_scenarios[:,:10]
    evapo_dict['medium climate change'] = evapo_scenarios[:,10:20]
    evapo_dict['high climate change'] = evapo_scenarios[:, 20:30]
    create_frequency_histograms(evapo_dict, xlabel='evaporation [m/10d]', folder=vizfolder,
                                filename='da_evaporation_decade')
    precip_scenarios = pd.read_csv(inputfolder + '/precipitation_scenarios_new.tss')  # precipitation at a given timestep from transient scenario [m/10d]
    precip_scenarios = precip_scenarios.values
    precip_dict = {}
    precip_dict['no climate change'] = precip_scenarios[:, :10]
    precip_dict['medium climate change'] = precip_scenarios[:, 10:20]
    precip_dict['high climate change'] = precip_scenarios[:, 20:30]
    create_frequency_histograms(precip_dict, xlabel='precipitation [m/10d]', folder=vizfolder,
                                filename='da_precipitation_decade')
    raindef_dict = {}
    raindef_dict['no climate change'] = precip_scenarios[:, :10] - evapo_scenarios[:, :10]
    raindef_dict['medium climate change'] = precip_scenarios[:, 10:20] - evapo_scenarios[:, 10:20]
    raindef_dict['high climate change'] = precip_scenarios[:, 20:30] - evapo_scenarios[:, 20:30]
    create_frequency_histograms(raindef_dict, xlabel='10-day rain deficit [m]', folder=vizfolder,
                                filename='da_rain_deficit_decade')
    # tables
    cropcodes = ['grass', 'corn', 'potatoes', '(sugar) beet', 'cereals', 'other', 'arboriculture',
                 'glasshouse horticulture', 'fruit', 'bulbs', 'deciduous forest', 'coniferous forest',
                 'opengrownnaturegeb', 'bare ground natures']
    cropcodes_select = cropcodes[:6]
    print(cropcodes_select)
    # Crop Factor
    CropFactTbl = pd.read_table(inputfolder + '/CropFactor.tbl', delimiter=',')
    CropFactTbl['decade'] = CropFactTbl['decade']
    CropFactTbl_new = CropFactTbl.drop(columns='glasshouse horticulture')

    fig, ax = plt.subplots(sharey=True, sharex=True, ncols=3, nrows=int(np.ceil(len(cropcodes_select)/3)))
    fig.subplots_adjust(hspace=0.5)
    ax = ax.ravel()
    for k in range(len(cropcodes_select)):
        CropFactTbl_new.plot(kind='line', x='decade', y=cropcodes_select[k], ax=ax[k],title=cropcodes_select[k], legend=False)
        ax[k].grid()
        x_axis = ax[k].axes.get_xaxis()
        x_label = x_axis.get_label()
        ##print isinstance(x_label, matplotlib.artist.Artist)
        x_label.set_visible(False)
    fig.text(0.5, 0.04, 'timesteps in year [10d]', ha='center')
    fig.text(0.02, 0.5, 'crop factor', va='center', rotation='vertical')

    for axes in ax.flat:
        ## check if something was plotted
        if not bool(axes.has_data()):
            plt.delaxes(axes)  ## delete if nothing is plotted in the axes obj

    fig.savefig(vizfolder + '/da_CropFactor.png', dpi=300, bbox_inches='tight')

    # Death Damage
    DeathPointTbl = pd.read_table(inputfolder + '/DeathDam.tbl', delimiter=',')
    DeathPointTbl['decade'] = DeathPointTbl['decade']
    DeathPointTbl = DeathPointTbl.drop(columns='glasshouse horticulture')
    DeathPointTbl_selected = DeathPointTbl[cropcodes_select]

    fig, ax = plt.subplots(sharey=True, sharex=True, ncols=3, nrows=int(np.ceil(len(cropcodes_select) / 3)))
    ax = ax.ravel()
    for k in range(len(cropcodes_select)):
        DeathPointTbl_selected.plot(kind='line', y=cropcodes_select[k], ax=ax[k], title=cropcodes_select[k],
                             legend=False)
        ax[k].grid()
        x_axis = ax[k].axes.get_xaxis()
        x_label = x_axis.get_label()
        ##print isinstance(x_label, matplotlib.artist.Artist)
        x_label.set_visible(False)
    fig.text(0.5, 0.04, 'timesteps in year [10d]', ha='center')
    fig.text(0.02, 0.5, 'damage at death point', va='center', rotation='vertical')

    for axes in ax.flat:
        ## check if something was plotted
        if not bool(axes.has_data()):
            plt.delaxes(axes)  ## delete if nothing is plotted in the axes obj

    fig.savefig(vizfolder + '/da_DeathDam.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Epot for various crops
    EpotTotRefTbl = pd.read_table(inputfolder + '/EpotTotRef.tbl', skiprows=1, names=['cropcode', 'Epot [mm]'])
    EpotTotRefTbl['cropcode'] = range(len(cropcodes))
    EpotTotRefTbl_selected = EpotTotRefTbl[EpotTotRefTbl['cropcode'] < len(cropcodes_select)]
    fig, ax = plt.subplots()
    EpotTotRefTbl_selected.plot(kind='bar', x='cropcode', y='Epot [mm]',
                       title='reference evaporation according to Makkink [mm/year]', ax=ax)
    ax.set_ylabel('Epot [mm/year]')
    plt.xticks(rotation=90)
    create_add_legend2(path_names=np.array(cropcodes_select), ax=ax)
    fig.savefig(vizfolder + '/da_EpotTotRef.png', dpi=300,bbox_inches='tight')

    # Productivity
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    CropProduct = pd.read_table(inputfolder + '/cropyield.tbl', skiprows=1, names=['cropcode', 'Productivity'],
                                delim_whitespace=True)
    CropPrize = pd.read_table(inputfolder + '/Cropprize.tbl', skiprows=1, names=['cropcode', 'CropPrize'],
                              delim_whitespace=True)
    cropcodes = ['grass', 'corn', 'potatoes', '(sugar) beet', 'cereals', 'other', 'arboriculture',
                               'fruit', 'bulbs', 'deciduous forest', 'coniferous forest',
                               'opengrownnaturegeb', 'bare ground natures']
    CropProduct['cropcode'] = range(len(cropcodes))
    CropProduct['CropPrize'] = CropPrize['CropPrize']
    CropInfo = CropProduct[CropProduct['cropcode'] < len(cropcodes_select)]
    CropInfo = CropInfo.values

    width = 0.45
    # ln1 = CropInfo.Productivity.plot(kind='bar', ax=ax1, fontsize=12, logy=True, position=1, width=width, color='g', label='Productivity [kg/ha, kg ds/ha, or stuk/ha]')
    # ln2 = CropInfo.CropPrize.plot(kind='bar', ax=ax2, fontsize=12, logy=True, position=0, width=width, color='b', label='Crop Prize per kg, stuk')
    ax1.bar(CropInfo[:, 0], CropInfo[:, 1], log=True, width=-width, align='edge', label='Productivity', color='g')
    ax2.bar(CropInfo[:, 0], CropInfo[:, 2], log=True, width=width, align='edge', label='Crop Prize', color='b')

    ax1.legend(loc=2)
    ax2.legend(loc=1)

    create_add_legend2(path_names=np.array(cropcodes_select), ax=ax2)
    ax1.set_ylabel('Productivity [kg/ha, kg ds/ha, or stuk/ha]')
    ax2.set_ylabel('Crop Prize per kg, stuk [EUR]')
    fig.savefig(vizfolder + '/da_CropInfo.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # Pf
    Pfveen = np.loadtxt(inputfolder + '/pf_peat.txt', skiprows=1, delimiter='\t')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ln1 = ax1.plot(calc_rootvol(np.linspace(0,0.1,50)), np.linspace(0,0.1,50),'-r', label='soilm_tm1')
    ax1.set_ylabel('soilm_tm1')
    ln2 = ax2.plot(Pfveen[:,0],Pfveen[:,1], label='pf')
    # ln2 = Pfveen.plot(kind='line', x='Root volume', y='Pf', ax=ax2, title='Soil moisture suction (Pf) of peat soil')
    ax2.set_ylabel('pf')
    ln3 = ax2.plot(np.linspace(0,100,9), calc_pf_fit(np.linspace(0,100,9)), label='curve_fit')

    ax1.set_xlabel('Root volume [%]')
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.grid()
    fig.savefig(vizfolder + '/da_Pf_peat.png', dpi=300, bbox_inches='tight')

    # Remaining Yield, yet to be harversted
    RemYield = pd.read_csv(inputfolder + '/RemYield.tbl', delimiter=',', header=0)
    RemYield = RemYield[['decade', *cropcodes_select]]

    fig, ax = plt.subplots(sharey=True, sharex=True, ncols=3, nrows=int(np.ceil(len(cropcodes_select) / 3)))
    ax = ax.ravel()
    for k in range(len(cropcodes_select)):
        RemYield.plot(kind='line',x='decade', y=cropcodes_select[k], ax=ax[k], title=cropcodes_select[k],
                                    legend=False)
        ax[k].grid()
        x_axis = ax[k].axes.get_xaxis()
        x_label = x_axis.get_label()
        ##print isinstance(x_label, matplotlib.artist.Artist)
        x_label.set_visible(False)
    fig.text(0.5, 0.04, 'timesteps in year [10d]', ha='center')
    fig.text(0.02, 0.5, 'remaining yield [%]', va='center', rotation='vertical')

    for axes in ax.flat:
        ## check if something was plotted
        if not bool(axes.has_data()):
            plt.delaxes(axes)  ## delete if nothing is plotted in the axes obj
    plt.xticks(rotation=90)
    fig.savefig(vizfolder + '/da_RemYield.png', dpi=300, bbox_inches='tight')
    plt.clf()

    # urb_scen = pd.read_csv(inputfolder + '/urb_scenarios.tss', header=None, delim_whitespace=True)
    # fig = plt.figure()
    # bp = plt.boxplot(urb_scen)
    # plt.ylabel('urban growth [ha/20years]')
    # plt.xlabel('urban land use update in year')
    # plt.xticks([1,2,3,4,5],[20, 40, 60, 80, 100])
    # plt.title('Variability of urban growth scenarios')
    # fig.savefig(vizfolder + '/trans_urb_growth.png', dpi=300)


# Tables



