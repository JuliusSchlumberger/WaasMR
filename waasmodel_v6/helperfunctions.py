from pcraster import *


def create_folder_hazard(location,realization_number, subfolder,subsubfolder='none'):
    '''
    Allows to create output folder & visualization folders. All generated data/figures are stored per day to avoid loss.
    '''
    notebook_dir = os.getcwd()
    location
    # create folder location
    location_dir = os.path.join(notebook_dir, location)
    try:
        os.mkdir(location_dir)
    except OSError as error:
        pass
    # create folder for realization
    realization_dir = os.path.join(location_dir, str(realization_number))
    try:
        os.mkdir(realization_dir)
    except OSError as error:
        pass
    # create stage folder
    out_dir = os.path.join(realization_dir, subfolder)
    try:
        os.mkdir(out_dir)
    except OSError as error:
        pass
    # create subfolder if stage == 1
    if subsubfolder != 'none':
        out_dir = os.path.join(out_dir, subsubfolder)
        try:
            os.mkdir(out_dir)
        except OSError as error:
            pass
    return out_dir

def Qhrelation(Q, Levq788, Fact788, Levq7150, Fact7150, Levq16000, Fact16000, Levq20000, Fact20000, Dif_2, Dif_6, Dif_7):
    '''
    Computation of the water level along river stretch for a given discharge
    :param Q:   mean river discharge [m3/s]
    :return:    water level along river stretch
    '''
    if Q < 788:
        h = (Levq788 * Fact788) - \
                    (Levq788 * Fact788) * ((788 - Q) / 788)
    elif Q < 7150:
        h = (Levq7150 * Fact7150) - Dif_2 * ((7150 - Q) / 6362)
    elif Q < 16000:
        h = (Levq16000 * Fact16000) - Dif_6 * ((16000 - Q) / 8850)
    elif Q < 20000:
        h = (Levq20000 * Fact20000) - Dif_7 * ((20000 - Q) / 4000)
    else:
        h = (Levq20000 * Fact20000) + (0.00005 * (Q - 20000))
    return h


def urban_growth(growth, LUmap):
    '''
    Updates land use map. Replaces part of the agricultural land with new urban areas. Inputs are provided in number
    of ha (100x100m) grid cells. Urban area can only grow.
    :param growth:      land change in certain year [ha]
    :param LUmap:       land use map to apply change to
    :return:            updated self.LandUse
    '''
    if growth > 0:
        # map with cell distances around all cities (resolution 100 m)
        SprStad = spread(ifthenelse(LUmap == ordinal(1), ordinal(1), ordinal(0)), 0, 1)
        # random cells next in direct proximity of all cities used for urban growth

        SprStad2 = ifthen(pcrand(LUmap == ordinal(9),
                                 pcrand(SprStad > 0, SprStad <= max(scalar(growth), scalar(100)))),
                          boolean(1))
        # new urban areas randomly grown
        SprStad3 = ifthen(order(uniform(SprStad2)) <= growth, ordinal(1))

        LandUse = cover(SprStad3, LUmap)

        # derive total agricultural area in the Waasregion
        agri_areas = maptotal(ifthen(LandUse == 9, scalar(1)))  # in [ha]!
        agri_areas = pcr2numpy(agri_areas, -999)[0, 0]
    else:
        LandUse = LUmap
        agri_areas = maptotal(ifthen(LandUse == 9, scalar(1)))  # in [ha]!
        agri_areas = pcr2numpy(agri_areas, -999)[0, 0]
    return LandUse, agri_areas