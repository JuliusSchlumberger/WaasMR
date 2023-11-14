from pcraster import *
import numpy as np
import pandas as pd



class ModelInputs:
    def __init__(self, stage,inputfolder='waasmodel_v6/inputs'):
        '''
        Initializing all inputs
        :param inputfolder: default: 'inputs'.
        '''
        self.inputfolder = inputfolder
        self.inputmaps = self.inputfolder + '/maps'
        
        # general inputs
        self.urb_scen = np.loadtxt(self.inputfolder + '/urb_scenarios.tss')

        # overarching inputs
        self.crop_type = 2  # change if not corn cultivated

        # inputs_floodmodule:
        # switches
        self.dike_maintenance = False
        self.floodresil = False
        self.f_ditch = False
        self.floodprone = False
        self.largedike = False
        self.localprotect = False
        self.noconstruction = False
        self.smalldike = False

        # timeseries
        self.ClimQmax = pd.read_csv(self.inputfolder + '/discharge_max_decade_new.tss')
        self.ClimQmax = self.ClimQmax.values


        self.randomR = np.loadtxt(self.inputfolder + '/randoms_new.txt')  # Random value for dike breaking for each year

        # maps
        self.Case = readmap(r"{}".format(self.inputmaps + '/case.pcr'))  # map of zeros
        self.DEM = readmap(r"{}".format(self.inputmaps + '/dem_ini.pcr'))
        self.DikeRtot = readmap(r"{}".format(
            self.inputmaps + '/dikerings_nominal.pcr'))  # dike rings nominal with, river included
        self.DikeRing = readmap(
            r"{}".format(self.inputmaps + '/dikerings_noriv.pcr'))  # nominal, floodplain not included
        self.LandUse = readmap(r"{}".format(self.inputmaps + '/land_ini.pcr'))  # land use see landuse_code.txt
        Levq788 = readmap(r"{}".format(self.inputmaps + '/levelsq788.pcr'))  # water level along river for discharge of 788 m3/s at Lobith
        Levq7150 = readmap(r"{}".format(self.inputmaps + '/levelsq7150.pcr'))
        Levq16000 = readmap(r"{}".format(self.inputmaps + '/levelsq16000.pcr'))
        Levq20000 = readmap(r"{}".format(self.inputmaps + '/levelsq20000.pcr'))
        self.PrimDike = readmap(
            r"{}".format(self.inputmaps + '/primdikes.pcr'))  # 1,2,3: North dike; 4,5: South Dike along river

        # tables
        CropFactTbl_in = np.loadtxt(self.inputfolder + '/CropFactor.tbl', skiprows=1, delimiter=',')  # cropcode, decade, cropfactor
        CropPrizeTbl = r"{}".format(self.inputfolder + '/Cropprize.tbl')  # EUR/kg / EUR/stuk
        DamFactTbl = pd.read_csv("{}".format(self.inputfolder + '/damfact.tbl'), names=['start', 'end',
                                                                                            'class', 'damfact'])  # from flood
        self.DamFactTbl = DamFactTbl[(DamFactTbl['class'] == 1) | (DamFactTbl['class'] == 2)]
        self.DamFactTbl_pivot = self.DamFactTbl.pivot(index='start', columns='class', values='damfact')
        # Resetting the index and accessing values
        self.DamFactTbl = self.DamFactTbl_pivot.reset_index().values

        self.DamFunctTbl = r"{}".format(self.inputfolder + '/damfunction.tbl')  # relation land use and damage function
        self.FragTbl = r"{}".format(self.inputfolder + '/FragTab01lsm.tbl')  # dike fragility table based on dike height
        self.MaxDamTbl = r"{}".format(self.inputfolder + '/maxdam2000.tbl')  # maximum damage per LU
        self.MaxDam_Agr = lookupscalar(CropPrizeTbl, self.crop_type)  # maximum damage for cultivated crop
        self.RfR_factors = pd.read_csv(self.inputfolder + '/RfR_factors.txt', delim_whitespace=True)
        self.raindef_mean = pd.read_csv(self.inputfolder + '/mean_raindeficit_NLD.txt', index_col=0)
        self.raindef_mean.loc[:,'deficit'] = self.raindef_mean.loc[:, 'deficit'] / 1000

        # parameters
        self.dikefrag_reduction_ini = 1
        self.dike_areas = 2 * 125  # [ha]
        self.dike_weight_rel = 1
        self.dr_dike = 0.2
        self.dyn_vulnerability_agr_fact = 1
        self.dyn_vulnerability_urb_fact = 1  # flood-flood consecutive for urban sector
        self.dyn_vulnerability_urb_factors = np.array([1.2, 0.8, 0.9, 0.95,
                                                       0.97])  # first year higher vulnerability, then reduced exposure
        self.riskaware_period = 5
        self.f_ditch_red = 0.0
        self.groundwlvl_dike = -1.0  # groundwater level is 1 m below surface
        self.HoverDEM_dike = -2  # river water level is on average 2 m above surface (because of dikes)
        self.last_flood = 10000  # arbitrary high value
        self.num_dikefails = 0
        self.porosity_dike = 0.5
        self.revenue_reduc = 1
        self.RfR = 1
        self.soilm_dike = 0.1

        # Additional initializations
        self.agrprize = pcr2numpy(self.MaxDam_Agr, -999)[0, 0]  # value of crop for drought module
        self.CropFactTbl_dike = CropFactTbl_in[:, 1]  # grass on dikes
        self.expos = lookupscalar(self.MaxDamTbl, self.LandUse)  # max damage per LU

        Levq788 = pcr_as_numpy(Levq788)
        Levq7150 = pcr_as_numpy(Levq7150)
        Levq16000 = pcr_as_numpy(Levq16000)
        Levq20000 = pcr_as_numpy(Levq20000)
        self.Levq788 = Levq788[~np.isnan(Levq788)]
        self.Levq7150 = Levq7150[~np.isnan(Levq7150)]
        self.Levq16000 = Levq16000[~np.isnan(Levq16000)]
        self.Levq20000 = Levq20000[~np.isnan(Levq20000)]
        
        self.Fact788 = self.RfR_factors.iloc[0][self.RfR]
        self.Fact7150 = self.RfR_factors.iloc[1][self.RfR]
        self.Fact16000 = self.RfR_factors.iloc[2][self.RfR]
        self.Fact20000 = self.RfR_factors.iloc[3][self.RfR]

        self.Dif_2 = (self.Levq7150 * self.Fact7150) - (self.Levq788 * self.Fact788)
        self.Dif_6 = (self.Levq16000 * self.Fact16000) - (self.Levq7150 * self.Fact7150)
        self.Dif_7 = (self.Levq20000 * self.Fact20000) - (self.Levq16000 * self.Fact16000)

        # inputs_drought_agr_module:
        # switches
        self.collectwater = False
        self.d_groundwater_irrigation = False  # switch to initiate usage of groundwater pumps
        self.d_river_irrigation = False  # switch to initiate usage of river water
        self.d_rainwater_tank = False  # switch to initiate collecting of rainwater

        # Time-series
        self.evapo_ref_scenarios = pd.read_csv(
            self.inputfolder + '/evaporation_scenarios_new.tss')  # reference evaporation according to Makkink [m/10d] at a given timestep
        self.evapo_ref_scenarios = self.evapo_ref_scenarios.values
        self.precip_scenarios = pd.read_csv(
            self.inputfolder + '/precipitation_scenarios_new.tss')  # precipitation at a given timestep from transient scenario [m/10d]
        self.precip_scenarios = self.precip_scenarios.values
        self.ReducDamageTbl = np.loadtxt(self.inputfolder + '/ReducDamage.tbl', skiprows=1,
                                         delimiter=',')  # cropcode, decade, reduc damage

        # tables
        CropFactTbl_in = np.loadtxt(self.inputfolder + '/CropFactor.tbl', skiprows=1,
                                         delimiter=',')  # cropcode, decade, cropfactor
        self.CropFactTbl = CropFactTbl_in[:, self.crop_type]
        CropYieldTbl = np.loadtxt(self.inputfolder + '/cropyield.tbl', skiprows=1)  # kg/ha, kg ds/ha or pieces/ha
        self.DeathDamageTbl = np.loadtxt(self.inputfolder + '/DeathDam.tbl', skiprows=1,
                                         delimiter=',')  # cropcode, decade, deathpoint
        self.EpotTotRefTbl = np.loadtxt(self.inputfolder + '/EpotTotRef.tbl', skiprows=1)  # cropcode, in mm
        self.RemYieldTbl = np.loadtxt(self.inputfolder + '/remYield.tbl', skiprows=1,
                                      delimiter=',')  # cropcode, decade, remaining yield
        self.ReducPointTbl = np.loadtxt(
            self.inputfolder + '/ReducPoint.tbl', skiprows=1,
            delimiter=',')  # cropcode, decade, reduction point // does not change over time, so could remove the decade

        # parameters
        self.caprise_m = 0.002  # default 0.001-0.002 m/decade (from Haasnoot et al. 2014)
        self.crop_vul = 0
        self.damfrac_t_d = 0
        self.damfrac_tot_d = 0
        self.DamAgr_d = 0
        self.DamAgr_d_tot = 0
        self.dc = 2.0  # Depth at which capillary rise is 0. This depends on the soil [m]
        self.dr = 0.2  # Root depth
        self.deathp = 0  # death point for crops
        self.deficit = 0
        self.epot_totgr = 0
        self.epot_totref = self.EpotTotRefTbl[self.crop_type - 1, 1] / 1000  # total reference epot in one year [m]
        self.HoverDEM = 2  # river water level is on average 2 m above surface (because of dikes)
        self.groundwlvl = - 1  # groundwater level is 1 m below surface
        self.pfmax = 4.2
        self.pfred = 2.55
        self.porosity = 0.5  # see potential_additionalinputs to change soiltype. Assumed peat
        self.rin = 1000  # resistance water seepage to river [d]
        self.rout = 1000  # resistance water infiltration from river [d]
        self.soil = 5  # switch to identify soil type. see inputdata_Rhinemodel.xlsx for soil property codes.
        self.soilm = 0.1
        seep = 4.5  # seepage [mm/d]
        self.seep = -1 * seep / 100  # [m/decade]
        self.soilpf = 1  # see potential_additionalinputs to change soiltype. Assumed peat
        # self.survfrac = 1
        self.yielddef = 0
        self.ypot_ref = CropYieldTbl[self.crop_type - 1, 1]  # average crop yield reference (kg ds/ha for maize and grass, bulbs pcs/ha, other kg/ha


        # Inputs added by Veerle
        agriculturemap = self.LandUse == 9  # Veerle Make map with only the agricultural areas
        self.agriculture = pcr2numpy(agriculturemap, -999)  # Turn agricultural map in numpy array
        # urbanmap = self.LandUse == 1  # Veerle Make map with only the agricultural areas
        # self.urban = pcr2numpy(urbanmap, -999)  # Turn agricultural map in numpy array
        # print('urban', np.count_nonzero(self.urban))
        # # print(error)
        self.non_agriculture = ~self.agriculture + 2  # Make a numpy array showing all areas that are not agriculture
        self.survfrac = pcr2numpy(agriculturemap, -999)  # Veerle: make spatial, starts with value of 1
        self.waterlogging_switch = True  # Switch to compare model with and without waterlogging conditions
        self.droughtdays = 0  # Set the number of consequent drought days to 0
        self.soilcracks_switch = True  # Switch to compare model with and wihtout
        self.agri_areas_exposed_count = 0
        self.droughtdays_threshold = 60
        self.storage_reduc = 0.1
        self.germ_stage_depth = 0.02
        self.grow_stage_depth = 0.08
        self.timelag = 2
        self.RP = 0.55
        self.RS = 0.25
        self.SP = 0.3

        # additional initializations
        agri_areas = maptotal(ifthen(self.LandUse == 9, scalar(1)))  # agricultural land [ha]
        self.agri_areas = pcr2numpy(agri_areas, -999)[0, 0]
        self.revenue_ref = self.ypot_ref * self.agri_areas * self.agrprize / 1000000

        if self.groundwlvl > 0:
            self.groundwlvl = np.minimum(self.groundwlvl, 10) * -1

        # inputs_drought_ship_module:
        # switches
        self.dredging_channel = False  # switch to start dredging
        self.multiuse = False # switch for ditches
        self.init_dredging = 0

        # timeseries
        self.ClimShip = pd.read_csv(self.inputfolder + '/discharge_mean_decade_new.tss')
        self.ClimShip = self.ClimShip.values

        # maps
        self.River_bln = boolean(readmap(r"{}".format(self.inputmaps + '/levelsq788.pcr')))
        DEM_riv = pcr_as_numpy(ifthen(self.River_bln == 1, self.DEM))
        self.DEM_riv = DEM_riv[~np.isnan(DEM_riv)]
        
        # tables
        self.ShipTypes = np.loadtxt(self.inputfolder + '/ship_characteristics.txt', skiprows=1, usecols=range(1, 10))

        # parameters
        self.cost_per_tonnage = 6.55  # reference transport costs per ton [EUR/t]
        # self.cost_per_tonnage = 0
        self.dredged_depth = 0
        freight_year = 142.35  # total annual freight volume [Mio t] 142.35
        self.freight_decade = freight_year / 36  # freight transported per 10 days [Mio t]
        self.min_load_capac = 0.4  # minimum required freight of max. freight to make ship economically feasible
        self.multimode_cost_red = 1.0
        self.road_transp = 17  # transport cost per ton full freight volume per decade per [EUR]
        self.shiptype_large = 1
        self.DamShp = 0

        # tables
        self.used_ships = self.ShipTypes[self.shiptype_large - 1, :]

        # inputs_decisionrules:
        self.control_output_string = 0  # to control that "TP reached" is appended just once per scenario
        self.lost_cropland = 0
        self.refperiod = 15
        self.yrs_btw_nmeas = 10

        self.f_u_indicator_past_avg = 80  # expected annual damage on a 25 year shifting average
        self.f_a_indicator_past_avg = 0.5  # expected annual damage on a 25 year shifting average
        self.d_a_indicator_past_avg = 0.5  # expected annual damage on a 25 year shifting average
        self.d_s_indicator_past_avg = 60  # expected annual damage on a 25 year shifting average

        self.f_a_tp_cond = 7
        self.f_u_tp_cond = 150
        self.d_a_tp_cond = 1.1
        self.d_s_tp_cond = 100

        self.f_a_tp_extreme_year = 30
        self.f_u_tp_extreme_year = 200
        self.d_a_tp_extreme_year = 8
        self.d_s_tp_extreme_year = 350

        self.y_lastm_f_a = -11
        self.y_lastm_f_u = -11
        self.y_lastm_d_a = -11
        self.y_lastm_d_s = -11

        all_mportfolios_f_a = pd.read_csv(f'{self.inputfolder}/stage{stage}_portfolios_flood_agr.txt',
                                          names=['1', '2', '3', '4'], dtype='str')
        all_mportfolios_f_u = pd.read_csv(f'{self.inputfolder}/stage{stage}_portfolios_flood_urb.txt',
                                          names=['1', '2', '3', '4'], dtype='str')
        all_mportfolios_d_a = pd.read_csv(f'{self.inputfolder}/stage{stage}_portfolios_drought_agr.txt',
                                          names=['1', '2', '3', '4'], dtype='str')
        all_mportfolios_d_s = pd.read_csv(f'{self.inputfolder}/stage{stage}_portfolios_drought_shp.txt',
                                          names=['1', '2', '3', '4'], dtype='str')
        self.all_mportfolios_f_a = all_mportfolios_f_a.values
        self.all_mportfolios_f_u = all_mportfolios_f_u.values
        self.all_mportfolios_d_a = all_mportfolios_d_a.values
        self.all_mportfolios_d_s = all_mportfolios_d_s.values

        self.pathways_list_f_a = '0'
        self.pathways_list_f_u = '0'
        self.pathways_list_d_a = '0'
        self.pathways_list_d_s = '0'

        self.measure_numbers = {
            'no_measure': 0,
            'd_resilient_crops': 1,
            'd_rain_irrigation': 2,
            'd_gw_irrigation': 3,
            'd_riv_irrigation': 4,
            'd_soilm_practice': 5,
            'd_multimodal_transport': 6,
            'd_medium_ships': 7,
            'd_small_ships': 8,
            'd_dredging': 9,
            'f_resilient_crops': 10,
            'f_ditches': 11,
            'f_local_support': 12,
            'f_dike_elevation_s': 13,
            'f_dike_elevation_l': 14,
            'f_maintenance': 15,
            'f_room_for_river': 16,
            'f_wet_proofing_houses': 17,
            'f_local_protect': 18,
            'f_awareness_campaign': 19
        }


        # inputs_toolbox(self):
        # switches
        self.floodresil_m = True
        self.f_ditch_m = True
        self.floodprone_m = True
        self.localprotect_m = True
        self.smalldike_m = True
        self.largedike_m = True
        self.noconstruction_m = True

        self.d_rainwater_tank_m = True  # switch to initiate collecting of rainwater
        self.d_groundwater_irrigation_m = True  # switch to initiate usage of groundwater pumps
        self.d_river_irrigation_m = True  # switch to initiate usage of river water
        self.collectwater_m = True
        self.dike_irrigation_m = True

        self.dredging_channel_m = True  # switch to start dredging

        # Tables

        self.DamfactTbl_floodresil_floodprone_localprotect = r"{}".format(
            self.inputfolder + '/measures/damfact_fres_floodprone_locprotect.tbl')
        self.DamfactTbl_floodresil_localprotect = r"{}".format(
            self.inputfolder + '/measures/damfact_fres_locprotect.tbl')
        self.DamfactTbl_floodprone_localprotect = r"{}".format(
            self.inputfolder + '/measures/damfact_floodprone_locprotect.tbl')
        self.DamfactTbl_localprotect = r"{}".format(self.inputfolder + '/measures/damfact_locprotect.tbl')
        self.DamfactTbl_floodresil_floodprone = r"{}".format(self.inputfolder + '/measures/damfact_fres_floodprone.tbl')
        self.DamfactTbl_floodprone = r"{}".format(self.inputfolder + '/measures/damfact_floodprone.tbl')
        self.Damfact_floodresil = r"{}".format(self.inputfolder + '/measures/damfact_fres.tbl')


        self.DamfactTbl_droughtresil_floodprone_localprotect = r"{}".format(
            self.inputfolder + '/measures/damfact_dres_floodprone_locprotect.tbl')
        self.DamfactTbl_droughtresil_localprotect = r"{}".format(
            self.inputfolder + '/measures/damfact_dres_locprotect.tbl')
        self.DamfactTbl_floodprone_localprotect = r"{}".format(
            self.inputfolder + '/measures/damfact_floodprone_locprotect.tbl')
        self.DamfactTbl_localprotect = r"{}".format(self.inputfolder + '/measures/damfact_locprotect.tbl')
        self.DamfactTbl_droughtresil_floodprone = r"{}".format(self.inputfolder + '/measures/damfact_dres_floodprone.tbl')
        self.DamfactTbl_floodprone = r"{}".format(self.inputfolder + '/measures/damfact_floodprone.tbl')
        self.Damfact_droughtresil = r"{}".format(self.inputfolder + '/measures/damfact_dres.tbl')

        self.ReducPointTbl_m = np.loadtxt(self.inputfolder + '/ReducPoint.tbl', skiprows=1, delimiter=',')
        self.ReducPointTbl_m[:, self.crop_type] = self.ReducPointTbl_m[:,self.crop_type] * 0.7

        # parameters
        # flooding
        self.dike_sprinkling_need = 1.5 * self.soilm_dike  # 150% of max soilmoisture to compensate for evaporation and irrigation loss [
        self.spot_chance = 0.5
        self.drought_thr = 0.1  # thr for when drought might lead to increased vulnerability to water induced damages
        self.Exponent = 15       # n to the power of the value to calculate reduction effect of dike stability
        self.dike_cons_thr_rain = 0.04  # thr for when rain cause extra vulnerability
        self.consec_time = 3  # amount of time after drought ended impact of drought still there
        self.dike_cons_thr_discharge = 12000  # thr for when discharge cause extra vulnerability
        self.dikedrought_cons_factor = 0.5  # consec factor (change in dikefrag due to consec hazard)
        self.insp_repair = 5
        self.rainfall_deficit = 0
        self.rainfall_deficit_thr = 0.15  # [m]
        self.sprink_dikes_tot = 0
        self.spot_chance = 0.5
        self.f_ditch_red_m = 0.2
        self.dike_increase_s = 0.5
        self.dike_increase_l = 1.0
        self.dike_increase_multisector = 2
        self.dike_weight_increase_s = 0.2  # change of relative dike weight (e.g. 1.2)
        self.dike_weight_increase_l = 0.4
        self.rout_rfr = 950  # resistance water infiltration from river [d]
        self.RfR_m = 3
        self.critQ = 10000
        self.ditchvolume_irrigation = 10000
        self.ditch_multiuse = False # default ditch is not used for irrigation

        self.Fact788_m = self.RfR_factors.iloc[0][self.RfR_m]
        self.Fact7150_m = self.RfR_factors.iloc[1][self.RfR_m]
        self.Fact16000_m = self.RfR_factors.iloc[2][self.RfR_m]
        self.Fact20000_m = self.RfR_factors.iloc[3][self.RfR_m]

        self.Dif_2_m = (self.Levq7150 * self.Fact7150_m) - (self.Levq788 * self.Fact788_m)
        self.Dif_6_m = (self.Levq16000 * self.Fact16000_m) - (self.Levq7150 * self.Fact7150_m)
        self.Dif_7_m = (self.Levq20000 * self.Fact20000_m) - (self.Levq16000 * self.Fact16000_m)

        # drought farmers
        self.crop_vul_m = 0.1
        self.collectwater_rain = 0  # collected water at beginning of scenario [m3]
        self.collectwater_gw = 0
        self.collectwater_riv = 0
        self.collectwater_ditch = 0
        self.irrigation_effectiveness = 0.9  # share of effective irrigation water. Rest goes into soilm
        self.gw_res_area = 1000 * 100 * 100  # groundwater reservoir size of 1000 ha
        self.gw_max_extr_rate = 0.02  # [m/(m2 *decade)] estimated based on https://www.oecd.org/greengrowth/sustainable-agriculture/groundwater-country-note-NLD-2015%20final.pdf
        self.gw_volume = self.gw_res_area * self.gw_max_extr_rate  # available water volume for sprinkling [m3/decade]
        self.Vpump = 1.0 * 60 * 60 * 24 * 10  # maximum pumping capacity from river [m3/decade] total pumping capacity 20m3/s
        self.sprinkfrac = 0.8  # grid cell area that requires irrigation. Assumed constant 80%
        self.tanksize = 4000  # [m3]

        # drought shipping
        self.change_years = [5, 10, 15]
        self.dredged_depths = [1.2, 0.8, 0.4, 0]
        self.multimode_cost_red_m = 0.85
        self.Qmin = 700  # minimum required discharge for ecological reasons
        self.Q_accum_thr = 1000*36*5  # notused; threshold for accumulated discharge over 5 years as an indicator of sedimentation
        self.shiptype_medium = 2
        self.shiptype_small = 3
