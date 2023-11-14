from waasmodel_v6.helperfunctions import urban_growth
from pcraster import *
import numpy as np


class MeasureToolbox:
    __slots__ = ('stage')
    def __init__(self,  stage):
        self.stage = stage

    def no_measure(self, module_transfer, inputs):
        return module_transfer,inputs.measure_numbers['no_measure']

    def f_resilient_crops(self, module_transfer, inputs):
        '''
        uses an adjusted version of the damage curve for flooding. Linear reduction by 10%-points.
        '''
        module_transfer.floodresil = inputs.floodresil_m
        module_transfer.DamFactTbl[:10,2] = module_transfer.DamFactTbl[:10,2] * 0.9

        # if module_transfer.pathways_list_d_a.find(str(inputs.measure_numbers['d_resilient_crops'])) != -1:
        #     module_transfer.pathways_list_d_a.replace(str(inputs.measure_numbers['d_resilient_crops']), '')
        #     module_transfer.pathways_list_d_a.replace('&&', '&')
        # else:
        #     "no drought resilient crop implemented that conflicts with implementing f_res_crops."
        module_transfer.ReducPointTbl = inputs.ReducPointTbl

        module_transfer.crop_vul = inputs.crop_vul_m
        module_transfer.d_resilient_crop_switch = False
        return module_transfer, inputs.measure_numbers['f_resilient_crops']

    def f_ditches(self, module_transfer, inputs):
        '''
        Water storage reduces agricultural land by 10 % but therefore reduces inundation by 10 cm's (on agricultural
        lands).
        '''
        if module_transfer.f_ditch == False:
            agri_area = maptotal(ifthen(module_transfer.LandUse == 9, scalar(1)))
            agri_area_np = pcr2numpy(agri_area, -999)
            # lossed area is 10% of total agricultural area --> number of cells (ha*ha) that need to be replaced need division by 100
            ditch_area = np.round(agri_area_np[0, 0] * 0.1 / 100, 0)
            # map with cell distances around all non-agricultural areas (resolution 100 m)
            Spr_nonagri = spread(ifthenelse(module_transfer.LandUse != ordinal(9), ordinal(1), ordinal(0)), 0, 1)
            # random cells next in direct proximity of non-agricultural areas used for urban growth
            Spr_nonagri2 = ifthen(pcrand(module_transfer.LandUse == ordinal(9),
                                         pcrand(Spr_nonagri > 0, Spr_nonagri <= max(scalar(ditch_area), scalar(300)))),
                                  boolean(1))
            # new areas representative of areas lost due to ditches
            Spr_nonagri3 = ifthen(order(uniform(Spr_nonagri2)) <= ditch_area, scalar(10))
            # updated land use map where lu class is overwritten where new ditch areas exist.
            NewLU = cover(ordinal(Spr_nonagri3), module_transfer.LandUse)

            module_transfer.LandUse = NewLU
            module_transfer.expos = lookupscalar(module_transfer.MaxDamTbl, NewLU)
            # derive total agricultural area in the Waasregion
            agri_areas = maptotal(ifthen(NewLU == 9, scalar(1)))  # in [ha]!
            module_transfer.agri_areas = pcr2numpy(agri_areas, -999)[0, 0]

            module_transfer.f_ditch = inputs.f_ditch_m
            module_transfer.f_ditch_red = inputs.f_ditch_red_m

            # lossed land
            module_transfer.lost_cropland += ditch_area * 100

            # ditches increase groundwater recharge
            module_transfer.rin -= 50

            agriculturemap = module_transfer.LandUse == 9  # Veerle Make map with only the agricultural areas
            module_transfer.agriculture = pcr2numpy(agriculturemap, -999)  # Turn agricultural map in numpy array

        return module_transfer, inputs.measure_numbers['f_ditches']

    def f_dike_elevation_s(self, module_transfer, inputs):
        '''
        small dike increase elevation by 0.5 meter. Chance of dike failure reduces.
        '''
        NewDEM = ifthenelse(module_transfer.PrimDike > 0,
                            module_transfer.DEM + inputs.dike_increase_s, module_transfer.DEM)
        module_transfer.DEM = NewDEM
        module_transfer.smalldike = inputs.smalldike_m  # relevant to increase costs for room for the river

        module_transfer.dike_weight_rel += inputs.dike_weight_increase_s  # added weight relative to the original dike

        return module_transfer,inputs.measure_numbers['f_dike_elevation_s']

    def f_dike_elevation_l(self, module_transfer, inputs):
        '''
        large dike elevation increase by 1.0 meter. Chance of dike failure reduces.
        '''
        NewDEM = ifthenelse(module_transfer.PrimDike > 0,
                            module_transfer.DEM + inputs.dike_increase_l, module_transfer.DEM)
        module_transfer.DEM = NewDEM
        module_transfer.largedike = inputs.largedike_m  # relevant to increase costs for room for the river

        module_transfer.dike_weight_rel += inputs.dike_weight_increase_l  # added weight relative to the original dike

        return module_transfer,inputs.measure_numbers['f_dike_elevation_l']

    def f_room_for_river(self, module_transfer, inputs):
        '''
            Adjust the land use classes and DEM in case RfR is installed. RfR overwrites initial LU classes. Primary
            dikes are shifted towards North/South respectively. Part of initial dike with nature class (lu=8) remains
            in nature class. For DEM, shifting existing dike height to North/South along with land use classe. For
            additional space in flood plain, reproduce same DEM information as in X neighboring rows.

            Current code is run once per scenario run and assumes that all measures are installed in the first year.
            Conflicts in space in later years due to advancing development cannot be simulated.
        '''
        if self.stage > 1:
            module_transfer.rout = inputs.rout_rfr  # reduces resistance to infiltrate from the river.
            module_transfer.rin -= 50   # reduces resistance to infiltrate from the river

        module_transfer.Fact788 = inputs.RfR_factors.iloc[0][inputs.RfR_m]
        module_transfer.Fact7150 = inputs.RfR_factors.iloc[1][inputs.RfR_m]
        module_transfer.Fact16000 = inputs.RfR_factors.iloc[2][inputs.RfR_m]
        module_transfer.Fact20000 = inputs.RfR_factors.iloc[3][inputs.RfR_m]

        module_transfer.Dif_2 = (inputs.Levq7150 * module_transfer.Fact7150) - (
                    inputs.Levq788 * module_transfer.Fact788)
        module_transfer.Dif_6 = (inputs.Levq16000 * module_transfer.Fact16000) - (
                    inputs.Levq7150 * module_transfer.Fact7150)
        module_transfer.Dif_7 = (inputs.Levq20000 * module_transfer.Fact20000) - (
                    inputs.Levq16000 * module_transfer.Fact16000)

        move_distance_RfR = int(inputs.RfR_factors.iloc[4][
                                    inputs.RfR_m] / 100)  # grid size is 100 m, here we get how many cells the dike moves
        PrimD_np = pcr_as_numpy(module_transfer.DEM)
        # shift Northern dike northwards and replace DEM at old dike location with riverbed elevation
        PrimD_np[[23 - move_distance_RfR, 23], :] = PrimD_np[[23, 23 - move_distance_RfR], :]
        PrimD_np[23, :] = PrimD_np[24, :]
        # shift Southern dike southwards and replace DEM at old dike location with riverbed elevation
        PrimD_np[[33, 33 + move_distance_RfR], :] = PrimD_np[[33 + move_distance_RfR, 33], :]
        PrimD_np[33, :] = PrimD_np[32, :]
        # convert back to pcr
        module_transfer.DEM = numpy2pcr(Scalar, PrimD_np, -999)
        # create mask to update land use
        mask_lu = np.ones_like(PrimD_np)  # rest of Waasregion
        mask_lu[23 - move_distance_RfR + 1:33 + move_distance_RfR, :] = 2  # river
        mask_temp = numpy2pcr(Nominal, mask_lu, 0)
        DikeRing_new = ifthen(mask_temp == 1, module_transfer.DikeRing)  # nominal, floodplain not included
        mask_lu[[23 - move_distance_RfR, 33 + move_distance_RfR], :] = 3  # dike locations
        mask_lu_pcr = numpy2pcr(Ordinal, mask_lu, 0)
        # Change land use within new embankments to river
        if self.stage == 3:  # multi-sector effect of measures
            lost_agri_land = pcr2numpy(maptotal(
                ifthenelse(pcrand(module_transfer.LandUse == ordinal(9), mask_lu_pcr >= ordinal(2)), scalar(1),
                           scalar(0))), -999)[0, 0]

        else:
            lost_agri_land = 0
        newLU = ifthenelse(mask_lu_pcr == 2, ordinal(14), module_transfer.LandUse)
        # Change land use at new dike locations to dike
        newLU = ifthenelse(mask_lu_pcr == 3, ordinal(18), newLU)
        # investigate loss of urban area
        urb_loss = ifthenelse(pcrand(mask_lu_pcr == 2, module_transfer.LandUse == 1), scalar(1), scalar(0))
        urb_loss_tot = maptotal(urb_loss)
        urb_loss_val = pcr2numpy(urb_loss_tot, -999)[0][0]
        # update all relevant parameters
        # move lossed urban areas somewhere else
        if urb_loss_val > 0:
            module_transfer.LandUse, nu_agri_areas = urban_growth(growth=int(urb_loss_val), LUmap=newLU)
        else:
            module_transfer.LandUse = newLU
        new_PrimDike = ifthenelse(mask_lu_pcr == 3, nominal(module_transfer.DikeRing),
                                  nominal(0))  # 1,2,3: North dike; 4,5: South Dike along river
        module_transfer.PrimDike = scalar(new_PrimDike)
        module_transfer.DikeRing = DikeRing_new
        module_transfer.expos = lookupscalar(module_transfer.MaxDamTbl, module_transfer.LandUse)
        if self.stage == 3:
            module_transfer.agri_areas -= (urb_loss_val + lost_agri_land)
            agriculturemap = module_transfer.LandUse == 9  # Veerle Make map with only the agricultural areas
            module_transfer.agriculture = pcr2numpy(agriculturemap, -999)  # Turn agricultural map in numpy array

        # lost land
        module_transfer.lost_cropland += lost_agri_land + urb_loss_val
        module_transfer.lost_cropland = module_transfer.lost_cropland
        return module_transfer,inputs.measure_numbers['f_room_for_river']

    def f_wet_proofing_houses(self, module_transfer, inputs):
        module_transfer.floodprone = inputs.floodprone_m
        module_transfer.DamFactTbl[:, 1] = module_transfer.DamFactTbl[:, 1] * 0.9

        return module_transfer,inputs.measure_numbers['f_wet_proofing_houses']

    def f_local_protect(self, module_transfer, inputs):
        module_transfer.localprotect = inputs.localprotect_m
        module_transfer.DamFactTbl[:8, 1] = 0

        return module_transfer ,inputs.measure_numbers['f_local_protect']

    def d_resilient_crops(self, module_transfer, inputs):
        '''uses an adjusted version of the crop factor for drought. Linear reduction by 10%-points.
        It affects e_pot which again determines the evaporation ratio and thus potential crop damage.
        '''

        module_transfer.floodresil = False
        module_transfer.d_resilient_crop_switch = True
        module_transfer.DamFactTbl[:10,2] = inputs.DamFactTbl[:10,2] * 1.1

        module_transfer.ReducPointTbl = inputs.ReducPointTbl_m
        return module_transfer, inputs.measure_numbers['d_resilient_crops']

    def d_rain_irrigation(self, module_transfer, inputs):
        '''Collects water that falls on area of 2% of agricultural land. (part of) tank volume is used for sprinkling (changin soilm) reduce. Effectiveness of irrigation is 90%. Updates eact and soilm'''
        module_transfer.d_rainwater_tank = True
        module_transfer.collectwater_switch = True

        return module_transfer, inputs.measure_numbers['d_rain_irrigation']

    def d_gw_irrigation(self, module_transfer, inputs):
        module_transfer.d_groundwater_irrigation = inputs.d_groundwater_irrigation_m
        module_transfer.collectwater_switch = inputs.collectwater_m
        return module_transfer,inputs.measure_numbers['d_gw_irrigation']

    def d_riv_irrigation(self, module_transfer, inputs):
        module_transfer.d_river_irrigation = inputs.d_river_irrigation_m
        module_transfer.collectwater_switch = inputs.collectwater_m

        return module_transfer,inputs.measure_numbers['d_riv_irrigation']

    def d_multimodal_transport(self, module_transfer, inputs):
        '''
        Reduces the extra cost of transport in case shipping on the river is not possible by 40%
        '''
        module_transfer.multimode_cost_red = inputs.multimode_cost_red_m
        return module_transfer,inputs.measure_numbers['d_multimodal_transport']

    def d_medium_ships(self, module_transfer, inputs):
        module_transfer.used_ships = inputs.ShipTypes[inputs.shiptype_medium - 1, :]
        return module_transfer,inputs.measure_numbers['d_medium_ships']

    def d_small_ships(self, module_transfer, inputs):
        module_transfer.used_ships = inputs.ShipTypes[inputs.shiptype_small - 1, :]

        return module_transfer,inputs.measure_numbers['d_small_ships']

    def d_dredging(self, module_transfer, inputs):
        module_transfer.dredging_channel = inputs.dredging_channel_m
        module_transfer.init_dredging = module_transfer.timestep

        return module_transfer,inputs.measure_numbers['d_dredging']

    def f_awareness_campaign(self, module_transfer, inputs):
        module_transfer.dyn_vulnerability_urb_factors = np.array([1.2, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.95, 0.95,
                                                                  0.97])
        module_transfer.riskaware_period = 10

        return module_transfer, inputs.measure_numbers['f_awareness_campaign']

    def f_maintenance(self, module_transfer, inputs):
        # dike sprinkling & irrigation
        module_transfer.dike_maintenance = True

        return module_transfer, inputs.measure_numbers['f_maintenance']

    def f_local_support(self, module_transfer, inputs):
        module_transfer.revenue_reduc_flood = 0.9
        return module_transfer, inputs.measure_numbers['f_local_support']

    def d_soilm_practice(self, module_transfer, inputs):
        module_transfer.soilm_max = inputs.soilm * 1.25
        return module_transfer, inputs.measure_numbers['d_soilm_practice']




