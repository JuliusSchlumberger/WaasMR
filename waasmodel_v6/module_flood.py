import numpy as np
from pcraster import *
import random
from waasmodel_v6.helperfunctions import Qhrelation
from waasmodel_v6.module_drought_agr import DAgrModule
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class FloodModule:
    __slots__ = ('analysis', 'test_all', 'stage', 'last_flood', 'soilm_dike', 'groundwlvl_dike', 'rainfall_deficit',
                 'dyn_vulnerability_agr_fact', 'sprink_dikes_tot', 'survfrac', 'num_dikefails', 'damfrac_tot_f',
                 'DamUrb_tot', 'DamAgr_f_tot','DamAgr_f', 'num_dikefails_tot', 'dyn_vulnerability_urb_fact','droughtperiod',
                 'dikedrought_time', 'higher_consecutive','droughtperiod_end')

    #@profile
    def __init__(self, analysis, test_all, stage, inputs):
        self.analysis = analysis
        self.test_all = test_all
        self.stage = stage

        # initialize special values
        self.last_flood = inputs.last_flood
        self.soilm_dike = inputs.soilm_dike
        self.groundwlvl_dike = inputs.groundwlvl_dike
        self.rainfall_deficit = inputs.rainfall_deficit
        self.dyn_vulnerability_agr_fact = inputs.dyn_vulnerability_agr_fact
        self.sprink_dikes_tot = inputs.sprink_dikes_tot
        self.num_dikefails = inputs.num_dikefails
        self.num_dikefails_tot = inputs.num_dikefails
        self.DamUrb_tot = 0
        self.DamAgr_f_tot = 0
        self.DamAgr_f = 0
        self.damfrac_tot_f = 0
        self.dyn_vulnerability_urb_fact = 1
        self.droughtperiod = False  # switch on
        self.dikedrought_time = 0  # reset dought effect timer
        self.higher_consecutive = False  # strong droughts case increased risk from conscec hazard
        self.droughtperiod_end = False  # switch on drought ended


    #@profile
    def run_module(self, timestep, scenario, module_transfer, inputs):

        Q, dike_fail_ini, year, timemod, remyield, evapo_ref, precip = self.initiate_timeinputs(timestep=timestep, scenario=scenario, inputs=inputs)
        self.reset_values_newyear(timemod=timemod, module_transfer=module_transfer, inputs=inputs)
        self.calculate_constant_revenue_reduction(inputs=inputs, module_transfer=module_transfer)

        if self.stage >= 2:
            dike_fail_ini, Qriv_reduc = self.drought2flood(timemod=timemod, module_transfer=module_transfer, Q=Q, dike_fail_ini=dike_fail_ini, inputs=inputs, evapo_ref=evapo_ref, precip=precip)
        else:
            Qriv_reduc = 0

        if (Q < 7500 and dike_fail_ini > 0.05 and self.test_all==False):
            wlvl = 0
            fragdike = 0
            floodtiming = 0
            DamUrb = 0
            self.DamAgr_f = 0
            self.num_dikefails = 0
        else:

            max_lvl, wlvl = self.calculate_waterlevels(Q=Q, module_transfer=module_transfer, inputs=inputs)
            FragDike, fragdike, Surf, TmpFragD = self.determine_dike_fragility(max_lvl=max_lvl,
                                                                               module_transfer=module_transfer,
                                                                               inputs=inputs)
            Dike_breach, flood_occured = self.check_if_flooding(dike_fail_ini=dike_fail_ini, FragDike=FragDike,
                                                                inputs=inputs)

            if flood_occured == 1:  # flood occured
                floodtiming = timemod
                LevelFlood = self.calculate_inundation_depth(Dike_breach=Dike_breach, Surf=Surf, TmpFragD=TmpFragD,
                                                             module_transfer=module_transfer)
                self.DamAgr_f, DamUrb = self.calculate_flood_impacts(LevelFlood=LevelFlood, module_transfer=module_transfer,
                                                                remyield=remyield, timestep=timestep, inputs=inputs)
            else:
                wlvl = 0
                fragdike = 0
                floodtiming = 0
                DamUrb = 0
                self.DamAgr_f = 0
                self.num_dikefails = 0


        self.create_output_dicts(DamAgr_f=self.DamAgr_f, wlvl=wlvl, fragdike=fragdike, floodtiming=floodtiming, Qriv_reduc=Qriv_reduc,
                                 DamUrb=DamUrb, module_transfer=module_transfer, Q=Q)
        return module_transfer

    #@profile
    def calculate_constant_revenue_reduction(self, inputs, module_transfer):
        rev_loss_lost_land = (inputs.ypot_ref * module_transfer.lost_cropland * inputs.agrprize) / 1000000
        if self.stage == 1:
            module_transfer.revenue_agr = np.maximum(np.sum(inputs.ypot_ref * module_transfer.agriculture) * inputs.agrprize / 1000000- rev_loss_lost_land, 0)

    #@profile
    def initiate_timeinputs(self, timestep, scenario, inputs):
        timemod = timestep % 36
        year = timestep / 36

        dike_fail_ini = inputs.randomR[timestep, 1]
        Q = inputs.ClimQmax[timestep, scenario - 1]
        evapo_ref = inputs.evapo_ref_scenarios[timestep, scenario - 1]
        precip = inputs.precip_scenarios[timestep, scenario - 1]
        remyield = np.maximum(0.02, inputs.RemYieldTbl[timemod, inputs.crop_type])

        return Q, dike_fail_ini, year, timemod, remyield, evapo_ref, precip

    #@profile
    def reset_values_newyear(self, timemod, module_transfer, inputs):
        if timemod == 0:
            self.sprink_dikes_tot = 0
            self.damfrac_tot_f = 0
            self.DamUrb_tot = 0
            self.DamAgr_f_tot = 0
            self.DamAgr_f = 0
            self.num_dikefails = 0
            self.num_dikefails_tot = 0
            module_transfer.survfrac = inputs.agriculture
            module_transfer.revenue_reduc = 1
            module_transfer.revenue_reduc = module_transfer.revenue_reduc_noflood
            self.rainfall_deficit = 0
            module_transfer.revenue_agr = inputs.revenue_ref

            pass

    #@profile
    def calculate_waterlevels(self, Q, module_transfer, inputs):
        # Calculate water level
        max_lvl = Qhrelation(Q=Q, Levq788=inputs.Levq788,
                             Fact788=module_transfer.Fact788, Levq7150=inputs.Levq7150,
                             Fact7150=module_transfer.Fact7150, Levq16000=inputs.Levq16000,
                             Fact16000=module_transfer.Fact16000, Levq20000=inputs.Levq20000,
                             Fact20000=module_transfer.Fact20000, Dif_2=module_transfer.Dif_2,
                             Dif_6=module_transfer.Dif_6, Dif_7=module_transfer.Dif_7)
        wlvl = np.max(max_lvl)
        return max_lvl, wlvl

    #@profile
    def determine_dike_fragility(self, max_lvl, module_transfer, inputs):

        # Compute dike fragility based on water level
        Levcov = pcr2numpy(inputs.Case + 1, 0)
        Levcov_new = max_lvl * np.ones_like(Levcov)
        Surf = numpy2pcr(Scalar, Levcov_new, -99)
        TmpFragD = ifthen(module_transfer.PrimDike > 0,
                          Surf - (module_transfer.DEM - 0.5))  # Water level relative to primary dike elevation
        # Derive where dike failure occurs along primary dikes
        FragDike = areamaximum((lookupscalar(inputs.FragTbl, TmpFragD)), module_transfer.DikeRing)  # for dike rings
        # plot(FragDike)
        fragdike = pcr2numpy(mapmaximum(FragDike), -999)[0, 0]

        return FragDike, fragdike, Surf, TmpFragD

    #@profile
    def check_if_flooding(self, dike_fail_ini, FragDike, inputs):
        dike_fail = inputs.Case + dike_fail_ini  # fixed random
        Dike_breach = ifthenelse(dike_fail < FragDike, scalar(1), scalar(0))
        flood_occured = pcr2numpy(mapmaximum(Dike_breach), 0)[0, 0]
        return Dike_breach, flood_occured

    #@profile
    def calculate_inundation_depth(self, Dike_breach, Surf, TmpFragD, module_transfer):
        # water level at dike element of breach
        WaterlevelFlood = ifthen(Dike_breach == 1, Surf)
        Surf2 = areamaximum(WaterlevelFlood, module_transfer.DikeRing)
        # Get bathtub inundation in case of dike breach
        LevelDB = ifthenelse(Dike_breach == 1, max(0, scalar(0.3) * Surf2 - module_transfer.DEM), 0)
        # Check inundation in case of no dike failure due to overtopping
        OT_rings = areamaximum(TmpFragD, module_transfer.DikeRing)
        LevelOT = ifthenelse(pcrand(Dike_breach == 0, OT_rings > 0), max(0, 0.1 * Surf - module_transfer.DEM), 0)
        # Use flood level in dike rings
        LevelFlood = LevelOT + LevelDB

        LevelFlood = ifthenelse(module_transfer.LandUse == 9, LevelFlood - module_transfer.f_ditch_red,
                                LevelFlood)  # effect of measure f_ditches on water levels


        for ring in range(2, 6):    # count number of dike rings flooded in the event
            Ring = ifthen(nominal(module_transfer.DikeRing) == nominal(ring), nominal(1))
            count_fail = mapmaximum(areamaximum(Dike_breach, Ring))
            self.num_dikefails += pcr2numpy(count_fail, 0)[0, 0]
        self.num_dikefails_tot += self.num_dikefails

        return LevelFlood

    #@profile
    def calculate_flood_impacts(self, LevelFlood, module_transfer, remyield, timestep, inputs):
        # Compute damage factor based on existing flood level and land use
        lookup_table = module_transfer.DamFactTbl
        LevelFlood_np = pcr2numpy(LevelFlood, 0)
        DamFunct = ifthenelse(module_transfer.LandUse == ordinal(1), nominal(1),
                              ifthenelse(module_transfer.LandUse == ordinal(9), nominal(2), nominal(0)))
        DamFunct_np = pcr2numpy(DamFunct, 0)

        # Function to find the nearest value in the lookup table
        def find_nearest_value(value, lu_class):
            closest_value = lookup_table[np.abs(lookup_table[:, 0] - value).argmin(), lu_class]
            return closest_value

        # Apply the function element-wise to the values ndarray
        DamFact = np.vectorize(find_nearest_value)(LevelFlood_np, DamFunct_np)

        # Interaction effects (mostly stage 3)
        self.interaction_effects(timestep=timestep, last_flood=self.last_flood,inputs=inputs, module_transfer=module_transfer)

        revenue_loss, DamAgr_f, DamUrb_f, module_transfer.survfrac = self.calculate_damage(DamFact=DamFact,
                                                                                           module_transfer=module_transfer,
                                                                                           remyield=remyield,
                                                                                           inputs=inputs)

        self.DamAgr_f_tot += DamAgr_f
        self.DamAgr_f += DamAgr_f
        module_transfer.revenue_agr = np.maximum(module_transfer.revenue_agr - revenue_loss, 0)

        # self.flood2drought(Q=self.Q)  # based on flooding, change of soil moisture values

        self.DamUrb_tot += DamUrb_f
        self.last_flood = timestep  # safe timing of this flood

        return self.DamAgr_f, DamUrb_f

    #@profile
    def interaction_effects(self, timestep, last_flood, inputs, module_transfer):
        '''This function is only called when flooding occured and thus damage is likely. It therefore only has
        to update parameters. '''
        # multi-hazard interaction
        # update vulnerability factor for consecutive flood-flood events
        self.consecutive_floods_urban(timestep=timestep, last_flood=last_flood, inputs=inputs, module_transfer=module_transfer)
        #  multi-sector interaction (agr. losses due to flood in urban sector)
        
        if self.stage == 3: # this function is only called when flood occurs. 
            module_transfer.revenue_reduc = module_transfer.revenue_reduc_flood

    #@profile
    def calculate_damage(self, DamFact, module_transfer, remyield, inputs):
        '''
        Updates the survival fraction at t (used as an input in the crop module as well) and uses damfrac_t to calculate the damage of flooding
        :param DamFact:
        :return:
        '''
        # Update Damfact

        # crop exposure
        expos_crop_max = inputs.ypot_ref * module_transfer.agriculture * inputs.agrprize / 1000000
        # expos_crop_now = expos_crop_max * module_transfer.survfrac * remyield
        expos_crop_now = expos_crop_max * module_transfer.survfrac

        expos_agr_structures = pcr2numpy(module_transfer.expos, 0) * module_transfer.agriculture - expos_crop_max
        expos_urb_structures = pcr2numpy(module_transfer.expos, 0) * pcr2numpy(module_transfer.LandUse == 1, 0)

        revenue_loss = np.sum(expos_crop_now.flatten() * DamFact.flatten())
        # print(expos_crop_max)
        # import matplotlib.pyplot as plt
        # plt.imshow(expos_crop_now *DamFact, cmap='hot', interpolation='nearest')
        # plt.show()
        DamAgr_t = np.sum(expos_agr_structures * DamFact) + revenue_loss
        # print(DamAgr_t, revenue_loss, DamFact, expos_crop_now)
        DamUrb_t = np.sum(expos_urb_structures * np.clip(self.dyn_vulnerability_urb_fact * DamFact,
                              a_min=None, a_max=1))

        survfrac_f = module_transfer.agriculture * (1 - module_transfer.survfrac * remyield * DamFact)

        return revenue_loss, DamAgr_t, DamUrb_t, survfrac_f  #, expos_new

    #@profile
    def consecutive_floods_urban(self, timestep, last_flood, inputs, module_transfer):
        
        '''
        Calculating dynamic vulnerability & exposure factor for consecutive flood events. After five years, the 
        default vulnerability & exposure apply (=1), if flood occurs during the recovery/awareness period, the
        flood impact is different (updated on previous timestep) and updated for the next timestep again.'''
        # time-dependent effects urban
        if timestep == 0 or timestep > last_flood + 5 * 36:
            self.dyn_vulnerability_urb_fact = inputs.dyn_vulnerability_urb_fact
        else:
            for recover_year in range(module_transfer.riskaware_period):
                if timestep == last_flood + 36 * recover_year:
                    self.dyn_vulnerability_urb_fact = self.dyn_vulnerability_urb_fact * module_transfer.dyn_vulnerability_urb_factors[recover_year]
                else:
                    self.dyn_vulnerability_urb_fact = self.dyn_vulnerability_urb_fact * module_transfer.dyn_vulnerability_urb_factors[0]
        self.dyn_vulnerability_urb_fact = np.minimum(self.dyn_vulnerability_urb_fact, 1.3)
    #@profile
    def drought2flood(self, timemod, module_transfer, Q, dike_fail_ini, inputs, evapo_ref, precip):
        '''Change of exposure charactersitics for crops (longer exposed to flooding) in case of high/low soilm.
           Change of dike fragility in case dike soilm gets below a threshold.'''

        # Irrigation of dikes (in case of measures implemented)
        cropfact = inputs.CropFactTbl_dike[timemod]
        epot = evapo_ref * cropfact * 1.2  # 1.2 is correction factor to align with more recent version of the complex model (NHI)
        self.rainfall_deficit += epot - precip

        if module_transfer.dike_maintenance and self.rainfall_deficit >= inputs.rainfall_deficit_thr and timemod >= 13 and timemod <= 30:  # Lars condition
            # irrigation
            collectwater_dikes_need = (inputs.dike_sprinkling_need / 1000) * (inputs.dike_areas * 10000) / (
                        10 * 24 * 60 * 60) * 2  # [m3/decade] effectiveness of pumping very low (from ship)
            if Q - collectwater_dikes_need < inputs.Qmin:
                collectwater_dikes_avail = np.maximum(0, Q - inputs.Qmin) * (10 * 24 * 60 * 60) / (
                            inputs.dike_areas * 10000)  # [m/decade]
                Qriv_reduc = collectwater_dikes_avail / (24 * 60 * 60) # [m3/s] water is taken within a day and spread over 10 day
            else:
                collectwater_dikes_avail = inputs.dike_sprinkling_need * (inputs.dike_areas * 10000) / (
                        10 * 24 * 60 * 60) # [m/decade]
                Qriv_reduc = collectwater_dikes_need / (24 * 60 * 60)  # [m3/s] water is taken within a day and spread over 10 days
            sprink_dikes = collectwater_dikes_avail  # [mm/decade]

        else:
            sprink_dikes = 0
            Qriv_reduc = 0


        self.sprink_dikes_tot += sprink_dikes
        # Calculate hydrologic conditions for dikes
        soilm_dike_tm1 = self.soilm_dike
        self.groundwlvl_dike, self.soilm_dike, epot, caprise_dike, redfact_dike, percol_dike, eact_dike, runoff_dike, wbalance_dike = DAgrModule(
            stage=self.stage, analysis=self.analysis, inputs=inputs).calculate_watercycle(
            timemod=timemod, evapo_ref=evapo_ref, precip=precip, CropFactTbl=inputs.CropFactTbl_dike,
            sprink=sprink_dikes, sprinkfrac=inputs.sprinkfrac,
            soilm=self.soilm_dike, groundwlvl=self.groundwlvl_dike, HoverDEM=inputs.HoverDEM_dike, inputs=inputs, module_transfer=module_transfer)

        # Calculate effects on dike
        dike_weight_red = (module_transfer.dike_weight_rel * np.minimum(np.maximum((0.0069 * self.soilm_dike/inputs.soilm_dike * 100 + 0.4131),0.01), 1))
        # dike_structure = np.minimum(module_transfer.dike_weight_rel * dike_weight_red * insp_repair,1)
        dikefrag_reduction = dike_weight_red ** (inputs.Exponent)

        # consecutive effect caused by high rainfall or high discharge
        if self.rainfall_deficit >= inputs.drought_thr:  # when drought happens based on threshold
            self.droughtperiod = True  # switch on
            self.dikedrought_time = 0  # reset dought effect timer
            if self.rainfall_deficit >= inputs.drought_thr + 0.15:
                self.higher_consecutive = True  # strong droughts case increased risk from conscec hazard

        if self.rainfall_deficit < inputs.drought_thr and self.droughtperiod == True:  # drought stop
            self.droughtperiod_end = True  # switch on drought ended
            self.droughtperiod = False  # switch off

        if self.droughtperiod_end:  # if drought eneded consec timer starts
            self.dikedrought_time += 1

            if precip >= inputs.dike_cons_thr_rain:  # lowering dikefrag due to consec rain
                if self.higher_consecutive:
                    dikefrag_reduction *= (inputs.dikedrought_cons_factor - 0.2)
                else:
                    dikefrag_reduction *= inputs.dikedrought_cons_factor

            if Q >= inputs.dike_cons_thr_discharge:  # lowering dikefrag due to consec discharge
                if self.higher_consecutive:
                    dikefrag_reduction *= (inputs.dikedrought_cons_factor - 0.2)
                else:
                    dikefrag_reduction *= inputs.dikedrought_cons_factor

            if self.dikedrought_time == inputs.consec_time:  # if times equal consec time consec effects ends
                self.droughtperiod_end = False  # resetting switches
                self.higher_consecutive = False

        if module_transfer.dike_maintenance and self.rainfall_deficit >= inputs.rainfall_deficit_thr + 50 / 1000 and timemod >= 13 and timemod <= 30:
            if timemod % 2 == 0:
                random_number = random.random()
                random_number = 0.4
                if inputs.spot_chance >= random_number:
                    dikefrag_reduction = np.minimum(dikefrag_reduction * inputs.insp_repair,2)
        # print(dikefrag_reduction)
        dike_fail_new = dike_fail_ini * dikefrag_reduction
        return dike_fail_new, Qriv_reduc

    #@profile
    def create_output_dicts(self, DamAgr_f, wlvl, fragdike, floodtiming, Qriv_reduc, DamUrb,
                            module_transfer, Q):
        module_transfer.Qriv_reduc = Qriv_reduc
        module_transfer.DamAgr_f = DamAgr_f

        if self.analysis:
            module_transfer.DikeFails_tot = self.num_dikefails_tot
            module_transfer.DamAgr_f_tot = self.DamAgr_f_tot
            module_transfer.DamUrb_tot = self.DamUrb_tot
            module_transfer.DamAgr_f = DamAgr_f
            module_transfer.DikeFails = self.num_dikefails
            module_transfer.wlvl = wlvl
            module_transfer.fragdike = fragdike
            module_transfer.floodtiming = floodtiming
            module_transfer.ClimQmax = Q
            module_transfer.sprink_dikes = self.sprink_dikes_tot
            module_transfer.DamUrb_f = DamUrb
        else:
            module_transfer.DikeFails_tot = self.num_dikefails_tot
            module_transfer.DamAgr_f_tot = self.DamAgr_f_tot
            module_transfer.DamUrb_tot = self.DamUrb_tot



