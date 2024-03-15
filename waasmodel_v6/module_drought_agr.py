import numpy as np
from pcraster import *

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.#print_stats)


class DAgrModule:
    __slots__ = (
        'stage', 'analysis', 'soilm', 'groundwlvl', 'damfrac_tot_d', 'epot_totgr', 'collectwater_rain',
        'collectwater_gw', 'rain_def', 'collectwater_riv', 'collectwater_ditch', 'survfrac', 'DamAgr_d_tot', 'yielddef',
        'sprink_rain', 'sprink_riv', 'sprink_gw'
    )
    #@profile
    def __init__(self, stage, analysis, inputs):
        self.analysis = analysis
        self.stage = stage

        # initialize special values
        self.soilm = inputs.soilm   # [m]
        self.groundwlvl = inputs.groundwlvl     # [m]
        self.damfrac_tot_d = inputs.damfrac_tot_d   # [-]
        self.epot_totgr = inputs.epot_totgr     # [m]
        self.collectwater_rain = inputs.collectwater_rain  # collected water at beginning of scenario [m3]
        self.collectwater_gw = inputs.collectwater_gw   # collected water at beginning of scenario [m3]
        self.collectwater_riv = inputs.collectwater_riv # collected water at beginning of scenario [m3]
        self.collectwater_ditch = inputs.collectwater_ditch # collected water at beginning of scenario [m3]
        self.DamAgr_d_tot = 0   # [Mio EUR]
        self.rain_def = 0   # [m]
        self.sprink_riv = 0
        self.sprink_rain = 0
        self.sprink_gw = 0

    #@profile
    def run_module(self, timestep, scenario, module_transfer, inputs):
        """
                Run the module to calculate the water cycle and deficit-based yield losses of the agricultural land.

                Args:
                    timestep (int): The current timestep.
                    scenario (int): The scenario number.
                    module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                    inputs (Inputs): An object containing input data.

                Returns:
                    ModuleTransfer: An updated object with calculated values.
                """

        # Initialize variables from other modules
        timemod, deathd, evapo_ref, precip, reducd, reducp, remyield, Q, rain_def_mean = self.initiate_timeinputs(
            timestep=timestep, scenario=scenario,
            module_transfer=module_transfer, inputs=inputs)

        # Correct the river discharge in case water is used for dike irrigation
        Q -= module_transfer.Qriv_reduc

        self.reset_values_newyear(timemod=timemod, module_transfer=module_transfer, inputs=inputs)
        if module_transfer.collectwater_switch:
            sprink_need = self.calculate_irrigation_need(timemod,inputs, module_transfer, soilm=self.soilm)
            Qriv_reduc = self.collectwater(Q, precip, module_transfer, sprink_need, inputs)
            sprink, sprink_lack = self.irrigation_calculation(areas=module_transfer.agri_areas,
                                                              module_transfer=module_transfer, sprink_need=sprink_need)
            collectwater = self.collectwater_rain + self.collectwater_gw + self.collectwater_riv + self.collectwater_ditch
        else:
            sprink = 0
            collectwater = 0
            Qriv_reduc = 0

        # Calculate hydrologic conditions for crops
        self.groundwlvl, self.soilm, epot, caprise, redfact, percol, eact, runoff, wbalance = self.calculate_watercycle(inputs=inputs,
            timemod=timemod, evapo_ref=evapo_ref, precip=precip, CropFactTbl=inputs.CropFactTbl, sprink=sprink,
            sprinkfrac=inputs.sprinkfrac,
            soilm=self.soilm, groundwlvl=self.groundwlvl, HoverDEM=inputs.HoverDEM, module_transfer=module_transfer)

        # calculate damage for crops
        eratio, damfrac_t_d, ypot, yact, DamAgr_d = self.calculate_crop_drought_damage(precip=precip, reducp=reducp, deathd=deathd,
                                                                                       reducd=reducd, epot=epot,
                                                                                       eact=eact, remyield=remyield,
                                                                                       timemod=timemod,
                                                                                       module_transfer=module_transfer,
                                                                                       inputs=inputs,
                                                                                       rain_def_mean=rain_def_mean)

        self.create_output_dicts(precip=precip, evapo_ref=evapo_ref, collectwater=collectwater, eratio=eratio,
                                 damfrac_t_d=damfrac_t_d, ypot=ypot, yact=yact, DamAgr_d=DamAgr_d,
                                 Qriv_reduc=Qriv_reduc, sprink=sprink, epot=epot, caprise=caprise,
                                 percol=percol, eact=eact, runoff=runoff, wbalance=wbalance,
                                 module_transfer=module_transfer)

        return module_transfer

    #@profile
    def initiate_timeinputs(self, timestep, scenario,module_transfer, inputs):
        """
                Initialize time-based inputs for the module.

                Args:
                    timestep (int): The current timestep.
                    scenario (int): The scenario number.
                    module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                    inputs (Inputs): An object containing input data.

                Returns:
                    Tuple[int, int, float, float, float, float, float, float, float]: A tuple containing the initialized time-based inputs.
                """
        timemod = timestep % 36 # timestep within a year

        deathd = inputs.DeathDamageTbl[timemod, inputs.crop_type]
        evapo_ref = inputs.evapo_ref_scenarios[timestep, scenario - 1]
        precip = inputs.precip_scenarios[timestep, scenario - 1]
        reducd = inputs.ReducDamageTbl[timemod, inputs.crop_type]
        reducp = module_transfer.ReducPointTbl[timemod, inputs.crop_type]
        remyield = inputs.RemYieldTbl[timemod, inputs.crop_type]
        Q = inputs.ClimQmax[timestep, scenario - 1]
        raindef_mean = inputs.raindef_mean.loc[timemod,:].values[0]

        return timemod, deathd, evapo_ref, precip, reducd, reducp, remyield, Q, raindef_mean

    #@profile
    def reset_values_newyear(self,timemod, module_transfer, inputs):
        """
                Reset the values at the start of a new year.

                Args:
                    timemod (int): The current timestep within a year.
                    module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                    inputs (Inputs): An object containing input data.

                Returns:
                    None
                """
        if timemod == 0:
            self.damfrac_tot_d = 0
            self.DamAgr_d_tot = 0
            self.epot_totgr = 0
            module_transfer.survfrac = inputs.agriculture
            self.yielddef = 0
            self.collectwater_rain = 0
            self.collectwater_gw = 0
            self.collectwater_riv = 0
            self.collectwater_ditch = 0
            self.rain_def = 0
            self.sprink_rain = 0
            self.sprink_gw = 0
            self.sprink_riv = 0
            if self.stage == 1:
                module_transfer.revenue_agr = inputs.revenue_ref
            pass

    #@profile
    def calculate_irrigation_need(self, timemod, inputs, module_transfer, soilm):
        """
                Calculate the irrigation need for crops based on the current timestep within a year.

                Args:
                    timemod (int): The current timestep within a year.
                    inputs (Inputs): An object containing input data.
                    module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                    soilm (float): The soil moisture level.

                Returns:
                    float: The irrigation need for crops.
                """
        #  irrigation for crops: only in the months of crop growth, for dike all year long
        if timemod >= 9 and timemod <= 27:
            sprink_need = np.maximum(0, inputs.sprinkfrac / module_transfer.irrigation_effectiveness * (
                    module_transfer.soilm_max - soilm))  # required water [m/m2/decade] for irrigation + losses
        else:
            sprink_need = 0
        return sprink_need

    def irrigation_calculation(self, areas, module_transfer, sprink_need):
        """
                Calculate the irrigation amount and shortage based on different irrigation methods and needs.

                Args:
                    areas (float): areas for irrigation.
                    module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                    sprink_need (float): The required irrigation water amount.

                Returns:
                    Tuple[float, float]: A tuple containing the total irrigation amount applied and the remaining irrigation need.
                """


        sprink = 0
        if module_transfer.d_rainwater_tank and sprink_need > 0:
            self.collectwater_rain, sprink_rain, sprink_need = self.sprinkling(collectwater=self.collectwater_rain,
                                                                           area=areas,
                                                                           sprink_need=sprink_need)
            sprink += sprink_rain
            self.sprink_rain = sprink_rain * module_transfer.agri_areas * 100 * 100 # [m/m2/10days] to [m3/10days]


        if module_transfer.d_groundwater_irrigation and sprink_need > 0:
            # groundwater cannot be stored if pumped up.
            _, sprink_gw, sprink_need = self.sprinkling(collectwater=self.collectwater_gw,
                                                                           area=areas,
                                                                           sprink_need=sprink_need)
            sprink += sprink_gw
            self.sprink_gw += sprink_gw * module_transfer.agri_areas * 100 * 100 # [m/m2/10days] to [m3/10days]
        if module_transfer.d_river_irrigation and sprink_need > 0:
            # water cannot bet stored if pumped up
            sprink_need_old = sprink_need
            _, sprink_riv, sprink_need = self.sprinkling(collectwater=self.collectwater_riv,
                                                                           area=areas,
                                                                           sprink_need=sprink_need)
            sprink += sprink_riv
            self.sprink_riv += sprink_riv * module_transfer.agri_areas * 100 * 100 # [m/m2/10days] to [m3/10days]

        return sprink, sprink_need

    #@profile
    def calculate_watercycle(self, timemod, evapo_ref, precip, CropFactTbl, sprink, sprinkfrac,soilm,groundwlvl, HoverDEM, module_transfer,inputs):
        """
                Calculate the water cycle components for the given time step.

                Args:
                    timemod (int): The time step within a year.
                    evapo_ref (float): Reference evapotranspiration according to Makkink [m/10days].
                    precip (float): Precipitation amount [m/10days]
                    CropFactTbl (List[float]): List of crop factors for different crops.
                    sprink (float): Amount of sprinkler irrigation [m/10days]
                    sprinkfrac (float): Sprinkler irrigation effectiveness factor.
                    soilm (float): Soil moisture [m]
                    groundwlvl (float): Groundwater level [m]
                    HoverDEM (float): HoverDEM value [m]
                    module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                    inputs: An object containing input parameters.

                Returns:
                    Tuple[float, float, float, float, float, float, float, float, float]: A tuple containing the updated ground
                    water level, soil moisture, potential evapotranspiration, capillary rise, reduction factor, percolation,
                    actual evapotranspiration, runoff, and water balance.

                """
        # Create crop-diverse map infos based on LandUse classes (time-step specific)
        cropfact = CropFactTbl[timemod]

        if groundwlvl > 0:
            groundwlvl_tm1 = np.minimum(groundwlvl, 10) * -1  # in m
        else:
            groundwlvl_tm1 = groundwlvl

        epot = evapo_ref * cropfact

        # Update effective soil moisture from irrigation
        soilm_tm1 = soilm + sprink * inputs.irrigation_effectiveness / sprinkfrac

        # Percolation
        percol = np.maximum(soilm_tm1 + precip - (module_transfer.soilm_max), 0)

        # Capillary rise
        capfact_r = np.maximum(0.0, (1 - (soilm_tm1 / (module_transfer.soilm_max))))
        capfact_w = 1.0 - ((-1 * groundwlvl_tm1 - 0.5 * inputs.dr) / (
                    inputs.dc - 0.5 * inputs.dr)) if -1.0 * groundwlvl_tm1 > 0.5 * inputs.dr and -1.0 * groundwlvl_tm1 < inputs.dc else 1.0 if -1.0 * groundwlvl_tm1 <= 0.5 * inputs.dr else 0
        capfact = capfact_r * capfact_w
        caprise = inputs.caprise_m * capfact

        # Soil moisture suction
        rootvol = np.minimum(100.0, np.maximum(0.0, (soilm_tm1 / inputs.dr) * 100.0))
        pf = self.calc_pf_fit(rootvol=rootvol) if inputs.soilpf == 1 else 0

        redfact = 0 if pf < inputs.pfred else 1.0 if pf > inputs.pfmax else (pf - inputs.pfred) / (
                    inputs.pfmax - inputs.pfred)

        eact = np.minimum((soilm_tm1 + precip + caprise - percol), (1.0 - redfact) * epot)

        if timemod >= 9 and timemod <= 26:
            self.rain_def += evapo_ref - precip
            self.rain_def = np.maximum(self.rain_def, 0)

        soilm = soilm_tm1 + precip + caprise - percol - eact
        runoff = soilm - (module_transfer.soilm_max) if soilm > module_transfer.soilm_max else 0.0
        soilm = soilm - runoff

        if HoverDEM <= groundwlvl_tm1:
            qflow = (HoverDEM - groundwlvl_tm1) / module_transfer.rout
        elif HoverDEM > groundwlvl_tm1:
            qflow = (HoverDEM - groundwlvl_tm1) / module_transfer.rin
        else:
            qflow = 0.0
        delta_h_ground = qflow * 10

        groundwlvl = groundwlvl_tm1 + (delta_h_ground - caprise + percol + inputs.seep) / inputs.porosity

        delta_groundwlvl = (groundwlvl - groundwlvl_tm1) * inputs.porosity
        delta_soilm = soilm - soilm_tm1

        wbalance = precip - runoff - eact + sprink - delta_groundwlvl - delta_soilm + inputs.seep + delta_h_ground

        return groundwlvl, soilm, epot, caprise, redfact, percol, eact, runoff, wbalance

    #@profile
    def calculate_crop_drought_damage(self, precip, reducp, deathd, reducd, epot,eact, remyield, timemod, module_transfer,inputs, rain_def_mean):
        """
        Calculate crop drought damage based on the given parameters.

        Args:
            reducp (float): Reduction factor for crop p.
            deathd (float): Death factor for crop d.
            reducd (float): Reduction factor for crop d.
            epot (float): Potential evapotranspiration.
            eact (float): Actual evapotranspiration.
            remyield (float): Remaining yield.
            timemod (int): The time step within a year.
            module_transfer (ModuleTransfer): An object containing transferred values from other modules.
            inputs (YourInputs): An object containing input parameters.
            rain_def_mean (float): Mean rain deficit.

        Returns:
            Tuple[float, float, float, float, float]: A tuple containing the calculated values for eratio, damfrac_t_d,
            ypot, yact, and DamAgr_d.

        """
        # Drought condition as defined in Van Loon et al. (2015)
        if rain_def_mean != 0 and self.rain_def - rain_def_mean > 0:
            module_transfer.droughtdays += 10  # only calculate damages if in growing season and deficit is larger than the mean:

            # Veerle: Calculate waterlogging conditions
            if self.stage >= 2:
                module_transfer.agri_areas_exposed_count = self.waterlogging(precip=precip, timemod=timemod, module_transfer=module_transfer,
                                                       inputs=inputs)


            # if self.stage == 1 or inputs.waterlogging_switch == False:
            # calculates the damage fraction for only drought exposed crops
            reducp = reducp + module_transfer.crop_vul
            deathp = np.maximum(0,inputs.deathp + module_transfer.crop_vul / 2)

            if epot == 0:
                eratio = 1
            else:
                eratio = eact / epot

            if eratio >= 1:
                damfrac = 0.0
            elif reducp <= eratio and eratio < 1.0:
                damfrac = reducd * (1.0 - eratio) / (1.0 - reducp)
            elif deathp < eratio and eratio < reducp:
                damfrac = reducd + (deathd - reducd) * (reducp - eratio) / (reducp - deathp)
            elif eratio <= deathp:
                damfrac = deathd
            else:
                raise ValueError('Error in calculating crop drought damage!')

            y = inputs.agriculture * damfrac

            if self.stage >= 2 and inputs.waterlogging_switch == True:
                # Part that calculates the damage fraction for waterlogged exposed crops
                # Calculate watterlogging damage (Based on AGRICOM)
                RP = np.maximum(module_transfer.SP,module_transfer.RP - module_transfer.crop_vul)  # Reductie punt / reduction point
                RS = module_transfer.RS  # Reductie schade / reduction damage
                SP = np.maximum(0,module_transfer.SP - module_transfer.crop_vul) # Sterftepunt / death point

                ratio = eact / epot  # actual evaporation / potential evaporation
                if ratio >= 1:
                    damage_fraction = 0
                elif ratio > RP and ratio < 1:
                    damage_fraction = RS * (1 - ratio) / (1 - RP)
                elif ratio > SP and ratio <= RP:
                    damage_fraction = RS + (1 - RS) * (RP - ratio) / (RP - SP)
                elif ratio <= SP:
                    damage_fraction = 1

                x = module_transfer.WL_time_exposed
                x[x < timemod - 2] = damfrac
                x[x >= timemod - 2] = damage_fraction
                # condlist = [module_transfer.WL_time_exposed >= timemod - 2, module_transfer.WL_time_exposed < timemod - 2]
                # choicelist = [module_transfer.WL_time_exposed + damage_fraction, module_transfer.WL_time_exposed*10]
                # spatial_damage_fraction = np.select(condlist, choicelist)
                # #print(x)
                y = x * inputs.agriculture

            survfracTm1 = module_transfer.survfrac
            module_transfer.survfrac = module_transfer.survfrac * ((1 - y) - inputs.non_agriculture)  # to take into account the stage of the crop, the remaining yield is included
            #print(module_transfer.survfrac)

            # Update survival fraction
            module_transfer.survfrac = module_transfer.survfrac * (1 - damfrac)
            damfrac_t_d = remyield * (survfracTm1 - module_transfer.survfrac)   # damage fraction taking into account which is yet to be harvested
            self.damfrac_tot_d = 1 - np.average(module_transfer.survfrac[module_transfer.agriculture==1])

            if 9 <= timemod <= 26:
                self.epot_totgr += epot

            efrac = self.epot_totgr / inputs.epot_totref
            ypot = inputs.ypot_ref * efrac
            yact = ypot * module_transfer.survfrac * (1 - int(module_transfer.d_resilient_crop_switch) * 0.2)
            self.yielddef = (ypot - yact) * module_transfer.agriculture
            DamAgr_d = np.sum(self.yielddef * inputs.agrprize / 1000000)    # [Mio EUR]
            self.DamAgr_d_tot = DamAgr_d

        else:   # no drought damages
            eratio = 1
            damfrac_t_d = 0
            module_transfer.droughtdays = 0

            if 9 <= timemod <= 26:
                self.epot_totgr += epot

            efrac = self.epot_totgr / inputs.epot_totref
            ypot = inputs.ypot_ref * efrac
            yact = ypot
            DamAgr_d = 0

        if timemod == 35:
            module_transfer.revenue_agr = np.maximum(module_transfer.revenue_agr - self.DamAgr_d_tot, 0)
            # if self.stage == 1:
            #     revenue_loss_d = (inputs.ypot_ref * module_transfer.lost_cropland * inputs.agrprize) / 1000000
            #     module_transfer.revenue_agr = np.maximum(module_transfer.revenue_agr - revenue_loss_d, 0)
            revenue_loss_d = (yact * module_transfer.lost_cropland * inputs.agrprize) / 1000000
            module_transfer.revenue_agr = np.maximum(module_transfer.revenue_agr - revenue_loss_d, 0)

            if self.stage == 3:
                revenue_loss_d = np.sum((yact * module_transfer.agriculture * inputs.agrprize * (
                        1 - module_transfer.revenue_reduc)) / 1000000)
                module_transfer.revenue_agr = np.maximum(module_transfer.revenue_agr - revenue_loss_d, 0)

        return eratio, damfrac_t_d, ypot, yact, DamAgr_d

    #@profile
    def collectwater(self, Q, precip, module_transfer, sprink_need, inputs):
        """
            Collect water for irrigation based on the given parameters.

            Args:
                Q (float): Flow rate of the river.
                precip (float): Amount of precipitation.
                module_transfer (ModuleTransfer): An object containing transferred values from other modules.
                sprink_need (float): Water requirement for sprinkler irrigation.
                inputs (YourInputs): An object containing input parameters.

            Returns:
                float: The reduction in river flow rate due to water collection.

            """
        Qriv_reduc = 0

        if module_transfer.d_rainwater_tank:
            self.collectwater_rain += precip * module_transfer.agri_areas * 0.1 * 100 * 100 # [m3/10days]
            self.collectwater_rain = np.minimum(inputs.tanksize * 50, self.collectwater_rain)

        if module_transfer.d_river_irrigation:
            if Q > inputs.Qmin:
                collectwater_need = sprink_need * (module_transfer.agri_areas * 100 * 100) # [m3/10days]
                self.collectwater_riv = np.minimum(inputs.Vpump,
                                            np.minimum((Q - inputs.Qmin) * (10 * 24 * 60 * 60), collectwater_need)) # [m3/10days]
            else:
                self.collectwater_riv = 0
            Qriv_reduc = self.collectwater_riv / (24 * 60 * 60 * 10) * 1.4   # [m3/s]

        if module_transfer.d_groundwater_irrigation:
            self.collectwater_gw = inputs.gw_volume # [m3/10days]

        return Qriv_reduc

    #@profile
    def sprinkling(self, collectwater, sprink_need, area):
        """
            Perform sprinkling irrigation based on the available water and water requirement.

            Args:
                collectwater (float): Amount of water available for irrigation.
                sprink_need (float): Water requirement for sprinkler irrigation.
                area (float): Area of land to be irrigated [ha]

            Returns:
                tuple: A tuple containing the updated values of collectwater, sprink, and sprink_need.

            """
        if sprink_need >= collectwater / (area * 100 * 100):
            sprink = collectwater / (area * 100 * 100)
            sprink_need -= sprink
            collectwater = 0
        else:
            sprink = sprink_need
            sprink_need = 0
            collectwater -= sprink * (area * 100 * 100)

        return collectwater, sprink, sprink_need

    def \
            waterlogging(self,  precip, timemod, module_transfer, inputs):
        if self.stage >= 2:
            # Do the pre-calculation work
            precip_temp = np.zeros((60, 125))

            def getinfiltrationmap(LandUse):
                landuse8 = LandUse == 8
                landuse9 = LandUse == 9
                landuse10 = LandUse == 10
                landuse17 = LandUse == 17
                landuse18 = LandUse == 18
                final = landuse8 | landuse9 | landuse10 | landuse17 | landuse18
                return final

            infiltration_temp = getinfiltrationmap(module_transfer.LandUse)
            infiltration_temp = pcr2numpy(infiltration_temp, -999)

            if inputs.soilcracks_switch == True:
                if module_transfer.droughtdays >= module_transfer.droughtdays_threshold:  # Only occurs if drought is longer than 60 days
                    maxstorage = module_transfer.soilm_max * (1 - module_transfer.storage_reduc)  # Decrease the max storage with 10%
                    possiblestorage = np.maximum(maxstorage - self.soilm, 0)
                else:
                    possiblestorage = module_transfer.soilm_max - self.soilm
            else:
                possiblestorage = module_transfer.soilm_max - self.soilm

                # Calculate numpy array's with precipitation and infiltration capacity
            precipitation_map = precip_temp + precip
            # #print(precipitation_map)
            infiltration_map = infiltration_temp * possiblestorage

            # Turn them into pcraster files
            precip_map = numpy2pcr(Scalar, precipitation_map, -999)
            infil_cap = numpy2pcr(Scalar, infiltration_map, -999)

            # Calculate Local Drainage Direction
            local_drainage = ldd(module_transfer.DEM)
            local_drainage_clean = lddrepair(local_drainage)
            # report(local_drainage_clean, 'ldd.map')
            # aguila('ldd.map')

            # Calculate water depth
            # state = accuthresholdstate(local_drainage_clean,precip_map, infil_cap)
            # plot(state)
            runoff = accuthresholdflux(local_drainage_clean, precip_map, infil_cap)
            # plot(runoff)

            # Depending on the growing season a certain depth causes damages
            if timemod >= 9 and timemod <= 12:  ## Germinating stage
                waterloggingdepth = runoff >= module_transfer.germ_stage_depth
            if timemod > 12 and timemod <= 27:  ## Plants in the growing season
                waterloggingdepth = runoff >= module_transfer.grow_stage_depth
            else:
                waterloggingdepth = runoff >= 100  ## Very high value that won't happen

            # Area of agricultural land damaged by waterlogging
            agriculture = module_transfer.LandUse == 9  # Make map with only the agricultural areas
            exposedagriculture = agriculture & waterloggingdepth  # Combine into map which shows the waterlogged agriculture
            agri_areas_exposed = pcr2numpy(exposedagriculture, -999)  # Convert pcraster map into numpy array
            agri_areas_exposed_count = np.count_nonzero(
                agri_areas_exposed)  # Calculate the exposed area [in ha]
            # #print(timemod)
            # #print(agri_areas_exposed_count)
            a = agri_areas_exposed
            # a[a == 1] = timemod ## if value in numpy array is 1, store the timestep
            module_transfer.WL_time_exposed[a == 1] = timemod  ## if value in numpy array is 1, store the timestep
            # #print(module_transfer.WL_time_exposed)

            return agri_areas_exposed_count


    #@profile
    def create_output_dicts(self, precip, evapo_ref, collectwater, eratio, damfrac_t_d, ypot, yact, DamAgr_d,
                            Qriv_reduc, sprink, module_transfer, epot, caprise, percol, eact, runoff,
                            wbalance):
        module_transfer.precip = precip
        module_transfer.evapo_ref = evapo_ref
        module_transfer.survfrac = module_transfer.survfrac
        module_transfer.Qriv_reduc += Qriv_reduc
        module_transfer.sprink_riv = self.sprink_riv
        module_transfer.sprink_gw = self.sprink_gw

        if self.analysis == 'analysis':
            module_transfer.DamAgr_d_tot = self.DamAgr_d_tot
            module_transfer.DamAgr_d = DamAgr_d
            module_transfer.caprise = caprise
            module_transfer.collectwater = collectwater
            module_transfer.epot = epot
            module_transfer.eact = eact
            module_transfer.eratio = eratio
            module_transfer.soilm = self.soilm
            module_transfer.groundwlvl = self.groundwlvl
            module_transfer.percol = percol
            module_transfer.wbalance = wbalance
            module_transfer.survfrac = module_transfer.survfrac
            module_transfer.runoff = runoff
            module_transfer.sprink = sprink
            module_transfer.damfrac_t_d = damfrac_t_d
            module_transfer.damfrac_tot_d = self.damfrac_tot_d
            module_transfer.yact = np.average(yact)
            module_transfer.ypot = ypot
        else:
            module_transfer.DamAgr_d_tot = self.DamAgr_d_tot
            module_transfer.DamAgr_d = self.DamAgr_d_tot
            # print(self.DamAgr_d_tot)

    #@profile
    def calc_pf_fit(self, rootvol):
        """Fit from input table (see Misc.py for the fit). Polyonimal fit 3rd degree."""
        pf = np.maximum(0, -2.46285709 * 10 ** (-5) * rootvol ** 3 + 3.53904855 * 10 ** (
            -3) * rootvol ** 2 - 2.02754270 * 10 ** (-1) * rootvol ** 1 + 6.89184020)
        return pf