import numpy as np
from waasmodel_v6.helperfunctions import Qhrelation


class DShpModule:
    __slots__ = ('stage', 'analysis', 'test_all', 'DamShp_tot')

    def __init__(self, stage, analysis, test_all):
        """
        Initialize the DShpModule.

        Args:
            stage (int): The stage of the module.
            analysis (bool): Flag indicating whether analysis is enabled.
            test_all (bool): Flag indicating whether to test all scenarios.
        """
        self.analysis = analysis
        self.stage = stage
        self.test_all = test_all
        self.DamShp_tot = 0

    def run_module(self, timestep, scenario, module_transfer, inputs):
        """
        Run the DShpModule for a given timestep and scenario.

        Args:
            timestep (int): The current timestep.
            scenario (int): The current scenario.
            module_transfer (ModuleTransfer): The module transfer instance.
            inputs (Inputs): The input parameters.

        Returns:
            module_transfer (ModuleTransfer): The updated module transfer instance.
        """
        Q, timemod = self.initiate_timeinputs(timestep, scenario, inputs)
        self.reset_values_newyear(timemod)

        if self.stage > 1:
            Q -= module_transfer.Qriv_reduc

        if Q > 4000 and not self.test_all:
            DamShp = 0
            dredged_depth = 0
            min_wdepth = 100
            load_fact = 1
        else:
            min_wdepth, dredged_depth = self.calculate_min_flowdepth(Q, module_transfer, timestep, scenario, inputs)
            load_fact = self.calculate_ship_loading(min_wdepth, module_transfer)
            DamShp = self.calculate_adapted_prize(load_fact, module_transfer, inputs)

        self.create_output_dicts(DamShp, load_fact, Q, min_wdepth, dredged_depth, module_transfer)
        return module_transfer

    def calculate_min_flowdepth(self, Q, module_transfer, timestep, scenario, inputs):
        """
        Calculate the minimum flow depth.

        Args:
            Q (float): The flow rate.
            module_transfer (ModuleTransfer): The module transfer instance.
            timestep (int): The current timestep.
            scenario (int): The current scenario.
            inputs (Inputs): The input parameters.

        Returns:
            min_wdepth (float): The minimum flow depth.
            dredged_depth (float): The dredged depth.
        """
        mean_lvl = Qhrelation(Q, Levq788=inputs.Levq788,
                              Fact788=module_transfer.Fact788, Levq7150=inputs.Levq7150,
                              Fact7150=module_transfer.Fact7150, Levq16000=inputs.Levq16000,
                              Fact16000=module_transfer.Fact16000, Levq20000=inputs.Levq20000,
                              Fact20000=module_transfer.Fact20000, Dif_2=module_transfer.Dif_2,
                              Dif_6=module_transfer.Dif_6, Dif_7=module_transfer.Dif_7)

        if module_transfer.dredging_channel:
            dredged_depth, module_transfer = self.dredge_depth(timestep, scenario, module_transfer, inputs)
        else:
            dredged_depth = 0

        min_wdepth = np.min(mean_lvl - inputs.DEM_riv) + dredged_depth
        return min_wdepth, dredged_depth

    def calculate_ship_loading(self, min_wdepth, module_transfer):
        """
        Calculate the ship loading factor.

        Args:
            min_wdepth (float): The minimum water depth.
            module_transfer (ModuleTransfer): The module transfer instance.

        Returns:
            load_fact (float): The ship loading factor.
        """
        load_fact = np.maximum(0.0, np.minimum(1.0, (min_wdepth - module_transfer.used_ships[2]) /
                                                (module_transfer.used_ships[0] - module_transfer.used_ships[1])))
        return load_fact

    def initiate_timeinputs(self, timestep, scenario, inputs):
        """
        Initialize time inputs.

        Args:
            timestep (int): The current timestep.
            scenario (int): The current scenario.
            inputs (Inputs): The input parameters.

        Returns:
            Q (float): The flow rate.
            timemod (int): The modified timestep.
        """
        timemod = timestep % 36
        Q = inputs.ClimShip[timestep, scenario - 1]
        return Q, timemod

    def reset_values_newyear(self, timemod):
        """
        Reset values at the start of a new year.

        Args:
            timemod (int): The modified timestep.
        """
        if timemod == 0:
            self.DamShp_tot = 0

    def create_output_dicts(self, DamShp, load_fact, Q, min_wdepth, dredged_depth, module_transfer):
        """
        Create output dictionaries for the module transfer instance.

        Args:
            DamShp (float): The ship damage.
            load_fact (float): The ship loading factor.
            Q (float): The flow rate.
            min_wdepth (float): The minimum flow depth.
            dredged_depth (float): The dredged depth.
            module_transfer (ModuleTransfer): The module transfer instance.
        """
        module_transfer.DamShp_tot = self.DamShp_tot

        if self.analysis:
            module_transfer.dredged_depth = dredged_depth
            module_transfer.DamShp = DamShp
            module_transfer.load_perc = load_fact
            module_transfer.ClimShip = Q
            module_transfer.wdepth_min = min_wdepth

    def calculate_adapted_prize(self, load_fact, module_transfer, inputs):
        """
        Calculate the adapted price and ship damage.

        Args:
            load_fact (float): The ship loading factor.
            module_transfer (ModuleTransfer): The module transfer instance.
            inputs (Inputs): The input parameters.

        Returns:
            DamShp (float): The ship damage.
        """
        if load_fact > inputs.min_load_capac:
            load = load_fact * inputs.freight_decade  # maximum transport volume per 10 days
            delayed_load = inputs.freight_decade - load
            adapted_price = np.minimum((inputs.cost_per_tonnage / load_fact) * 0.5, inputs.road_transp)
        else:
            load = 0
            delayed_load = inputs.freight_decade
            adapted_price = inputs.road_transp

        extra_transp_volume = inputs.freight_decade - load  # per 10 days
        DamShp = adapted_price * extra_transp_volume * module_transfer.multimode_cost_red
        self.DamShp_tot += DamShp
        return DamShp

    def dredge_depth(self, timestep, scenario, module_transfer, inputs):
        """
        Identify the remaining dredge depth.

        Args:
            timestep (int): The current timestep.
            scenario (int): The current scenario.
            module_transfer (ModuleTransfer): The module transfer instance.
            inputs (Inputs): The input parameters.

        Returns:
            dredged_depth (float): The dredged depth.
            module_transfer (ModuleTransfer): The updated module transfer instance.
        """
        if self.stage > 1:
            if timestep < 5 * 36:
                Q_accum = np.sum(inputs.ClimQmax[:timestep, scenario - 1])
            else:
                Q_accum = np.sum(inputs.ClimQmax[timestep - 5 * 36:timestep, scenario - 1])
            Q_accum = 10000000000000
        else:
            Q_accum = 1000000000000

        if timestep < module_transfer.init_dredging + 36 * inputs.change_years[0] and Q_accum > inputs.Q_accum_thr:
            dredged_depth = inputs.dredged_depths[0]
        elif timestep < module_transfer.init_dredging + 36 * inputs.change_years[1] or (
                Q_accum < inputs.Q_accum_thr and timestep < module_transfer.init_dredging + 36 *
                inputs.change_years[0]):
            dredged_depth = inputs.dredged_depths[1]
        elif timestep < module_transfer.init_dredging + 36 * inputs.change_years[2] or (
                Q_accum < inputs.Q_accum_thr and timestep < module_transfer.init_dredging + 36 *
                inputs.change_years[1]):
            dredged_depth = inputs.dredged_depths[2]
        elif timestep == module_transfer.init_dredging + 36 * inputs.change_years[2] or (
                Q_accum < inputs.Q_accum_thr and timestep > module_transfer.init_dredging + 36 *
                inputs.change_years[1]):
            dredged_depth = inputs.dredged_depths[0]
            module_transfer.init_dredging = module_transfer.timestep
        else:
            dredged_depth = inputs.dredged_depth

        return dredged_depth, module_transfer