from waasmodel_v6.module_measures_toolbox import MeasureToolbox
import numpy as np

class MeasureModule:
    """
        Class representing a measure module.

        Attributes:
            stage (int): The stage of the measure module.
            hs_pair (str): The hazard-sector pair.
            toolbox (MeasureToolbox): The measure toolbox instance.
            y_lastm_f_a (int): Year of last measure implemented for flood_agr.
            y_lastm_f_u (int): Year of last measure implemented for flood_urb.
            y_lastm_d_a (int): Year of last measure implemented for drought_agr.
            y_lastm_d_s (int): Year of last measure implemented for drought_shp.
            decision_parameter_f_a_list (np.ndarray): List of decision parameters for flood_agr.
            decision_parameter_f_u_list (np.ndarray): List of decision parameters for flood_urb.
            decision_parameter_d_a_list (np.ndarray): List of decision parameters for drought_agr.
            decision_parameter_d_s_list (np.ndarray): List of decision parameters for drought_shp.

        Methods:
            __init__: Initializes a MeasureModule instance.
            initialize_portfolio: Initializes the measure portfolio.
            run_model: Runs the model for a given timestep.
            update_decision_parameter: Updates the decision parameters.
            check_tippingpoints: Checks for tipping points and implements measures.
            implement_measure: Implements a measure from the portfolio.
    """
    #@profile
    def __init__(self, stage, hazard_sector_pair, portfolio_numbers, inputs, module_transfer):
        """
            Initializes a MeasureModule instance.

            Args:
                stage (int): The stage of the measure module.
                hazard_sector_pair (str): The hazard-sector pair.
                portfolio_numbers (dict): Dictionary containing portfolio numbers for different measures.
                inputs: Input parameters.
                module_transfer: Transfer module instance.
        """
        self.stage = stage
        self.hs_pair = hazard_sector_pair
        self.toolbox = MeasureToolbox(stage=self.stage)

        module_transfer.mportfolio_f_a, module_transfer.mportfolio_f_u, module_transfer.mportfolio_d_a, module_transfer.mportfolio_d_s = self.initialize_portfolio(
            portfolio_numbers, inputs)

        # Initialize special parameters

        self.y_lastm_f_a = inputs.y_lastm_f_a
        self.y_lastm_f_u = inputs.y_lastm_f_u
        self.y_lastm_d_a = inputs.y_lastm_d_a
        self.y_lastm_d_s = inputs.y_lastm_d_s

        self.decision_parameter_f_a_list = np.ones(
            (inputs.refperiod,1)) * inputs.f_a_indicator_past_avg
        self.decision_parameter_f_u_list = np.ones(
            (inputs.refperiod,1)) * inputs.f_u_indicator_past_avg
        self.decision_parameter_d_a_list = np.ones(
            (inputs.refperiod,1)) * inputs.d_a_indicator_past_avg
        self.decision_parameter_d_s_list = np.ones(
            (inputs.refperiod,1)) * inputs.d_s_indicator_past_avg

    #@profile
    def initialize_portfolio(self, portfolio_numbers, inputs):
        """
            Initializes the measure portfolio.

            Args:
                portfolio_numbers (dict): Dictionary containing portfolio numbers for different measures.
                inputs: Input parameters.

            Returns:
                tuple: A tuple containing the initialized measure portfolios.
        """

        mportfolio_f_a = inputs.all_mportfolios_f_a[portfolio_numbers['flood_agr']]
        mportfolio_f_u = inputs.all_mportfolios_f_u[portfolio_numbers['flood_urb']]
        mportfolio_d_a = inputs.all_mportfolios_d_a[portfolio_numbers['drought_agr']]
        mportfolio_d_s = inputs.all_mportfolios_d_s[portfolio_numbers['drought_shp']]
        # print(inputs.all_mportfolios_f_a)
        # print(error)


        # remove nan values from entries
        mportfolio_f_a = mportfolio_f_a[mportfolio_f_a != 'nan']
        mportfolio_f_u = mportfolio_f_u[mportfolio_f_u != 'nan']
        mportfolio_d_a = mportfolio_d_a[mportfolio_d_a != 'nan']
        mportfolio_d_s = mportfolio_d_s[mportfolio_d_s != 'nan']
        return list(mportfolio_f_a), list(mportfolio_f_u), list(mportfolio_d_a), list(mportfolio_d_s)

    #@profile
    def run_model(self, timestep, module_transfer, inputs):
        """
                Runs the model for a given timestep.

                Args:
                    timestep (int): The current timestep.
                    module_transfer: Transfer module instance.
                    inputs: Input parameters.

                Returns:
                    module_transfer: Updated module transfer instance.
                """
        timemod = int(timestep % 36)
        year = int(timestep / 36)
        if timemod == 0:
            module_transfer = self.update_decision_parameter(year=year, module_transfer=module_transfer, inputs=inputs)
            module_transfer = self.check_tippingpoints(year=year, module_transfer=module_transfer, inputs=inputs)  # implement measures, update pathways list, update in_meas parameters
        return module_transfer

    #@profile
    def update_decision_parameter(self, year, module_transfer, inputs):
        """
                Updates the decision parameters.

                Args:
                    year (int): The current year.
                    module_transfer: Transfer module instance.
                    inputs: Input parameters.

                Returns:
                    module_transfer: Updated module transfer instance.
                """
        replace_value = int(year % inputs.refperiod)  # selector of element in list to replace by new value
        if self.hs_pair == 'flood_agr' or self.stage > 1:
            new_value = module_transfer.DamAgr_f_tot
            self.decision_parameter_f_a_list[replace_value] = new_value
            f_a_decision_value = np.sum(self.decision_parameter_f_a_list) / inputs.refperiod
            module_transfer.f_a_decision_value = f_a_decision_value
        if self.hs_pair == 'flood_urb' or self.stage > 1:
            new_value = module_transfer.DamUrb_tot
            self.decision_parameter_f_u_list[replace_value] = new_value
            f_u_decision_value = np.sum(self.decision_parameter_f_u_list) / inputs.refperiod
            module_transfer.f_u_decision_value = f_u_decision_value
        if self.hs_pair == 'drought_agr' or self.stage > 1:
            new_value = module_transfer.DamAgr_d_tot
            self.decision_parameter_d_a_list[replace_value] = new_value
            d_a_decision_value = np.sum(self.decision_parameter_d_a_list) / inputs.refperiod
            module_transfer.d_a_decision_value = d_a_decision_value
        if self.hs_pair == 'drought_shp' or self.stage > 1:
            # print(self.decision_parameter_d_s_list)
            new_value = module_transfer.DamShp_tot
            self.decision_parameter_d_s_list[replace_value] = new_value
            d_s_decision_value = np.sum(self.decision_parameter_d_s_list) / inputs.refperiod
            module_transfer.d_s_decision_value = d_s_decision_value
        return module_transfer

    #@profile
    def check_tippingpoints(self, year, module_transfer, inputs):
        """
                Checks for tipping points and implements measures.

                Args:
                    year (int): The current year.
                    module_transfer: Transfer module instance.
                    inputs: Input parameters.

                Returns:
                    module_transfer: Updated module transfer instance.
                """
        if self.hs_pair == 'flood_agr' or self.stage > 1:
            if (module_transfer.f_a_decision_value >= inputs.f_a_tp_cond or module_transfer.DamAgr_f_tot >= inputs.f_a_tp_extreme_year) and year >= self.y_lastm_f_a + inputs.yrs_btw_nmeas:
                next_measure = []
                if len(module_transfer.mportfolio_f_a) >= 1:
                    next_measure = module_transfer.mportfolio_f_a[0]
                module_transfer.mportfolio_f_a, module_transfer.pathways_list_f_a, module_transfer = self.implement_measure(
                    mportfolio=module_transfer.mportfolio_f_a, pathways_list=module_transfer.pathways_list_f_a,
                    module_transfer=module_transfer, inputs=inputs)

                # remove last measure from other sectoral pathways (dike elevation, maintenance etc.)
                try:
                    module_transfer.mportfolio_f_u.remove(next_measure)
                except ValueError:
                    pass
                    # print(f"{next_measure} is not in the list.")

                #
                # if next_measure == 'f_resilient_crops' and str(
                #         inputs.measure_numbers['d_resilient_crops']) in module_transfer.pathways_list_d_a:
                #     module_transfer.pathways_list_d_a = module_transfer.pathways_list_d_a.replace(
                #         str(inputs.measure_numbers['d_resilient_crops']), '').replace('&&', '&')

                self.y_lastm_f_a = year  # store year of last measure implemented
        if self.hs_pair == 'flood_urb' or self.stage > 1:
            if (module_transfer.f_u_decision_value >= inputs.f_u_tp_cond or module_transfer.DamUrb_tot >= inputs.f_u_tp_extreme_year) and year >= self.y_lastm_f_u + inputs.yrs_btw_nmeas:
                next_measure = []
                if len(module_transfer.mportfolio_f_u) >=1:
                    next_measure = module_transfer.mportfolio_f_u[0]
                module_transfer.mportfolio_f_u, module_transfer.pathways_list_f_u, module_transfer = self.implement_measure(
                    mportfolio=module_transfer.mportfolio_f_u, pathways_list=module_transfer.pathways_list_f_u,
                    module_transfer=module_transfer, inputs=inputs)
                self.y_lastm_f_u = year  # store year of last measure implemented

                # remove last measure from other sectoral pathways (dike elevation, maintenance etc.)
                try:
                    module_transfer.mportfolio_f_a.remove(next_measure)
                except ValueError:
                    pass

        if self.hs_pair == 'drought_agr' or self.stage > 1:
            if (module_transfer.d_a_decision_value >= inputs.d_a_tp_cond or module_transfer.DamAgr_d_tot >= inputs.d_a_tp_extreme_year) and year >= self.y_lastm_d_a + inputs.yrs_btw_nmeas:
                next_measure = []
                if len(module_transfer.mportfolio_d_a) >= 1:
                    next_measure = module_transfer.mportfolio_d_a[0]
                module_transfer.mportfolio_d_a, module_transfer.pathways_list_d_a, module_transfer = self.implement_measure(
                    mportfolio=module_transfer.mportfolio_d_a, pathways_list=module_transfer.pathways_list_d_a,
                    module_transfer=module_transfer, inputs=inputs)
                self.y_lastm_d_a = year  # store year of last measure implemented

                # if next_measure == 'f_resilient_crops' and str(
                #         inputs.measure_numbers['d_resilient_crops']) in module_transfer.pathways_list_d_a:
                #     module_transfer.pathways_list_d_a = module_transfer.pathways_list_d_a.replace(
                #         str(inputs.measure_numbers['d_resilient_crops']), '').replace('&&', '&')

        if self.hs_pair == 'drought_shp' or self.stage > 1:
            if (module_transfer.d_s_decision_value >= inputs.d_s_tp_cond or module_transfer.DamShp_tot >= inputs.d_s_tp_extreme_year) and year >= self.y_lastm_d_s + inputs.yrs_btw_nmeas:
                module_transfer.mportfolio_d_s, module_transfer.pathways_list_d_s, module_transfer = self.implement_measure(
                    mportfolio=module_transfer.mportfolio_d_s, pathways_list=module_transfer.pathways_list_d_s, module_transfer=module_transfer, inputs=inputs)
                self.y_lastm_d_s = year  # store year of last measure implemented
        return module_transfer

    #@profile
    def implement_measure(self, mportfolio, pathways_list, module_transfer, inputs):
        """
        Helper function to choose and implement a measure.

        Args:
            mportfolio: List of initialized methods, each attributed by additional input parameters.
            pathways_list: List to store information when a measure has been implemented.
            module_transfer: Transfer module instance.
            inputs: Input parameters.

        Returns:
            mportfolio: Updated mportfolio list.
            pathways_list: Updated pathways_list.
            module_transfer: Updated module transfer instance.
        """

        if not mportfolio or mportfolio[0] in ['no_measure', np.NaN]:  # check if a measure is still available
            mportfolio = []
            type_update = 'failing'
            # print(type_update)
            pathways_list = pathways_list

        elif len(mportfolio) == 1:  # check if list of measures only contains one measure
            choice = getattr(self.toolbox, mportfolio[0])  # select measure
            module_transfer, measure_number = choice(module_transfer, inputs)  # call function to implement measure
            type_update = 'last_measure'
            # print(type_update)
            pathways_list = pathways_list + '&' + str(measure_number)
            mportfolio = []  # remove measure from list of options
        else:
            type_update = 'another_measure'
            # print(type_update)
            choice = getattr(self.toolbox, mportfolio[0])  # select measure
            module_transfer, measure_number = choice(module_transfer, inputs)  # call function to implement. Return updated in_meas parameters
            # print(measure_number)
            pathways_list = pathways_list + '&' + str(measure_number)
            mportfolio = mportfolio[1:]  # remove measure from list of options
        return mportfolio, pathways_list, module_transfer
