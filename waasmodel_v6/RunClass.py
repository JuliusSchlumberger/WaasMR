import numpy as np
import time
import pandas as pd
from waasmodel_v6.Waasmodel_v6 import WaasModel
from waasmodel_v6.inputs_runclass import RunClassInputs

from collections import namedtuple
import gzip
import os
import re

class WrapperRunModel:
    """
    A class that wraps the execution of a model for different scenarios and timesteps.
    """

    __slots__ = (
    'hs_pair', 'inputs', 'scenarios', 'timesteps', 'stage', 'portfolio_numbers', 'realization', 'analysis', 'test_all',
    'portfolio_number', 'scenario_dict', 'column_names', 'keys')

    def __init__(self, hazard_sector_pair, portfolio_numbers, first_scenario, last_scenario, first_timestep=0,
                 last_timestep=3600,
                 stage=1, realization=0, analysis='out_only', test_all=False):
        """
        Initialize the WrapperRunModel object.

        Args:
            hazard_sector_pair (str): The hazard-sector pair.
            portfolio_numbers (dict): Dictionary of portfolio numbers.
            first_scenario (int): The first scenario index.
            last_scenario (int): The last scenario index.
            first_timestep (int, optional): The first timestep. Defaults to 0.
            last_timestep (int, optional): The last timestep. Defaults to 3600.
            stage (int, optional): The stage. Defaults to 1.
            realization (int, optional): The realization number. Defaults to 0.
            analysis (str, optional): The analysis type. Defaults to 'out_only'.
            test_all (bool, optional): Whether to test all scenarios. Defaults to False.
        """
        self.hs_pair = hazard_sector_pair
        self.realization = str(realization).zfill(6)
        self.scenarios = range(first_scenario, last_scenario)
        self.timesteps = range(first_timestep, last_timestep)
        self.stage = stage
        self.analysis = analysis
        self.test_all = test_all
        self.inputs = RunClassInputs()  # Update with appropriate values
        self.column_names = self.inputs.realization_names

        self.scenario_dict = {}
        self.portfolio_numbers = portfolio_numbers
        self.keys = self.prepare_outputs()

    def wrapper_runner(self, outfolder: str, measures: list = ['no_measure']) -> None:
        """
        Wrapper method to run the model for scenarios and save the outputs.

        Args:
            outfolder (str): The output folder path.
            measures (list, optional): List of measures. Defaults to ['no_measure'].
        """
        self.control_prints()
        self.run_scenarios(measures=measures)
        self.save_model_outputs(outfolder)

    def control_prints(self):
        """
        Print the appropriate message based on the stage.
        """
        if self.stage == 2:
            print('Model runs accounting for interactions across hazards (per sector!)')
        elif self.stage == 3:
            print('Model runs accounting for hazard interactions across hazards & sectors')
        else:
            print('Model runs in a siloed approach.')

    def prepare_outputs(self):
        """
        Prepare the output keys based on the stage, hazard-sector pair, and analysis type.

        Returns:
            list: List of output keys.
        """
        return self.inputs.keys_dict[self.stage][self.hs_pair][self.analysis]

    def initiate_model_and_portfolio(self, measures) -> WaasModel:
        """
        Initialize the model and portfolio.

        Args:
            measures (list): List of measures implemented from the start.

        Returns:
            WaasModel: Initialized WaasModel object.
        """
        waas_model = WaasModel(portfolio_numbers=self.portfolio_numbers,
                               stage=self.stage,
                               analysis=self.analysis,
                               test_all=self.test_all,
                               hazard_sector_pair=self.hs_pair
                               )
        for measure_to_implement in measures:
            waas_model.implement_measure_once(measure_to_implement=[measure_to_implement])  # Implement measure if true
        return waas_model

    def run_scenarios(self, measures: list[str]) -> None:
        """
        Run the model for each scenario.

        Args:
            measures (list[str]): List of measures to be implemented at timestep 0.
        """
        for s in self.scenarios:
            storage_dict = {key: [] for key in self.keys}
            scenario_time = time.time()
            waas_model = self.initiate_model_and_portfolio(measures=measures)
            storage_dict = self.run_timesteps(model=waas_model,
                                              scenario=s,
                                              storage_dict=storage_dict
                                              )
            self.scenario_dict[s] = storage_dict
            print("--- %s seconds (scenario %s) ---" % (time.time() - scenario_time, s))

    def run_timesteps(self, model, scenario, storage_dict):
        """
        Run the model for each timestep.

        Args:
            model (WaasModel): WaasModel object.
            scenario (int): Scenario index.
            storage_dict (dict): Dictionary to store the outputs.

        Returns:
            dict: Updated storage_dict with outputs.
        """
        for t in self.timesteps:
            model_outputs = self.run_model(model=model, scenario=scenario, timestep=t)
            storage_dict = self.store_outputs(model_outputs=model_outputs, storage_dict=storage_dict)
        return storage_dict

    def run_model(self, model, scenario, timestep):
        """
        Run the model for a specific scenario and timestep.

        Args:
            model (WaasModel): WaasModel object.
            scenario (int): Scenario index.
            timestep (int): Timestep.

        Returns:
            object: Model outputs.
        """
        outputs = model.run_model(scenario, timestep)
        return outputs

    def store_outputs(self, model_outputs, storage_dict):
        """
        Store the model outputs in the storage dictionary.

        Args:
            model_outputs (object): Model outputs.
            storage_dict (dict): Dictionary to store the outputs.

        Returns:
            dict: Updated storage_dict with outputs.
        """
        for i in self.keys:
            storage_dict[i] = np.append(storage_dict[i], getattr(model_outputs, i))
        return storage_dict

    def save_model_outputs(self, output_folder):
        """
        Save the model outputs to a file.

        Args:
            outfolder (str): Output folder path.
        """
        # Get the path of the current script
        script_path = os.path.abspath(__file__)
        # Get the parent folder of the script
        parent_folder = os.path.dirname(script_path)
        # Get the parent folder's parent folder
        grandparent_folder = os.path.dirname(parent_folder)

        destination_folder = f'{grandparent_folder}/{output_folder}/{str(self.realization).zfill(6)}/stage_{self.stage}/{self.hs_pair}'
        # Create the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        file_name = f'{destination_folder}/portfolio_{str(self.portfolio_numbers["flood_agr"]).zfill(2)}_{str(self.portfolio_numbers["drought_agr"]).zfill(2)}_{str(self.portfolio_numbers["flood_urb"]).zfill(2)}_{str(self.portfolio_numbers["drought_shp"]).zfill(2)}.csv'
        self.store_keys_to_filepath(file_name=file_name)

    def store_keys_to_filepath(self, file_name):
        """
        Store the output keys to a file.

        Args:
            file_name (str): File name.
        """
        data_list = []
        for key in self.keys:
            for scenario in self.scenarios:
                values = self.scenario_dict[scenario][key]
                if self.analysis == 'analysis':
                    annual_values = values
                    round_values = 3
                elif self.analysis == 'out_only':
                    annual_values = values[35::36]
                    round_values = 2
                year = 1
                cc_scenario_name_number = re.split('(\d+)', self.column_names[scenario - 1])
                for value in annual_values:
                    if isinstance(value, str):
                        data_list.append({
                            'year': year,
                            'value': value,
                            'output': key,
                            'stage': self.stage,
                            'sector_hazard': self.hs_pair,
                            'cc_scenario': cc_scenario_name_number[0],
                            'climvar': int(cc_scenario_name_number[1]),
                            'portfolio': f'{str(self.portfolio_numbers["flood_agr"]).zfill(2)}_{str(self.portfolio_numbers["drought_agr"]).zfill(2)}_{str(self.portfolio_numbers["flood_urb"]).zfill(2)}_{str(self.portfolio_numbers["drought_shp"]).zfill(2)}',
                            'realization': self.realization,
                        })
                    else:
                        data_list.append({
                            'year': year,
                            'value': round(value, round_values),
                            'output': key,
                            'stage': self.stage,
                            'sector_hazard': self.hs_pair,
                            'cc_scenario': cc_scenario_name_number[0],
                            'climvar': int(cc_scenario_name_number[1]),
                            'portfolio': f'{str(self.portfolio_numbers["flood_agr"]).zfill(2)}_{str(self.portfolio_numbers["drought_agr"]).zfill(2)}_{str(self.portfolio_numbers["flood_urb"]).zfill(2)}_{str(self.portfolio_numbers["drought_shp"]).zfill(2)}',
                            'realization': self.realization,
                        })
                    year += 1

        output_df = pd.DataFrame(data_list)

        with open(file_name, 'w') as f_out:
            output_df.to_csv(f_out, index=False)

        with open(file_name, 'rb') as f_in, gzip.open(f'{file_name}.gz', 'wb') as f_out:
            f_out.writelines(f_in)

        os.remove(file_name)
