from waasmodel_v6.inputs_waasmodel import ModelInputs
from waasmodel_v6.module_flood import FloodModule
from waasmodel_v6.module_drought_agr import DAgrModule
from waasmodel_v6.module_drought_shp import DShpModule
from waasmodel_v6.module_measure import MeasureModule
from waasmodel_v6.module_flood_update_system import UpdateGeneralsystem
from waasmodel_v6.inputs_transferdict import TransferDicts

from typing import List


class WaasModel:
    def __init__(self, hazard_sector_pair: str, stage: int, analysis: str, test_all: bool, portfolio_numbers: dict):
        self.hazard_sector_pair = hazard_sector_pair
        self.stage = stage
        self.analysis = analysis
        self.test_all = test_all
        self.inputs = ModelInputs(stage)
        self.transfer_dict = TransferDicts(inputs=self.inputs)

        self.update_system = UpdateGeneralsystem()
        self.f_module = FloodModule(stage=self.stage, analysis=self.analysis, test_all=self.test_all,
                                    inputs=self.inputs)
        self.d_agr_module = DAgrModule(stage=self.stage, analysis=self.analysis, inputs=self.inputs)
        self.d_shp_module = DShpModule(stage=self.stage, analysis=self.analysis, test_all=self.test_all)
        self.measure_module = MeasureModule(stage=self.stage, hazard_sector_pair=self.hazard_sector_pair,
                                            portfolio_numbers=portfolio_numbers, inputs=self.inputs,
                                            module_transfer=self.transfer_dict)

    def implement_measure_once(self, measure_to_implement: List[str]):
        if measure_to_implement != ['no_measure']:
            _, _, transfer_dict = self.measure_module.implement_measure(inputs=self.inputs,
                                                                        mportfolio=measure_to_implement,
                                                                        pathways_list='current_conditions',
                                                                        module_transfer=self.transfer_dict)

    def run_model(self, scenario_index: int, timestep_index: int):
        transfer_dict = self.transfer_dict

        if timestep_index % 36 == 0:
            transfer_dict = self.measure_module.run_model(timestep=timestep_index, module_transfer=transfer_dict,
                                                          inputs=self.inputs)


        if timestep_index % (36 * 20) == 0: # all 20 years
            # pass
            transfer_dict = self.update_system.run_module(module_transfer=transfer_dict, timestep=timestep_index,
                                                          inputs=self.inputs)

        if (self.hazard_sector_pair in ['flood_urb', 'flood_agr']) or (self.stage in [2, 3]):
            transfer_dict = self.f_module.run_module(timestep=timestep_index, inputs=self.inputs,
                                                     scenario=scenario_index, module_transfer=transfer_dict)

        if (self.hazard_sector_pair == 'drought_agr') or (self.stage in [2, 3]):
            transfer_dict = self.d_agr_module.run_module(timestep=timestep_index, scenario=scenario_index,
                                                         module_transfer=transfer_dict, inputs=self.inputs)

        if (self.hazard_sector_pair == 'drought_shp') or (self.stage == 3):
            transfer_dict = self.d_shp_module.run_module(timestep=timestep_index, scenario=scenario_index,
                                                         module_transfer=transfer_dict, inputs=self.inputs)

        return transfer_dict
