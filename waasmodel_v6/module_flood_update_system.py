import random
import numpy as np
from waasmodel_v6.helperfunctions import urban_growth
from pcraster import *


class UpdateGeneralsystem:
    def __init__(self):
        """
        Updates land use map and corresponding exposure. Reset some parameters at start of each year.
        """
        self.urb_scenario = random.randint(0, 10)
        self.urb_scenario = 1

    def run_module(self, timestep, module_transfer, inputs):
        urb_growth = self.initiate_timeinputs(timestep, inputs)
        urb_growth = 0

        module_transfer.lost_cropland += np.maximum(0, urb_growth)
        landuse, agri_areas, no_constr = urban_growth(growth=urb_growth, LUmap=module_transfer.LandUse, no_constr=module_transfer.no_constr)
        expos = lookupscalar(inputs.MaxDamTbl, landuse)  # max damage per LU
        # derive total agricultural area in the Waasregion
        agri_areas = maptotal(ifthen(landuse == 9, scalar(1)))  # in [ha]!
        agri_areas = pcr2numpy(agri_areas, -999)[0, 0]
        self.create_output_dicts(landuse, expos, agri_areas, no_constr, module_transfer)
        return module_transfer

    def initiate_timeinputs(self, timestep, inputs):
        urb_growth = inputs.urb_scen[int(timestep / 36 / 20 - 1), self.urb_scenario]
        return urb_growth

    def create_output_dicts(self, landuse, expos, agri_areas, no_constr, module_transfer):
        module_transfer.LandUse = landuse
        module_transfer.no_constr = no_constr
        module_transfer.expos = expos
        module_transfer.agri_areas = agri_areas
