import pandas as pd
import os

class RunClassInputs:
    def __init__(self):
        stages = [1, 2, 3]
        sectors = {1: ['flood_agr', 'flood_urb', 'drought_agr', 'drought_shp'],
                   2: ['multihaz_agr', 'multihaz_urb'],
                   3: ['multihaz_multisec']}
        script_path = os.path.abspath(__file__)
        # Get the parent folder of the script
        parent_folder = os.path.dirname(script_path)
        # Get the parent folder's parent folder
        grandparent_folder = os.path.dirname(parent_folder)
        ClimQmax = pd.read_csv(f'{parent_folder}/inputs/discharge_max_decade_new.tss')
        self.realization_names = list(ClimQmax.columns)

        self.keys_dict = {}
        for stage in stages:
            self.keys_dict[stage] = {}
            for sector in sectors[stage]:
                self.keys_dict[stage][sector] = {}
                for analysis in ['out_only', 'analysis']:
                    self.keys_dict[stage][sector][analysis] = {}

        self.keys_dict[1]['flood_agr']['out_only'] = ['f_a_decision_value', 'DamAgr_f_tot', 'pathways_list_f_a',
                                                      'revenue_agr']
        # self.keys_dict[1]['flood_agr']['analysis'] = ['f_a_decision_value', 'DikeFails_tot', 'DamAgr_f_tot',
        #                                               'lost_cropland', 'pathways_list_f_a',
        #                                               'DamAgr_f', 'DikeFails', 'wlvl', 'fragdike', 'floodtiming',
        #                                               'ClimQmax', 'sprink_dikes', 'revenue_agr']
        self.keys_dict[1]['flood_agr']['analysis'] = ['f_a_decision_value', 'DamAgr_f','pathways_list_f_a', 'DikeFails', 'wlvl', 'fragdike',
                                                      'ClimQmax', 'sprink_dikes']
        self.keys_dict[1]['flood_urb']['out_only'] = ['f_u_decision_value', 'DamUrb_tot',
                                                      'pathways_list_f_u', ]
        self.keys_dict[1]['flood_urb']['analysis'] = ['f_u_decision_value', 'DikeFails_tot', 'DamUrb_tot',
                                                      'lost_cropland', 'pathways_list_f_u',
                                                      'DamUrb_f', 'DikeFails', 'wlvl', 'fragdike', 'floodtiming',
                                                      'ClimQmax', 'sprink_dikes']
        self.keys_dict[1]['drought_agr']['out_only'] = ['d_a_decision_value', 'DamAgr_d_tot',
                                                        'pathways_list_d_a', 'revenue_agr', 'sprink_riv', 'sprink_gw']
        self.keys_dict[1]['drought_agr']['analysis'] = ['d_a_decision_value', 'DamAgr_d_tot',
                                                        'pathways_list_d_a','caprise',
                                                        'collectwater', 'epot', 'eact', 'soilm', 'groundwlvl',
                                                        'percol', 'wbalance', 'survfrac', 'runoff', 'sprink',
                                                        'damfrac_tot_d', 'yact', 'ypot', 'revenue_agr']
        self.keys_dict[1]['drought_agr']['analysis'] = ['d_a_decision_value', 'DamAgr_d_tot',
                                                        'pathways_list_d_a', 'caprise',
                                                        'collectwater', 'epot', 'eact', 'soilm', 'groundwlvl',
                                                        'percol', 'wbalance', 'survfrac', 'runoff', 'sprink',
                                                        'damfrac_tot_d', 'yact', 'ypot', 'revenue_agr']
        self.keys_dict[1]['drought_shp']['out_only'] = ['d_s_decision_value', 'DamShp_tot',
                                                        'pathways_list_d_s', ]
        self.keys_dict[1]['drought_shp']['analysis'] = ['d_s_decision_value', 'DamShp_tot',
                                                        'pathways_list_d_s', 'DamShp',
                                                        'load_perc', 'ClimShip', 'wdepth_min']
        self.keys_dict[2]['multihaz_agr']['out_only'] = ['f_a_decision_value', 'DamAgr_f_tot', 'pathways_list_f_a',
                                                        'd_a_decision_value', 'DamAgr_d_tot',
                                                        'pathways_list_d_a', 'revenue_agr', 'sprink_riv', 'sprink_gw']
        self.keys_dict[2]['multihaz_agr']['analysis'] = ['f_a_decision_value', 'DamAgr_f', 'pathways_list_f_a',
                                                      'DikeFails', 'wlvl', 'fragdike',
                                                      'ClimQmax', 'sprink_dikes']
        # self.keys_dict[2]['multihaz_agr']['analysis'] = ['f_a_decision_value', 'd_a_decision_value', 'DikeFails_tot',
        #                                                  'DamAgr_f_tot',
        #                                                  'lost_cropland', 'pathways_list_f_a',
        #                                                  'DamAgr_f', 'DikeFails', 'wlvl', 'fragdike', 'floodtiming',
        #                                                  'ClimQmax', 'sprink_dikes', 'DamAgr_d', 'caprise',
        #                                                  'collectwater',
        #                                                  'epot', 'eact', 'eratio', 'soilm', 'groundwlvl', 'percol',
        #                                                  'wbalance', 'survfrac', 'runoff', 'sprink', 'damfrac_t_d',
        #                                                  'damfrac_tot_d', 'yact', 'ypot', 'revenue_agr']
        self.keys_dict[2]['multihaz_urb']['out_only'] = ['f_u_decision_value', 'DamUrb_tot',
                                                      'pathways_list_f_u']
        self.keys_dict[2]['multihaz_urb']['analysis'] = ['f_u_decision_value', 'DikeFails_tot', 'DamUrb_tot',
                                                         'lost_cropland', 'pathways_list_f_u',
                                                         'DamUrb_f', 'DikeFails', 'wlvl', 'fragdike', 'floodtiming',
                                                         'ClimQmax', 'sprink_dikes']

        self.keys_dict[3]['multihaz_multisec']['out_only'] = ['f_a_decision_value', 'DamAgr_f_tot', 'pathways_list_f_a',
                                                      'd_a_decision_value', 'DamAgr_d_tot',
                                                        'pathways_list_d_a', 'revenue_agr', 'f_u_decision_value', 'DamUrb_tot',
                                                      'pathways_list_f_u','d_s_decision_value', 'DamShp_tot',
                                                        'pathways_list_d_s', 'sprink_riv', 'sprink_gw']
        # self.keys_dict[3]['multihaz_multisec']['analysis'] = ['f_a_decision_value', 'd_a_decision_value',
        #                                                       'f_u_decision_value', 'd_s_decision_value',
        #                                                       'DikeFails_tot', 'DamAgr_f_tot',
        #                                                       'lost_cropland',
        #                                                       'pathways_list_f_a', 'DamAgr_f',
        #                                                       'DikeFails', 'wlvl',
        #                                                       'fragdike', 'floodtiming', 'ClimQmax', 'sprink_dikes',
        #                                                       'DamUrb_tot',
        #                                                       'pathways_list_f_u',
        #                                                       'DamUrb_f',
        #                                                       'DamShp_tot',
        #                                                       'pathways_list_d_s',
        #                                                       'DamShp', 'load_perc',
        #                                                       'ClimShip', 'wdepth_min',
        #                                                       'DamAgr_d_tot',
        #                                                       'pathways_list_d_a',
        #                                                       'DamAgr_d', 'caprise',
        #                                                       'collectwater', 'epot', 'eact',
        #                                                       'eratio', 'soilm', 'groundwlvl', 'percol', 'wbalance',
        #                                                       'survfrac', 'runoff',
        #                                                       'sprink', 'damfrac_t_d', 'damfrac_tot_d', 'yact', 'ypot',
        #                                                       'revenue_agr']
        self.keys_dict[3]['multihaz_multisec']['analysis'] = ['f_a_decision_value', 'DamAgr_f', 'pathways_list_f_a',
                                                         'DikeFails', 'wlvl', 'fragdike',
                                                         'ClimQmax', 'sprink_dikes']

