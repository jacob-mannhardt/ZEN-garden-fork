"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      May-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  ETH colors for plots
==========================================================================================================================================================================="""
import cmcrameri.cm as cm
import numpy as np
from matplotlib.colors import ListedColormap

class ETHColors:

    def __init__(self):
        self.color_scheme = "eth" # "eth" or "cm", cm is by Fabio Crameri https://www.fabiocrameri.ch/colourmaps/
        self.cm_to_white = False
        self.overwrite_eth = True
        # load ETH colors
        self.load_colors()
        # set colors
        self.setColors()

    def retrieveColors(self, inputComponents, category):
        assert category in self.colors.keys(), f"category {category} not known. Currently categories {list(self.colors.keys())} available."
        _listColors = []
        if type(inputComponents) == "str":
            _listColors.append(self.retrieveSpecificColor(inputComponents, category))
        else:
            for component in inputComponents:
                _listColors.append(self.retrieveSpecificColor(component, category))
        return _listColors

    def retrieveSpecificColor(self, component, category):
        if component in self.colors[category]:
            _color = self.colors[category][component]
        elif component in self.manualColors:
            _color = self.manualColors[component]
        else:
            print(f"component {component} is neither in colors of category {category} nor in manual colors. Set to blue")
            # _color = self.getColor("blue")
            _color = np.random.rand(3)
        return _color

    def setColors(self):
        self.colors = {}
        self.setColorsCosts()
        self.setColorsTechs()
        self.setColorsScenarios()
        self.setManualColors()

    def setColorsCosts(self):
        self.colors["costs"] = {"cost_total": self.getColor("blue"), "capex_total": self.getColor("petrol"), "opex_total": self.getColor("bronze"), "cost_carrier_total": self.getColor("red"),
            "cost_carbon_emissions_total": self.getColor("purple"), }

    def setColorsTechs(self):
        self.colors["techs"] = {
            "natural_gas_turbine": self.getColor("petrol"),
            "natural_gas": self.getColor("petrol"),
            "Natural gas turbine": self.getColor("petrol"),
            "hard_coal_plant": self.getColor("grey","dark"),
            "Hard coal plant": self.getColor("grey","dark"),
            "hard_coal_boiler": self.getColor("grey","dark"),
            "hard_coal_boiler_DH": self.getColor("grey",80),
            "natural_gas_turbine_CCS": self.getColor("petrol",60),
            "hard_coal_plant_CCS": self.getColor("grey",60),
            "coal": self.getColor("bronze","dark"),
            "Coal": self.getColor("bronze","dark"),
            "nuclear": self.getColor("purple",80),
            "Nuclear": self.getColor("purple",80),
            "lignite_coal_plant": self.getColor("bronze","dark"),
            "Lignite coal plant": self.getColor("bronze","dark"),
            "oil_plant": self.getColor("grey"),
            "oil": self.getColor("grey"),
            "lng_terminal": self.getColor("grey",40),
            "wind_onshore": self.getColor("blue",80),
            "wind": self.getColor("blue",60),
            "Wind": self.getColor("blue",60),
            "wind_solar": self.getColor("blue",60),
            "solar_PV": self.getColor("bronze",60),
            "solar_photovoltaics": self.getColor("bronze",60),
            "photovoltaics": self.getColor("bronze",60),
            "wind_offshore": self.getColor("blue","dark"),
            "biomass_plant": self.getColor("green"),
            "Biomass plant": self.getColor("green"),
            "biomass_boiler": self.getColor("green"),
            "biomass_boiler_DH": self.getColor("green",60),
            "biomass_plant_CCS": self.getColor("green",60),
            "CCS": self.getColor("red"),
            "waste_plant": self.getColor("grey",60),
            "waste": self.getColor("purple",60),
            "waste_boiler_DH": self.getColor("grey",60),
            "run-of-river_hydro": self.getColor("lavender",80),
            "reservoir_hydro": self.getColor("lavender",60),
            "hydro": self.getColor("lavender",60),
            "Hydro": self.getColor("blue"),
            "renewables": self.getColor("green", 60),
            "heat_pump": self.getColor("blue"),
            "heat_pump_DH": self.getColor("blue",60),
            "natural_gas_boiler": self.getColor("petrol",60),
            "natural_gas_boiler_DH": self.getColor("petrol",40),
            "oil_boiler": self.getColor("bronze",60),
            "oil_boiler_DH": self.getColor("bronze",20),
            "electrode_boiler": self.getColor("lavender"),
            "electrode_boiler_DH": self.getColor("lavender",60),
            "district_heating_grid": self.getColor("grey",60),
            "electrified_heating": self.getColor("green",80),
            "fossil-based_heating": self.getColor("red", 60),
            "others": self.getColor("red",60),
            "Others": self.getColor("red",60),
            "battery": self.getColor("yellow",100),
            "hydrogen_storage": self.getColor("purple",60),
            "natural_gas_storage": self.getColor("petrol",60),
            "pumped_hydro": self.getColor("blue",60),
            "storage": self.getColor("lavender",60),
            "power_line": self.getColor("grey",60),
            "natural_gas_pipeline": self.getColor("petrol",60),
            "biomass_to_cement_fuel": self.getColor("green"),
            "waste_to_cement_fuel": self.getColor("grey"),
            "coal_to_cement_fuel": self.getColor("bronze","dark"),
            "hydrogen_to_cement_fuel": self.getColor("blue",40),
            "DAC": self.getColor("grey",60),
            "steel": self.getColor("blue",60),
            "cement": self.getColor("bronze",60),
            "passenger_transport": self.getColor("blue",60),
            "truck_transport": self.getColor("petrol",60),
            "aviation": self.getColor("lavender",60),
            "shipping": self.getColor("bronze",60),
            # aggregated
            "chemicals": self.getColor("grey",60),
            "aviation_shipping": self.getColor("bronze",100),
            "Aviation+Shipping": self.getColor("bronze",100),
            "road_transport": self.getColor("bronze",60),
            "road transport": self.getColor("bronze",60),
            "cement_steel": self.getColor("red",60),
            "Cement+Steel": self.getColor("lavender",100),
            "hydrogen": self.getColor("lavender",60),
            "heat": self.getColor("red",80),
            "electricity": self.getColor("blue",100),
            "renewable": self.getColor("green",60),
            "fossil": self.getColor("grey",100),
            "Electricity and Heat": self.getColor("lavender",80),
            "Transport": self.getColor("bronze",80),
            "Industry": self.getColor("green",80),
        }

    def setColorsScenarios(self):
        self.colors["scenarios"] = {
            # short
            "short0": self.getColor("blue",60),
            "short100": self.getColor("blue","dark"),
            "short100noPer": self.getColor("bronze",60),
            "short100noStor": self.getColor("bronze","dark"),
            # new
            # "ref_PI": self.getColor("green"), "ref_PC": self.getColor("blue"), "ref_MC": self.getColor("red"), "ref_MI": self.getColor("petrol"),
            "PI": self.getColor("green"),
            "PI_previous": self.getColor("green",60),
            "PI_old": self.getColor("green",60),
            "PI_no_lim": self.getColor("bronze"),
            "PI_CAC": self.getColor("green", 60),
            "PC": self.getColor("blue"),
            "PC_old": self.getColor("blue",60),
            "PC_tdr_med": self.getColor("blue"),
            "PC_tdr_low": self.getColor("blue","dark"),
            "PC_tdr_high": self.getColor("blue",60),
            # "PC_med_history": self.getColor("green"),
            # "PC_med_equality": self.getColor("blue"),
            # "PC_med_equity": self.getColor("red"),
            # "PC_high_history": self.getColor("green","dark"),
            # "PC_high_equality": self.getColor("blue","dark"),
            # "PC_high_equity": self.getColor("red","dark"),
            # "PC_superhigh_history": self.getColor("green",60),
            # "PC_superhigh_equality": self.getColor("blue",60),
            # "PC_superhigh_equity": self.getColor("red",60),
            "MI_old": self.getColor("petrol",60),
            "MC_old": self.getColor("red",60),
            "MC": self.getColor("red"),
            "MC_previous": self.getColor("red",60),
            "MC_dep1": self.getColor("red",80),
            "MC_dep2": self.getColor("red",40),
            "MC_long1_med": self.getColor("purple"),
            "MC_long2_med": self.getColor("purple"),
            "MC_long3_med": self.getColor("blue"),
            "MC_long4_med": self.getColor("bronze"),
            "MC_long5_med": self.getColor("green"),
            "MC_long1_low": self.getColor("purple","dark"),
            "MC_long2_low": self.getColor("purple","dark"),
            "MC_long3_low": self.getColor("blue","dark"),
            "MC_long4_low": self.getColor("bronze","dark"),
            "MC_long5_low": self.getColor("green","dark"),
            "MC_long1_high": self.getColor("purple",60),
            "MC_long2_high": self.getColor("purple",60),
            "MC_long3_high": self.getColor("blue",60),
            "MC_long4_high": self.getColor("bronze",60),
            "MC_long5_high": self.getColor("green",60),
            "MC_2000ts": self.getColor("red","dark"),
            "MC_1000ts": self.getColor("red",80),
            "MC_500ts": self.getColor("red",60),
            "MC_200ts": self.getColor("red",40),
            "MC_1yearint": self.getColor("purple"),
            "MC_CAC_PI": self.getColor("green"),
            "MC_CAC_PC": self.getColor("blue"),
            "MC_CAC_PC_med": self.getColor("blue"),
            "MC_CAC_PC_low": self.getColor("blue","dark"),
            "MC_CAC_PC_high": self.getColor("blue",60),
            "MC_CAC_PI_med": self.getColor("green"),
            "MC_CAC_PI_low": self.getColor("green","dark"),
            "MC_CAC_PI_high": self.getColor("green",60),
            "MC_CAC_PC_old": self.getColor("blue",60),
            "MC_CAC_man_lin": self.getColor("red",40),
            "MC_CAC_man": self.getColor("red"),
            "MC_CAC_man_exp": self.getColor("red","dark"),
            "MI": self.getColor("petrol"),
            #
            "PI_new": self.getColor("green"),
            "PC_new": self.getColor("blue"),
            "MI_new": self.getColor("petrol"),
            "MC_new": self.getColor("red"),
            "PI_min": self.getColor("green",20),
            "PC_min": self.getColor("blue",20),
            "MI_min": self.getColor("petrol",20),
            "MC_min": self.getColor("red",20),
            "min": self.getColor("grey",20),
            "PI_noLR": self.getColor("green",80),
            "PC_noLR": self.getColor("blue",80),
            "MI_noLR": self.getColor("petrol",80),
            "MC_noLR": self.getColor("red",80),
            "noLR": self.getColor("grey",80),
            "PI_max": self.getColor("green","dark"),
            "PC_max": self.getColor("blue","dark"),
            "MI_max": self.getColor("petrol","dark"),
            "MC_max": self.getColor("red","dark"),
            "max": self.getColor("grey","dark"),
            "PI_DR0": self.getColor("green",20),
            "PC_DR0": self.getColor("blue",20),
            "MI_DR0": self.getColor("petrol",20),
            "MC_DR0": self.getColor("red",20),
            "DR0": self.getColor("grey",20),
            "PI_DR3": self.getColor("green",80),
            "PC_DR3": self.getColor("blue",80),
            "MI_DR3": self.getColor("petrol",80),
            "MC_DR3": self.getColor("red",80),
            "DR3": self.getColor("grey",80),
            "PI_DR9": self.getColor("green","dark"),
            "PC_DR9": self.getColor("blue","dark"),
            "MI_DR9": self.getColor("petrol","dark"),
            "MC_DR9": self.getColor("red","dark"),
            "DR9": self.getColor("grey","dark"),
            "PI_DR_scenario_DR_p00_000": self.getColor("green",20),
            "PC_DR_scenario_DR_p00_000": self.getColor("blue",20),
            "MI_DR_scenario_DR_p00_000": self.getColor("petrol",20),
            "MC_DR_scenario_DR_p00_000": self.getColor("red",20),
            "DR_scenario_DR_p00_000": self.getColor("grey",20),
            "PI_DR_scenario_DR_p00_001": self.getColor("green",80),
            "PC_DR_scenario_DR_p00_001": self.getColor("blue",80),
            "MI_DR_scenario_DR_p00_001": self.getColor("petrol",80),
            "MC_DR_scenario_DR_p00_001": self.getColor("red",80),
            "DR_scenario_DR_p00_001": self.getColor("grey",80),
            "PI_DR_scenario_DR_p00_002": self.getColor("green","dark"),
            "PC_DR_scenario_DR_p00_002": self.getColor("blue","dark"),
            "MI_DR_scenario_DR_p00_002": self.getColor("petrol","dark"),
            "MC_DR_scenario_DR_p00_002": self.getColor("red","dark"),
            "DR_scenario_DR_p00_002": self.getColor("grey","dark"),
            "PI_COC_scenario_COC_p00_000": self.getColor("green",20),
            "PC_COC_scenario_COC_p00_000": self.getColor("blue",20),
            "MI_COC_scenario_COC_p00_000": self.getColor("petrol",20),
            "MC_COC_scenario_COC_p00_000": self.getColor("red",20),
            "COC_scenario_COC_p00_000": self.getColor("grey",20),
            "PI_COC_scenario_COC_p00_001": self.getColor("green",40),
            "PC_COC_scenario_COC_p00_001": self.getColor("blue",40),
            "MI_COC_scenario_COC_p00_001": self.getColor("petrol",40),
            "MC_COC_scenario_COC_p00_001": self.getColor("red",40),
            "COC_scenario_COC_p00_001": self.getColor("grey",40),
            "PI_COC_scenario_COC_p00_002": self.getColor("green",80),
            "PC_COC_scenario_COC_p00_002": self.getColor("blue",80),
            "MI_COC_scenario_COC_p00_002": self.getColor("petrol",80),
            "MC_COC_scenario_COC_p00_002": self.getColor("red",80),
            "COC_scenario_COC_p00_002": self.getColor("grey",80),
            "PI_COC_scenario_COC_p00_003": self.getColor("green","dark"),
            "PC_COC_scenario_COC_p00_003": self.getColor("blue","dark"),
            "MI_COC_scenario_COC_p00_003": self.getColor("petrol","dark"),
            "MC_COC_scenario_COC_p00_003": self.getColor("red","dark"),
            "COC_scenario_COC_p00_003": self.getColor("grey","dark"),
            "PI_fuelprices_scenario_0_p00_000": self.getColor("green",40),
            "PC_fuelprices_scenario_0_p00_000": self.getColor("blue",40),
            "MI_fuelprices_scenario_0_p00_000": self.getColor("petrol",40),
            "MC_fuelprices_scenario_0_p00_000": self.getColor("red",40),
            "fuelprices_scenario_0_p00_000": self.getColor("grey",40),
            "PI_fuelprices_scenario_0_p00_001": self.getColor("green","dark"),
            "PC_fuelprices_scenario_0_p00_001": self.getColor("blue","dark"),
            "MI_fuelprices_scenario_0_p00_001": self.getColor("petrol","dark"),
            "MC_fuelprices_scenario_0_p00_001": self.getColor("red","dark"),
            "fuelprices_scenario_0_p00_001": self.getColor("grey","dark"),
            "PI_nosub": self.getColor("green","dark"),
            "PC_nosub": self.getColor("blue","dark"),
            "MI_nosub": self.getColor("petrol","dark"),
            "MC_nosub": self.getColor("red","dark"),
            "nosub": self.getColor("grey","dark"),
            "PI_fullsub": self.getColor("green",80),
            "PC_fullsub": self.getColor("blue",80),
            "MI_fullsub": self.getColor("petrol",80),
            "MC_fullsub": self.getColor("red",80),
            "fullsub": self.getColor("grey",80),
            "PI_fullsuball": self.getColor("green",20),
            "PC_fullsuball": self.getColor("blue",20),
            "MI_fullsuball": self.getColor("petrol",20),
            "MC_fullsuball": self.getColor("red",20),
            "fullsuball": self.getColor("grey",20),
            "PI_carbon_scenario_": self.getColor("green",40),
            "PC_carbon_scenario_": self.getColor("blue",40),
            "MI_carbon_scenario_": self.getColor("petrol",40),
            "MC_carbon_scenario_": self.getColor("red",40),
            "carbon_scenario_": self.getColor("grey",40),
            "PI_carbon_scenario_0": self.getColor("green","dark"),
            "PC_carbon_scenario_0": self.getColor("blue","dark"),
            "MI_carbon_scenario_0": self.getColor("petrol","dark"),
            "MC_carbon_scenario_0": self.getColor("red","dark"),
            "carbon_scenario_0": self.getColor("grey","dark"),
            "PI_fuelprices_scenario_": self.getColor("green",40),
            "PC_fuelprices_scenario_": self.getColor("blue",40),
            "MI_fuelprices_scenario_": self.getColor("petrol",40),
            "MC_fuelprices_scenario_": self.getColor("red",40),
            "fuelprices_scenario_": self.getColor("grey",40),
            "PI_fuelprices_scenario_0": self.getColor("green","dark"),
            "PC_fuelprices_scenario_0": self.getColor("blue","dark"),
            "MI_fuelprices_scenario_0": self.getColor("petrol","dark"),
            "MC_fuelprices_scenario_0": self.getColor("red","dark"),
            "fuelprices_scenario_0": self.getColor("grey","dark"),
            "PI_sr_scenario_0_p00_001": self.getColor("green","dark"),
            "PC_sr_scenario_0_p00_001": self.getColor("blue","dark"),
            "MI_sr_scenario_0_p00_001": self.getColor("petrol","dark"),
            "MC_sr_scenario_0_p00_001": self.getColor("red","dark"),
            "sr_scenario_0_p00_001": self.getColor("grey","dark"),
            "PI_sr_scenario_0_p00_000": self.getColor("green",40),
            "PC_sr_scenario_0_p00_000": self.getColor("blue",40),
            "MI_sr_scenario_0_p00_000": self.getColor("petrol",40),
            "MC_sr_scenario_0_p00_000": self.getColor("red",40),
            "sr_scenario_0_p00_000": self.getColor("grey",40),
            "PI_hydro_scenario_10": self.getColor("green",20),
            "PC_hydro_scenario_10": self.getColor("blue",20),
            "MI_hydro_scenario_10": self.getColor("petrol",20),
            "MC_hydro_scenario_10": self.getColor("red",20),
            "hydro_scenario_10": self.getColor("grey",20),
            "PI_hydro_scenario_2": self.getColor("green",60),
            "PC_hydro_scenario_2": self.getColor("blue",60),
            "MI_hydro_scenario_2": self.getColor("petrol",60),
            "MC_hydro_scenario_2": self.getColor("red",60),
            "hydro_scenario_2": self.getColor("grey",60),
            "PI_hydro_scenario_029": self.getColor("green",80),
            "PC_hydro_scenario_029": self.getColor("blue",80),
            "MI_hydro_scenario_029": self.getColor("petrol",80),
            "MC_hydro_scenario_029": self.getColor("red",80),
            "hydro_scenario_029": self.getColor("grey",80),
            "PI_bat_scenario_0": self.getColor("green",60),
            "PC_bat_scenario_0": self.getColor("blue",60),
            "MI_bat_scenario_0": self.getColor("petrol",60),
            "MC_bat_scenario_0": self.getColor("red",60),
            "bat_scenario_0": self.getColor("grey",60),
            "PI_var_tdr": self.getColor("green",60),
            "PC_var_tdr": self.getColor("blue",60),
            "MI_var_tdr": self.getColor("petrol",60),
            "MC_var_tdr": self.getColor("red",60),
            "var_tdr": self.getColor("grey",60),
            "PI_demand_scenario_low": self.getColor("green",60),
            "PC_demand_scenario_low": self.getColor("blue",60),
            "MI_demand_scenario_low": self.getColor("petrol",60),
            "MC_demand_scenario_low": self.getColor("red",60),
            "demand_scenario_low": self.getColor("grey",60),
            "PI_demand_scenario_high": self.getColor("green","dark"),
            "PC_demand_scenario_high": self.getColor("blue","dark"),
            "MI_demand_scenario_high": self.getColor("petrol","dark"),
            "MC_demand_scenario_high": self.getColor("red","dark"),
            "demand_scenario_high": self.getColor("grey","dark"),
            # WES
            "WES_nofe": self.getColor("blue",60),
            'WES_fe_tdr_lim_cdr': self.getColor("green",100),
            'WES_fe_tdr_cdr': self.getColor("green",60),
            'WES_fe_lim_cdr_pess': self.getColor("red",60),
            'WES_fe_lim_carbon_pess': self.getColor("red",100),
            # old
            "WES_nofe_long_8": self.getColor("blue",60),
            "WES_nofe_long": self.getColor("blue",60),
            "WES_nofe_long_dh": self.getColor("blue",60),
            "WES_nofe_PC": self.getColor("blue",60),
            "WES_nofe_PI": self.getColor("orange",60),
            "WES_nofe_PC_large": self.getColor("green","dark"),
            "WES_nofe_PI_large": self.getColor("orange","dark"),
            'WES_fe_tdr_lim_cdr_long_8': self.getColor("green",60),
            'WES_fe_tdr_lim_cdr_long': self.getColor("green",60),
            'WES_fe_tdr_lim_cdr_long_dh': self.getColor("green",60),
            'WES_fe_tdr_lim_CATF_cdr': self.getColor("green",60),
            'WES_fe_lim_cdr': self.getColor("purple"),
            'WES_fe_cost_cdr': self.getColor("petrol"),
            'WES_fe_lim_cdr_pess_long_8': self.getColor("red",60),
            'WES_fe_lim_cdr_pess_long': self.getColor("red",60),
            'WES_fe_lim_cdr_pess_long_dh': self.getColor("red",60),
            'WES_fe_cost_cdr_pess': self.getColor("petrol","dark"),
            "WES_nofe_OG": self.getColor("blue",60),
            "WES_fe_lim_cdr_pess_OG": self.getColor("red",60),
            "WES_fe_lim_carbon_pess_OG": self.getColor("red",100),
            "WES_fe_tdr_cdr_OG": self.getColor("green",60),
            "WES_fe_tdr_lim_cdr_OG": self.getColor("green",100),
            "WES_nofe_noct": self.getColor("blue",60),
            "WES_fe_lim_cdr_pess_noct": self.getColor("red",60),
            "WES_fe_lim_carbon_pess_noct": self.getColor("red",100),
            "WES_fe_tdr_cdr_noct": self.getColor("green",60),
            "WES_fe_tdr_lim_cdr_noct": self.getColor("green",100),
            "WES_fe_lim_cdr_pess_op": self.getColor("purple", 60),
            "WES_fe_lim_carbon_pess_op": self.getColor("purple", 100),
            "NoErr": self.getColor("blue",60),
            "OptiAll": self.getColor("green",100),
            "Opti": self.getColor("green",100),
            "Pess": self.getColor("red",100),
            "OptiCDR_noErrCCS": self.getColor("petrol",40),
            "OptiCDR_PessCCS": self.getColor("petrol","dark"),
            "PessCDR_noErrCCS": self.getColor("red",60),
            "OptiCCS_noErrCDR": self.getColor("bronze",40),
            "OptiCCS_PessCDR": self.getColor("bronze","dark"),
            "PessCCS_noErrCDR": self.getColor("purple",60),
            "OptiCDR_PessCCUS": self.getColor("green",60),
            "OptiCDR_PessCCU": self.getColor("petrol",100),
            "OptiCDRS_noErrCCU": self.getColor("lavender",60),
            "OptiCDRS_PessCCU": self.getColor("lavender",100),
            "OptiCDRU_noErrCCS": self.getColor("purple",60),
            "OptiCDRU_PessCCS": self.getColor("purple",100),
            "PessCDRS_noErrCCU": self.getColor("red",100),
            "PessCDRU": self.getColor("red",60),
            "PessCDRS": self.getColor("red",100),
            "PessCDRU_OptiCCS": self.getColor("purple",60),
            "PessCDRS_OptiCCU": self.getColor("purple",100),
            "NoErr_expDAC": self.getColor("blue",60),
            "OptiAll_expDAC": self.getColor("green",100),
            "OptiCDR_noErrCCUS_expDAC": self.getColor("petrol",40),
            "OptiCDR_PessCCUS_expDAC": self.getColor("green",60),
            "OptiCDR_PessCCU_expDAC": self.getColor("petrol",100),
            "OptiCDR_PessCCS_expDAC": self.getColor("petrol","dark"),
            "OptiCDRS_noErrCCU_expDAC": self.getColor("lavender",60),
            "OptiCDRS_PessCCU_expDAC": self.getColor("lavender",100),
            "OptiCDRU_noErrCCS_expDAC": self.getColor("purple",60),
            "OptiCDRU_PessCCS_expDAC": self.getColor("purple",100),
            "PessCDRS_noErrCCS_expDAC": self.getColor("red",60),
            "PessCDRU_noErrCCS_expDAC": self.getColor("red",60),
            "PessCDRS_noErrCCU_expDAC": self.getColor("red",100),
            "NoErr_scen": self.getColor("blue",60),
            "NoErr_scen_scenario_": self.getColor("blue",60),
            "NoErr_scen_scenario_low": self.getColor("blue",40),
            "NoErr_scen_scenario_high": self.getColor("blue",80),
            "OptiAll_scen": self.getColor("green",100),
            "OptiAll_scen_scenario_": self.getColor("green",100),
            "OptiAll_scen_scenario_low": self.getColor("green",80),
            "OptiAll_scen_scenario_high": self.getColor("green","dark"),
            "OptiCDR_PessCCUS_scen": self.getColor("green",60),
            "OptiCDR_PessCCUS_scen_scenario_": self.getColor("green",60),
            "OptiCDR_PessCCUS_scen_scenario_low": self.getColor("green",40),
            "OptiCDR_PessCCUS_scen_scenario_high": self.getColor("green",80),
            "PessCDRS_scen": self.getColor("red",100),
            "PessCDRS_scen_scenario_": self.getColor("red",100),
            "PessCDRS_scen_scenario_low": self.getColor("red",80),
            "PessCDRS_scen_scenario_high": self.getColor("red","dark"),
            "PessCDRU_scen": self.getColor("red",60),
            "PessCDRU_scen_scenario_": self.getColor("red",60),
            "PessCDRU_scen_scenario_low": self.getColor("red",40),
            "PessCDRU_scen_scenario_high": self.getColor("red",80),
            # 2MF study
            "2MF_base": self.getColor("grey"),
            "2MF_base_i15": self.getColor("grey","dark"),
            "2MF_base_i5": self.getColor("grey",60),
            "2MF_base_o5": self.getColor("grey","dark"),
            "2MF_base_o1": self.getColor("grey",60),
            "2MF_base_15_5": self.getColor("grey","dark"),
            "2MF_base_15_1": self.getColor("grey","dark"),
            "2MF_base_5_5": self.getColor("grey",60),
            "2MF_base_5_1": self.getColor("grey",60),
            "2MF_cap_i15": self.getColor("blue","dark"),
            "2MF_cap_i5": self.getColor("blue",60),
            "2MF_cap_o5": self.getColor("blue","dark"),
            "2MF_cap_o1": self.getColor("blue",60),
            "2MF_cap_15_5": self.getColor("blue", "dark"),
            "2MF_cap_15_1": self.getColor("blue", "dark"),
            "2MF_cap_5_5": self.getColor("blue", 60),
            "2MF_cap_5_1": self.getColor("blue", 60),
            "2MF_gen_i15": self.getColor("green","dark"),
            "2MF_gen_i5": self.getColor("green",60),
            "2MF_gen_o5": self.getColor("green","dark"),
            "2MF_gen_o1": self.getColor("green",60),
            "2MF_gen_15_5": self.getColor("green", "dark"),
            "2MF_gen_15_1": self.getColor("green", "dark"),
            "2MF_gen_5_5": self.getColor("green", 60),
            "2MF_gen_5_1": self.getColor("green", 60),
            "2MF_emission_i15": self.getColor("red","dark"),
            "2MF_emission_i5": self.getColor("red",60),
            "2MF_emission_o5": self.getColor("red","dark"),
            "2MF_emission_o1": self.getColor("red",60),
            "2MF_emission_15_5": self.getColor("red","dark"),
            "2MF_emission_15_1": self.getColor("red","dark"),
            "2MF_emission_5_5": self.getColor("red",60),
            "2MF_emission_5_1": self.getColor("red",60),
            "2MF_generic_15_5": self.getColor("grey","dark"),
            "2MF_generic_15_1": self.getColor("grey","dark"),
            "2MF_generic_5_5": self.getColor("grey",60),
            "2MF_generic_5_1": self.getColor("grey",60),
        }

    def setManualColors(self):
        self.manualColors = {}
        self.manualColors["Carbon emission budget"]             = self.getColor("grey","dark")
        self.manualColors["Carbon Intensity"]                   = self.getColor("BW", 100)
        self.manualColors["Final Cumulative Costs"]             = self.getColor("blue")
        self.manualColors["Final Cumulative NPC"]               = self.getColor("blue")
        self.manualColors["Maximum New Electricity Capacities"] = self.getColor("red")
        self.manualColors["Spillover Rate Impact"]              = [self.getColor("red"), self.getColor("green")]
        self.manualColors["Reduced electrified heat"]           = self.getColor("green",60)
        self.manualColors["Reduced electricity demand"]         = self.getColor("red","dark")
        self.manualColors["electricity"]                        = self.getColor("blue")
        self.manualColors["heat"]                               = self.getColor("red")

    def getColor(self, color, shade=100):
        assert color in self.baseColors, f"color {color} not in base colors. Select from {list(self.baseColors.keys())}."
        assert shade in self.baseColors[color], f"shade {shade} not in shades of color {color}. Select from {list(self.baseColors[color].keys())}."
        return self.hex2rgb(self.baseColors[color][shade])

    def load_colors(self):
        if self.color_scheme == "eth":
            self.load_colors_eth()
        elif self.color_scheme == "cm":
            self.load_colors_cm()
        else:
            raise KeyError(f"color scheme {self.color_scheme} not known.")
    def load_colors_eth(self):
        self.baseColors = {}
        self.baseColors["blue"] = {100: "#215CAF", 80: "#4D7DBF", 60: "#7A9DCF", 40: "#A6BEDF", 20: "#D3DEEF", 10: "#E9EFF7", "dark": "#08407E", }
        self.baseColors["petrol"] = {100: "#007894", 80: "#3395AB", 60: "#66AFC0", 40: "#99CAD5", 20: "#CCE4EA", 10: "#E7F4F7", "dark": "#00596D", }
        self.baseColors["green"] = {100: "#627313", 80: "#818F42", 60: "#A1AB71", 40: "#C0C7A1", 20: "#E0E3D0", 10: "#EFF1E7", "dark": "#365213", }
        self.baseColors["bronze"] = {100: "#8E6713", 80: "#A58542", 60: "#BBA471", 40: "#D2C2A1", 20: "#E8E1D0", 10: "#F4F0E7", "dark": "#956013", }
        self.baseColors["yellow"] = {100: "#8E6713", 80: "#A58542", 60: "#BBA471", 40: "#D2C2A1", 20: "#E8E1D0",10: "#F4F0E7", "dark": "#956013", }
        self.baseColors["orange"] = {100: "#8E6713", 80: "#A58542", 60: "#BBA471", 40: "#D2C2A1", 20: "#E8E1D0",10: "#F4F0E7", "dark": "#956013", }
        self.baseColors["red"] = {100: "#B7352D", 80: "#C55D57", 60: "#D48681", 40: "#E2AEAB", 20: "#F1D7D5", 10: "#F8EBEA", "dark": "#96272D", }
        self.baseColors["purple"] = {100: "#A30774", 80: "#B73B92", 60: "#CA6CAE", 40: "#DC9EC9", 20: "#EFD0E3", 10: "#F8E8F3", "dark": "#8C0A59", }
        self.baseColors["lavender"] = {100: "#A30774", 80: "#B73B92", 60: "#CA6CAE", 40: "#DC9EC9", 20: "#EFD0E3", 10: "#F8E8F3", "dark": "#8C0A59", }
        self.baseColors["grey"] = {100: "#6F6F6F", 80: "#8C8C8C", 60: "#A9A9A9", 40: "#C5C5C5", 20: "#E2E2E2", 10: "#F1F1F1", "dark": "#575757", }
        self.baseColors["BW"] = {100: "#000000", 80: "#8C8C8C", 60: "#A9A9A9", 40: "#C5C5C5", 20: "#E2E2E2", 10: "#F1F1F1", 0: "#FFFFFF"}
        # overwrite
        if self.overwrite_eth:
            self.baseColors["lavender"] = self.extract_from_cm(cm.oleron, c120=0.01, c100=0.1, c80=0.2, c60=0.27, c40=0.32, c20=0.4,c10=0.45)
            self.baseColors["bronze"] = self.extract_from_cm(cm.fes,c120= 0.67,c100= 0.72,c80= 0.77,c60=0.83,c40=0.88,c20=0.94,c10=0.99)
            self.baseColors["purple"] = self.extract_from_cm(cm.bam, c120=0.01, c100=0.07, c80=0.15, c60=0.25, c40=0.32, c20=0.4,c10=0.45)
            self.baseColors["orange"] = self.extract_from_cm(cm.lajolla,c120= 0.45,c100= 0.4,c80= 0.32,c60=0.25,c40=0.15,c20=0.07,c10=0.01)
            self.baseColors["yellow"] = self.create_colordict(cm.bamako(0.8),N=121)

    def load_colors_cm(self):
        self.baseColors = {}
        if self.cm_to_white:
            blue = self.create_colordict(cm.roma(0.95),N=121)
            petrol = self.create_colordict(cm.batlow(0.2),N=121)
            green = self.create_colordict(cm.cork(0.9),N=121)
            bronze = self.create_colordict(cm.lisbon(0.65),N=121)
            yellow = self.create_colordict(cm.bamako(0.8),N=121)
            orange = self.create_colordict(cm.roma(0.17),N=121)
            red = self.create_colordict(cm.vik(0.9),N=121)
            purple = self.create_colordict(cm.bam(0),N=121)
            lavender = self.create_colordict(cm.oleron(0),N=121)
            grey = self.create_colordict(cm.turku(0.1),N=121)
            BW = self.create_colordict(cm.grayC(0.9),N=121)
        else:
            # blue = self.extract_from_cm(cm.roma,c120= 0.99,c100= 0.93,c80= 0.85,c60=0.75,c40=0.68,c20=0.6,c10=0.55)
            blue = self.extract_from_cm(cm.vik,c120= 0.05,c100= 0.13,c80= 0.2,c60=0.25,c40=0.32,c20=0.4,c10=0.45)
            petrol = self.extract_from_cm(cm.nuuk,c120= 0.01,c100= 0.07,c80= 0.15,c60=0.25,c40=0.32,c20=0.4,c10=0.45)
            # petrol = self.extract_from_cm(cm.bukavu,c120= 0.01,c100= 0.07,c80= 0.15,c60=0.25,c40=0.32,c20=0.4,c10=0.45)
            # petrol = self.extract_from_cm(cm.batlow,c120= 0.01,c100= 0.07,c80= 0.12,c60=0.17,c40=0.23,c20=0.28,c10=0.33)
            green = self.extract_from_cm(cm.cork,c120= 0.99,c100= 0.93,c80= 0.85,c60=0.75,c40=0.68,c20=0.6,c10=0.55)
            # bronze = self.extract_from_cm(cm.broc,c120= 0.99,c100= 0.93,c80= 0.85,c60=0.75,c40=0.68,c20=0.6,c10=0.55)
            bronze = self.extract_from_cm(cm.fes,c120= 0.67,c100= 0.72,c80= 0.77,c60=0.83,c40=0.88,c20=0.94,c10=0.99)
            # yellow = self.extract_from_cm(cm.bamako,c120= 0.67,c100= 0.72,c80= 0.77,c60=0.83,c40=0.88,c20=0.94,c10=0.99)
            yellow = self.create_colordict(cm.bamako(0.8),N=121)
            orange = self.extract_from_cm(cm.lajolla,c120= 0.45,c100= 0.4,c80= 0.32,c60=0.25,c40=0.15,c20=0.07,c10=0.01)
            # orange = self.extract_from_cm(cm.roma, c120=0.01, c100=0.07, c80=0.15, c60=0.25, c40=0.32, c20=0.4,c10=0.45)
            red = self.extract_from_cm(cm.vik,c120= 0.99,c100= 0.93,c80= 0.85,c60=0.75,c40=0.68,c20=0.6,c10=0.55)
            purple = self.extract_from_cm(cm.bam, c120=0.01, c100=0.07, c80=0.15, c60=0.25, c40=0.32, c20=0.4,c10=0.45)
            lavender = self.extract_from_cm(cm.oleron, c120=0.01, c100=0.1, c80=0.2, c60=0.27, c40=0.32, c20=0.4,c10=0.45)
            # lavender = self.extract_from_cm(cm.acton, c120=0.01, c100=0.1, c80=0.2, c60=0.27, c40=0.32, c20=0.4,c10=0.45)
            # grey = self.extract_from_cm(cm.turku, c120=0.01, c100=0.07, c80=0.15, c60=0.25, c40=0.32, c20=0.4,c10=0.45)
            grey = self.extract_from_cm(cm.grayC, c120=0.9, c100=0.8, c80=0.64, c60=0.5, c40=0.3, c20=0.14,c10=0.01)
            BW = self.extract_from_cm(cm.grayC, c120=0.9, c100=0.8, c80=0.64, c60=0.5, c40=0.3, c20=0.14,c10=0.01)
        self.baseColors["blue"] = blue
        self.baseColors["petrol"] = petrol
        self.baseColors["green"] = green
        self.baseColors["bronze"] = bronze
        self.baseColors["yellow"] = yellow
        self.baseColors["orange"] = orange
        self.baseColors["red"] = red
        self.baseColors["purple"] = purple
        self.baseColors["lavender"] = lavender
        self.baseColors["grey"] = grey
        self.baseColors["BW"] = BW

    def extract_from_cm(self,cm,c120,c100,c80,c60,c40,c20,c10):
        """ extracts values from cm """
        cm_dict = {}
        cm_dict["dark"] = cm(c120)
        cm_dict[100] = cm(c100)
        cm_dict[80] = cm(c80)
        cm_dict[60] = cm(c60)
        cm_dict[40] = cm(c40)
        cm_dict[20] = cm(c20)
        cm_dict[10] = cm(c10)
        return cm_dict

    def create_colordict(self, color,N=256):
        """ creates custom colormap and write to dict """
        cm = self.create_colormap(color,N=N).colors
        cdict = {
            "dark": cm[120],
            100: cm[100],
            80: cm[80],
            60: cm[60],
            40: cm[40],
            20: cm[20],
            10: cm[10],
        }
        return cdict

    def create_colormap(self, color,N=256):
        """ creates custom colormap"""
        vals = np.ones((N,4))
        # lowCol = self.ethcolors.getColor("grey",shade=20)
        baseCol = [0.98,0.98,0.98]
        baseCol = [0.999,0.999,0.999]
        # single color
        if isinstance(color, tuple):
            vals[:, 0] = np.linspace(baseCol[0],color[0], N)
            vals[:, 1] = np.linspace(baseCol[1],color[1], N)
            vals[:, 2] = np.linspace(baseCol[2],color[2], N)
        elif isinstance(color, list):
            Ndiv = N/len(color)
            Nidx = 0
            assert int(Ndiv) == Ndiv, f"requested number {len(color)} of colors does not translate to even step size"
            Ndiv = int(Ndiv)
            for idxCol,col in enumerate(color):
                if idxCol%2 == 0:
                    startCol = col
                    endCol = baseCol
                else:
                    startCol = baseCol
                    endCol = col
                vals[Nidx:Nidx+Ndiv, 0] = np.linspace(startCol[0],endCol[0], Ndiv)
                vals[Nidx:Nidx+Ndiv, 1] = np.linspace(startCol[1],endCol[1], Ndiv)
                vals[Nidx:Nidx+Ndiv, 2] = np.linspace(startCol[2],endCol[2], Ndiv)
                Nidx += Ndiv
        else:
            raise TypeError

        cmap = ListedColormap(vals)
        return cmap

    @staticmethod
    def hex2rgb(hexString, normalized=True):
        if type(hexString) != str:
            return hexString
        if normalized:
            _fac = 255
        else:
            _fac = 1
        hexString = hexString.lstrip('#')
        rgb = tuple(int(hexString[i:i + 2], 16) / _fac for i in (0, 2, 4))
        return rgb
