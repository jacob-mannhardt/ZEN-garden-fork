from plotting.helpers import *

def work_over_opex_df(df):
    df = df.rename(index=lambda x: 'hydro' if 'hydro' in x else x)
    df = df.rename(index=lambda x: 'wind' if 'wind' in x else x)
    df = df.rename(index=lambda x: x.replace('_DH', '') if '_DH' in x else x)
    df = df.groupby(level=['scenario', 'technology']).sum()

    return df

def to_carrier(df):
    df = df.rename(index=lambda x: x.replace('_plant', '') if '_plant' in x else x)
    df = df.rename(index=lambda x: x.replace('_turbine', '') if '_turbine' in x else x)
    df = df.rename(index=lambda x: 'lignite' if 'lignite_coal' in x else x)
    df = df.groupby(level=['scenario', 'technology']).sum()
    return df

def work_over_carrier_df(df):
    df = df.drop(['passenger_mileage', 'natural_gas_industry'], level='carrier')
    df = df.rename(index=lambda x: 'wind' if 'wind' in x else x)
    df = df.rename(index=lambda x: x.replace('_DH', '') if '_DH' in x else x)
    df = df.groupby(level=['scenario', 'technology']).sum()
    return df

def single_decision_plot():
    opex = r.get_total("cost_opex") #"cost_opex_total", "cost_carrier_total",
    opex = indexmapping(opex, special_model_name=model_name)
    opex.index = opex.index.rename('scenario', level=0)
    opex = opex.groupby(level=['scenario', 'technology']).sum()

    opex_5 = opex[opex.index.get_level_values('scenario') == '15_5']
    opex_1 = opex[opex.index.get_level_values('scenario') == '15_1']
    opex_5 = work_over_opex_df(opex_5)
    opex_1 = work_over_opex_df(opex_1)

    carrier_cost = r.get_total("cost_carrier")
    carrier_cost = indexmapping(carrier_cost, special_model_name=model_name)
    carrier_cost.index = carrier_cost.index.rename('scenario', level=0)
    carrier_cost = carrier_cost.groupby(['scenario', 'carrier']).sum()

    carrier_cost_5 = carrier_cost[carrier_cost.index.get_level_values('scenario') == '15_5']
    carrier_cost_1 = carrier_cost[carrier_cost.index.get_level_values('scenario') == '15_1']

    transport_sector = ["BEV", "HEV", "ICE_diesel", "ICE_petrol", "PHEV"]
    heat_sector = ["biomass_boiler", "district_heating_grid", "electrode_boiler", "hard_coal_boiler",
                   "heat_pump", "natural_gas_boiler", "oil_boiler", "waste_boiler"]
    electricity_sector = ["biomass_plant", "hard_coal_plant", "lignite_coal_plant", "natural_gas_turbine",
                          "nuclear", "oil_plant", "photovoltaics", "hydro"  "waste_plant", "wind"]


    sector = electricity_sector
    plot_df_1 = opex_1.loc[opex_1.index.get_level_values('technology').isin(sector)]
    plot_df_1 = to_carrier(plot_df_1)

    plot_df_5 = opex_5.loc[opex_5.index.get_level_values('technology').isin(sector)]

    print("done")
    pass





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# model_name = "PC_ct_pass_tra_base"
# model_name = "PC_ct_pass_tra_ETS1only"
# model_name = "PC_ct_pass_tra_ETS1and2"
# model_name = ("PC_ct_pass_tra_cap_add_target")
# model_name = ("PC_ct_pass_tra_gen_target")
# model_name = ("PC_ct_pass_tra_coal_cap_phaseout")
# model_name = ("PC_ct_pass_tra_coal_gen_phaseout")

model_names = ["PC_ct_pass_tra_base"] # "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_coal_cap_phaseout", "PC_ct_pass_tra_coal_gen_phaseout"
sectors = ["electricity", "heat", "passenger_mileage"] #  "electricity", "heat", "passenger_mileage"

for model_name in model_names:
    r, config = read_in_special_results(model_name)

    for sector in sectors:
        single_decision_plot()


