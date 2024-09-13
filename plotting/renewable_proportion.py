from plotting.helpers import *

def ren_capacity_addition_plot(model_name):
    print("Plotting the proportion of renewable capacity addition")
    # Set the color palette from Seaborn
    sns.set_palette('deep')

    r, config = read_in_results(model_name)
    temp_df = r.get_total("capacity_addition")

    # indexing
    if desired_scenarios != "dont_index":
        df = indexmapping(temp_df, special_model_name=model_name)

    df.columns = rename_columns(df.columns)
    # df.index = df.index.rename('scenario', level=0)
    df = df.loc[df.index.get_level_values('capacity_type') == "power"]
    df = df.groupby(['technology']).sum() #  'scenario',

    # technology groups
    system = r.get_system()
    set_conversion_technologies = system['set_conversion_technologies']
    renewable_technologies = system['set_renewable_technologies']

    tech_el = ['biomass_plant', 'hard_coal_plant', 'lignite_coal_plant', 'natural_gas_turbine',
                       'nuclear', 'oil_plant', 'photovoltaics', 'reservoir_hydro', 'run-of-river_hydro',
                       'waste_plant', 'wind_offshore', 'wind_onshore']
    tech_heat =  ['biomass_boiler', 'biomass_boiler_DH', 'electrode_boiler', 'electrode_boiler_DH',
                          'hard_coal_boiler_DH', 'heat_pump', 'heat_pump_DH', 'natural_gas_boiler',
                          'natural_gas_boiler_DH', 'oil_boiler', 'oil_boiler_DH', 'waste_boiler_DH']
    tech_tra = ['BEV', 'HEV', 'ICE_diesel', 'ICE_petrol', 'PHEV']
    remainder = ['district_heating_grid', 'industrial_gas_consumer', 'lng_terminal']
    all_technologies = list(set(tech_el + tech_heat + tech_tra + remainder))
    # assert set(set_conversion_technologies).issubset(set(all_technologies))

    ren_tech_el = list(set(tech_el).intersection(set(renewable_technologies)))
    ren_tech_heat = list(set(tech_heat).intersection(set(renewable_technologies)))
    ren_tech_tra = list(set(tech_tra).intersection(set(renewable_technologies)))

    for scenario in desired_scenarios:

        # get the scenario df
        plot_df = df
        # plot_df = df.loc[df.index.get_level_values('scenario') == scenario]
        # plot_df = plot_df.droplevel('scenario')

        el_add = plot_df.loc[plot_df.index.get_level_values('technology').isin(tech_el)].sum()
        ren_el_add = plot_df.loc[plot_df.index.get_level_values('technology').isin(ren_tech_el)].sum()
        el_series = ren_el_add/el_add

        heat_add = plot_df.loc[plot_df.index.get_level_values('technology').isin(tech_heat)].sum()
        ren_heat_add = plot_df.loc[plot_df.index.get_level_values('technology').isin(ren_tech_heat)].sum()
        heat_series = ren_heat_add/heat_add

        tra_add = plot_df.loc[plot_df.index.get_level_values('technology').isin(tech_tra)].sum()
        ren_tra_add = plot_df.loc[plot_df.index.get_level_values('technology').isin(ren_tech_tra)].sum()
        tra_series = ren_tra_add/tra_add

        # Creating the figure and plotting the data
        plt.figure(figsize=(10, 6))
        plt.plot(el_series.index, el_series, label='Electricity', color='blue')
        plt.plot(el_series.index, heat_series, label='Heat', color='red')
        plt.plot(el_series.index, tra_series, label='Transportation', color='green')

        # Adding title and labels
        plt.title(f"Renewable capacity addition proportion \n{model_name}   Scenario: {scenario}")
        plt.xlabel('Time')
        plt.ylabel('Proportion')

        # Adding legend
        plt.legend()

        # Display the plot
        plt.show()



    pass


def ren_generation_plot():
    pass




















# Defined Models ==================================
desired_scenarios = ['15_1', '15_5', '5_1', '5_5']

# model_name = "PC_ct_pass_tra_base"

# model_name = "PC_ct_pass_tra_ETS1and2"

model_name = "PC_ct_pass_tra_cap_add_target"

# model_name = "PC_ct_pass_tra_gen_target"

# model_name = "PC_ct_test"

# model_name = "PC_ct_test_different"


ren_capacity_addition_plot(model_name)

ren_generation_plot()