from plotting.helpers import *


def el_gen_costs_over_time():
    global model_name
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]

    dfs = []
    for name in model_name:
        r, config = read_in_results(name)
        desired_scenarios = ["15_1", "15_5"]

        costs = ["cost_opex"] # , "cost_carrier"
        df = {}
        for c in costs:
            df[c] = r.get_total(c)
            # Reset the index to work with it as columns
            df[c].reset_index(inplace=True)
            print("reset complete")
            # The 'level_0' column contains the scenario information, so we will rename it
            df[c].rename(columns={'level_0': 'scenario'}, inplace=True)
            # Now we set the multi-index again, this time including the 'scenario'
            if c == "cost_opex":
                df[c].set_index(['scenario', 'technology', 'location'], inplace=True)
                df[c] = df[c].groupby(['scenario', 'technology']).sum()
            elif c == "cost_carrier":
                df[c].set_index(['scenario', 'carrier', 'node'], inplace=True)
                df[c] = df[c].groupby(['scenario', 'carrier']).sum()
            else: raise ValueError("this type of df does not fit here")
            df[c] = indexmapping(df[c], special_model_name=name)
            # reducing to the desired scenarios
            df[c] = df[c].loc[df[c].index.get_level_values('scenario').isin(desired_scenarios)]

            # # List of technologies to remove
            technologies_to_remove = [
                 "biomass_boiler", "biomass_boiler_DH", "carbon_pipeline",
                 "district_heating_grid", "electrode_boiler", "electrode_boiler_DH",
                 "hard_coal_boiler_DH", "heat_pump", "heat_pump_DH", "industrial_gas_consumer", "lng_terminal",
                 "natural_gas_boiler", "natural_gas_boiler_DH", "natural_gas_pipeline", "natural_gas_storage",
                 "hydrogen_storage", "oil_boiler", "oil_boiler_DH", "waste_boiler_DH"] # "hard_coal_boiler", ???
            # list_of_not_others = ['hard_coal_plant', 'hard_coal_plant_CCS', 'lignite_coal_plant',
            #                       'natural_gas_turbine', 'natural_gas_turbine_CCS', 'nuclear', 'biomass_plant',
            #                       'biomass_plant_CCS', 'photovoltaics', 'wind_offshore', 'wind_onshore', 'power_line']
            # list_of_others = ['oil_plant', 'waste_plant']
            # list_of_storage = ['battery']
            # list_of_hydro = ['pumped_hydro', 'reservoir_hydro', 'run-of-river_hydro']
            # Remove rows where technology matches any in the removal list
            df[c].drop(index=technologies_to_remove, level='technology', inplace=True)
            # # list of technologies in storage
            df[c].rename(index={'oil_plant': 'Others', 'waste_plant': 'Others'}, level='technology', inplace=True)
            df[c].rename(index={'battery': 'storage'}, level='technology', inplace=True)
            df[c].rename(index={'pumped_hydro': 'Hydro', 'reservoir_hydro': 'Hydro', 'run-of-river_hydro': 'Hydro'}, level='technology', inplace=True)
            df[c] = df[c].groupby(level=['scenario', 'technology']).sum()

        df = pd.concat(df, keys=df.keys())



        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df_totals = df
    df_totals = df_totals.T.sum()

    # Check if the number of desired scenarios is exactly four
    if len(desired_scenarios) == 4:
        # Set up a 2x2 grid of plots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        legend_labels = {}  # Dictionary to hold unique legend labels without scenario tags and their handles

        for i, scenario in enumerate(desired_scenarios):
            scenario_df = df.xs(scenario, level='scenario', drop_level=False)
            scenario_df = scenario_df.T
            scenario_df.plot(kind='bar', stacked=True, ax=axs[i])

            axs[i].set_title(f'Scenario {scenario}')
            axs[i].set_xlabel('Technology')
            axs[i].set_ylabel('OPEX')

            # Retrieve handles and labels for the current subplot
            handles, labels = axs[i].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                # Extract technology name from the label
                # Assuming label format is like '(cost_opex, 15_1, technology)'
                technology = label.split(', ')[-1].rstrip(')')
                if technology not in legend_labels:  # Add to dictionary if not already present
                    legend_labels[technology] = handle

        # Hide the individual legends for each subplot
        for ax in axs:
            ax.get_legend().remove()

        plt.subplots_adjust(hspace=0.5)

        global_min = min(ax.get_ylim()[0] for ax in axs)
        global_max = max(ax.get_ylim()[1] for ax in axs)
        for ax in axs:
            ax.set_ylim(global_min, global_max)

        # Now create a single legend with the collected unique handles and labels, without scenario tags
        fig.legend(legend_labels.values(), legend_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1),
                   ncol=math.ceil(len(legend_labels) / 4))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(top=0.85)
        plt.show(block=True)

    print("done")


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

    el_gen_costs_over_time()
