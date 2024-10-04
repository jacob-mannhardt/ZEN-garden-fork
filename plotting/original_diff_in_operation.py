from plotting.helpers import *



def flow_converstion_output_percent_plot(model_name):
    print("getting all sector emissions")

    r, config = read_in_special_results(model_name)
    desired_scenarios = ['5_5', '5_1']

    if model_name.startswith("PC_ct_"):
        e_5_1, em_5_1 = r.get_all_sector_emissions("scenario_2")
        e_5_5, em_5_5 = r.get_all_sector_emissions("scenario_3")
    else:
        raise NotImplementedError("Check the scenarios and model chosen")
    el_emissions_5_5 = sum_series_in_nested_dict(e_5_5['electricity'])
    heat_emissions_5_5 = sum_series_in_nested_dict(e_5_5['heat'])
    el_emissions_5_1 = sum_series_in_nested_dict(e_5_1['electricity'])
    heat_emissions_5_1 = sum_series_in_nested_dict(e_5_1['heat'])

    df_output = r.get_total("flow_conversion_output")
    df_output.reset_index(inplace=True)
    print("reset complete")
    # The 'level_0' column contains the scenario information, so we will rename it
    df_output.rename(columns={'level_0': 'scenario'}, inplace=True)
    df_output.set_index(['scenario', 'technology', 'carrier', 'node'], inplace=True)
    # Perform the aggregation by summing over the 'node' level of the index
    df_output = df_output.groupby(level=['scenario', 'technology', 'carrier']).sum()
    print("aggregation complete")
    # indexing
    df_output = indexmapping(df_output, special_model_name=model_name)
    # reducing to the desired scenarios
    df_output = df_output.loc[df_output.index.get_level_values('scenario').isin(desired_scenarios)]
    mask = (df_output.index.get_level_values('carrier') != 'carbon') \
           & (df_output.index.get_level_values('carrier') != 'natural_gas_industry') \
           & (df_output.index.get_level_values('carrier') != 'natural_gas')
    # Apply the mask
    masked_df = df_output[mask]
    renewables = ['biomass_boiler', "biomass_boiler_DH", 'biomass_plant', 'biomass_plant_CCS',
        'district_heating_grid','electrode_boiler', "electrode_boiler_DH", 'heat_pump', "heat_pump_DH", 'photovoltaics', 'reservoir_hydro',
       'run-of-river_hydro', 'waste_plant', 'wind_offshore', 'wind_onshore']
    conventional = ['hard_coal_boiler', "hard_coal_boiler_DH", 'hard_coal_plant', 'hard_coal_plant_CCS',
       'lignite_coal_plant', 'lng_terminal', "natural_gas_boiler_DH", 'natural_gas_boiler',
       'natural_gas_turbine', 'natural_gas_turbine_CCS', 'nuclear',
       'oil_boiler', "oil_boiler_DH", 'oil_plant', 'waste_plant', "waste_boiler_DH"]
    masked_df['category'] = masked_df.index.get_level_values('technology').map(
        lambda x: 'renewable' if x in renewables else 'conventional' if x in conventional else 'unknown')

    mask_heat = masked_df.index.get_level_values('carrier') == 'heat'
    heat_df = masked_df[mask_heat]

    mask_electricity = masked_df.index.get_level_values('carrier') == 'electricity'
    electricity_df = masked_df[mask_electricity]

    # for conventional tech barchart
    mask_conv_heat = masked_df.index.get_level_values('carrier').isin(['heat', 'district_heat'])
    conv_heat_df = masked_df[mask_conv_heat]
    mask_conventional = conv_heat_df['category'] == 'conventional'
    conv_heat_df = conv_heat_df[mask_conventional]
    conv_heat_df = conv_heat_df.drop(['category'], axis=1)
    conv_heat_df = conv_heat_df.reset_index('carrier', drop=True)

    updated_df = pd.DataFrame()
    # Iterate through unique scenarios
    for scenario in conv_heat_df.index.get_level_values('scenario').unique():
        scenario_df = conv_heat_df.xs(scenario, level='scenario', drop_level=False)
        # Identify technologies with '_DH' and their counterparts
        for tech in scenario_df.index.get_level_values('technology').unique():
            if tech.endswith('_DH'):
                non_dh_tech = tech[:-3]  # Identify the non-DH counterpart
                dh_row_index = (scenario, tech)  # Tuple for MultiIndex selection
                non_dh_row_index = (scenario, non_dh_tech)

                # Check if both '_DH' and non-'_DH' versions exist in the DataFrame
                if dh_row_index in scenario_df.index and non_dh_row_index in scenario_df.index:
                    # Sum values from '_DH' row into its non-'_DH' counterpart
                    conv_heat_df.loc[non_dh_row_index, :] += conv_heat_df.loc[dh_row_index, :]

                    rows_to_drop = []
                    rows_to_drop.append(conv_heat_df.loc[[dh_row_index]])
                    if rows_to_drop:  # Check if the list is not empty
                        updated_df = pd.concat(rows_to_drop)
                        conv_heat_df = conv_heat_df.drop(updated_df.index)
                    else:
                        updated_df = pd.DataFrame()

    mask_conventional = electricity_df['category'] == 'conventional'
    conv_el_df = electricity_df[mask_conventional]
    conv_el_df = conv_el_df.drop(['category'], axis=1)
    conv_el_df = conv_el_df.reset_index('carrier', drop=True)

    updated_df = pd.DataFrame()
    # Iterate through unique scenarios
    for scenario in conv_el_df.index.get_level_values('scenario').unique():
        scenario_df = conv_el_df.xs(scenario, level='scenario', drop_level=False)
        # Identify technologies with '_DH' and their counterparts
        for tech in scenario_df.index.get_level_values('technology').unique():
            if tech.endswith('_CCS'):
                non_ccs_tech = tech[:-4]  # Identify the non-DH counterpart
                ccs_row_index = (scenario, tech)  # Tuple for MultiIndex selection
                non_ccs_row_index = (scenario, non_ccs_tech)

                # Check if both '_DH' and non-'_DH' versions exist in the DataFrame
                if ccs_row_index in scenario_df.index and non_ccs_row_index in scenario_df.index:
                    # Sum values from '_DH' row into its non-'_DH' counterpart
                    conv_el_df.loc[non_ccs_row_index, :] += conv_el_df.loc[ccs_row_index, :]

                    rows_to_drop = []
                    rows_to_drop.append(conv_el_df.loc[[ccs_row_index]])
                    if rows_to_drop:  # Check if the list is not empty
                        updated_df = pd.concat(rows_to_drop)
                        conv_el_df = conv_el_df.drop(updated_df.index)
                    else:
                        updated_df = pd.DataFrame()

    # Apply the renaming function to the 'technology' part of the index
    new_index = [(item, rename_technology(tech)) for item, tech in conv_el_df.index]
    conv_el_df.index = pd.MultiIndex.from_tuples(new_index, names=conv_el_df.index.names)

    # Apply the renaming function to the 'technology' part of the index
    new_index = [(item, rename_technology(tech)) for item, tech in conv_heat_df.index]
    conv_heat_df.index = pd.MultiIndex.from_tuples(new_index, names=conv_heat_df.index.names)


    yaxisinpercent = False
    if False:
        conv_heat_df = normalize_by_scenario(conv_heat_df) * 100
        conv_el_df = normalize_by_scenario(conv_el_df) * 100
        yaxisinpercent = True

    # district heating special treatment:
    original_index_names = heat_df.index.names
    for scenario in desired_scenarios:
        row_to_duplicate = heat_df.loc[scenario].loc["district_heating_grid"]
        carrier_value = "heat"
        modified_row = row_to_duplicate.copy(deep=True)
        modified_row['category'] = 'conventional'  # Update the 'category' column
        new_index = (scenario, 'district_heating_grid_conventional', carrier_value)
        modified_row.index = pd.MultiIndex.from_tuples([new_index])

        temp_output_df = df_output.loc[scenario]
        temp_output_df = temp_output_df.loc[temp_output_df.index.get_level_values('carrier') == 'district_heat']
        ren_technologies = ["biomass_boiler_DH", "electrode_boiler_DH", "heat_pump_DH"]
        temp_output_df = temp_output_df.loc[temp_output_df.index.get_level_values('technology').isin(ren_technologies)]
        total_ren_temp_output_df = temp_output_df.sum()

        result_df = modified_row.iloc[:, :-1] - total_ren_temp_output_df
        # Append the last column of 'modified_row' back to the result
        result_df[modified_row.columns[-1]] = modified_row.iloc[:, -1]

        # Append it
        heat_df = pd.concat([heat_df, result_df])
    heat_df.index.names = original_index_names
    heat_df = heat_df.sort_index()

    for scenario in desired_scenarios:
        diff = (
                heat_df.loc[(scenario, 'district_heating_grid'), heat_df.columns[:-1]] -
                heat_df.loc[(scenario, 'district_heating_grid_conventional'), heat_df.columns[:-1]]
        ).values
        heat_df.loc[(scenario, 'district_heating_grid'), heat_df.columns[:-1]] = diff

    percentage_renewable_electricity = calculate_renewable_percentage(electricity_df)
    percentage_renewable_heat = calculate_renewable_percentage(heat_df)

    # cumulative emission calc for red cross in figure
    cumulative_emissions = r.get_total("carbon_emissions_cumulative")
    cumulative_emissions = indexmapping(cumulative_emissions)
    cumulative_emissions = cumulative_emissions.loc[desired_scenarios]
    carbon_budget = 16580
    scenario_positions_values = []
    for scenario in desired_scenarios:
        scenario_row = cumulative_emissions.loc[scenario]
        # Calculate differences between consecutive values
        consecutive_diffs = scenario_row.diff()
        # Find the maximum difference
        max_consecutive_difference = consecutive_diffs.max()

        position_met = None
        # Iterate over each value in the row
        for position, value in scenario_row.items():
            # Check if adding the max_consecutive_difference to the current value meets or exceeds the carbon budget
            if value + max_consecutive_difference >= carbon_budget:
                position_met = position
                calculated_value = value + max_consecutive_difference
                break
        # Only add to the list if a position and calculated value were found
        if position_met is not None and calculated_value is not None:
            scenario_positions_values.append((scenario, position_met, calculated_value))

    # emission comparison figure
    for scenario, position, value in scenario_positions_values:
        if scenario == '5_1':
            position_5_1 = position
            y_value_5_1 = value
            break

    if True:
        fig, axs = plt.subplots(3, 2, figsize=(9, 13))  # Creating a 2x2 grid of subplots

        # Top-left plot (Emissions delta for electricity)    (scenario_emissions_delta_el.columns
        axs[1, 0].plot(el_emissions_5_5.index, el_emissions_5_5,
                       label='IF:10yr,  OF:10yr', color='black', linestyle='-', zorder=2)
        axs[1, 0].plot(el_emissions_5_1.index, el_emissions_5_1,
                       label='IF:10yr,  OF:2yr', color='black', linestyle=':', zorder=2) # 'IF:30a,  OF:2a'
        axs[1, 0].scatter(position_5_1, el_emissions_5_1[position_5_1],
                          color='red', marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[1, 0].set_title('Electricity Generation Emissions')
        axs[1, 0].set_ylabel('Annual Emissions [Mt CO$_{2}$]', fontsize=12)
        axs[1, 0].grid(True, zorder=1)

        # Top-right plot (Emissions delta for heat)  scenario_emissions_delta_heat.sum()
        axs[1, 1].plot(heat_emissions_5_5.index, heat_emissions_5_5,
                       label='IF:10yr,  OF:10yr', color='black', linestyle='-', zorder=2)
        axs[1, 1].plot(heat_emissions_5_1.index, heat_emissions_5_1,
                       label='IF:10yr,  OF:2yr', color='black', linestyle=':', zorder=2) # 'IF:30a,  OF:2a'
        axs[1, 1].scatter(position_5_1, heat_emissions_5_1[position_5_1], color='red',
                          marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[1, 1].set_title('Heat Generation Emissions')
        axs[1, 1].set_ylabel('Annual Emissions [Mt CO$_{2}$]', fontsize=12)
        axs[1, 1].grid(True, zorder=1)

        # Bottom-left plot (Percentage renewable electricity)
        axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['5_1'],
                       label='IF:10yr,  OF:2yr', linestyle=':', color='black', zorder=2) # 'IF:30a,  OF:2a'
        axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['5_5'],
                       label='IF:10yr,  OF:10yr', linestyle='-', color='black', zorder=2)
        axs[0, 0].scatter(position_5_1, percentage_renewable_electricity.loc['5_1'][position_5_1], color='red',
                          marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[0, 0].set_title('Electricity from Renewable Generation')
        axs[0, 0].set_ylabel('Renewable Electricity [%]', fontsize=12)
        axs[0, 0].grid(True, zorder=1)

        # Bottom-right plot (Percentage renewable heat)
        axs[0, 1].plot(percentage_renewable_electricity.columns, percentage_renewable_heat.loc['5_1'],
                       label='IF:10yr,  OF:2yr', linestyle=':', color='black', zorder=2) # 'IF:30a,  OF:2a'
        axs[0, 1].plot(percentage_renewable_electricity.columns, percentage_renewable_heat.loc['5_5'], label='5_5',
                       linestyle='-', color='black', zorder=2)
        axs[0, 1].scatter(position_5_1, percentage_renewable_heat.loc['5_1'][position_5_1], color='red', marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[0, 1].set_title('Heat from Renewable Generation')
        axs[0, 1].set_ylabel('Renewable Heat [%]', fontsize=12)
        axs[0, 1].grid(True, zorder=1)

        width = 0.35  # Width of the bars

        def colors(index):
            if index == "hard_coal": color = "black"
            elif index == "nuclear": color = "#eed202"
            elif index == "natural_gas": color = "#377eb8"
            elif index == "oil": color = "#66c2a5"
            elif index == "waste": color = "#999999"
            elif index == "lignite_coal": color = "#a65628"
            else: raise NotImplementedError("Technology-Color not implemented")
            return color

        # Dictionary to hold technology_name: color pairs
        color_map = {}
        a = 0
        # Plotting for conv_el_df (Electricity) on the left subplot of the last row
        for scenario in desired_scenarios:
            bottom = np.zeros(15)  # Reset the bottom array for each scenario
            plotting_df = conv_el_df.loc[scenario]
            for count, index in enumerate(plotting_df.index):
                    # Calculate position offset based on scenario
                    positions = np.arange(0, 15) - width / 2 if scenario == '5_1' else np.arange(0, 15) + width / 2
                    # Extract technology name for the label
                    technology_name = format_string(index)
                    if yaxisinpercent != True:
                        axs[2, 0].bar(positions, (plotting_df.loc[index])/1000, width, bottom=bottom, label=technology_name, color=colors(index))
                        bottom += (plotting_df.loc[index]) / 1000
                    else:
                        axs[2, 0].bar(positions, plotting_df.loc[index], width, bottom=bottom, label=technology_name, color=colors(index))
                        bottom += plotting_df.loc[index]
                    if technology_name not in color_map:
                        color_map[technology_name] = colors(index)
                    if index == "nuclear":
                        if a < 1:
                            # Calculate the x and y positions for the label and arrow
                            x = positions[2]
                            y = bottom[2]
                            label_y_position = y + (0.1 * y)
                            label_x_position = x + 0.5
                            # Draw an arrow pointing to the bar
                            axs[2, 0].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                            # Add text label slightly above and to the right/left depending on position
                            axs[2, 0].text(label_x_position+2.5, label_y_position, 'IF:10yr,  OF:10yr', ha='center')
                        else:
                            # Calculate the x and y positions for the label and arrow
                            x = positions[2]
                            y = bottom[2]
                            label_y_position = y + (0.16 * y)
                            label_x_position = x + 0.3
                            # Draw an arrow pointing to the bar
                            axs[2, 0].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                            # Add text label slightly above and to the right/left depending on position
                            axs[2, 0].text(label_x_position + 2.4, label_y_position + (0.01 * y), 'IF:10yr,  OF:2yr', ha='center') #  'IF:30a,  OF:2a'
                        a += 1
        if yaxisinpercent != True:
            axs[2, 0].set_ylabel('Conventional Energy [TWh$_{el}$]', fontsize=12)
        else:
            axs[2, 0].set_ylabel('Carrier Mix of Conventional Electricity Provided [%]', fontsize=12)
        axs[2, 0].set_title('Carrier Mix of Conv. Electricity Generation')


        a=0
        for scenario in desired_scenarios:
            bottom = np.zeros(15)  # Reset the bottom array for each scenario
            plotting_df = conv_heat_df.loc[scenario]
            for count, index in enumerate(plotting_df.index):
                    # Calculate position offset based on scenario
                    positions = np.arange(0, 15) - width / 2 if scenario == '5_1' else np.arange(0, 15) + width / 2
                    # Extract technology name for the label
                    technology_name = format_string(index)
                    if yaxisinpercent != True:
                        axs[2, 1].bar(positions, (plotting_df.loc[index])/1000, width, bottom=bottom, label=technology_name, color=colors(index))
                        bottom += (plotting_df.loc[index])/1000
                    else:
                        axs[2, 1].bar(positions, plotting_df.loc[index], width, bottom=bottom, label=technology_name, color=colors(index))
                        bottom += plotting_df.loc[index]
                    if technology_name not in color_map:
                        color_map[technology_name] = colors(index)

                    if index == "waste":
                        if a < 1:
                            # Calculate the x and y positions for the label and arrow
                            x = positions[2]
                            y = bottom[2]
                            label_y_position = y + (0.1 * y)
                            label_x_position = x + 0.5
                            if model_name.endswith("_5"):
                                # Draw an arrow pointing to the bar
                                axs[2, 1].plot([x, label_x_position], [y, label_y_position + (0.05 * y)], color="black", linewidth=1)
                                # Add text label slightly above and to the right/left depending on position
                                axs[2, 1].text(label_x_position + 2.5, label_y_position + (0.05 * y), 'IF:10yr,  OF:10yr',
                                               ha='center')
                            else:
                                # Draw an arrow pointing to the bar
                                axs[2, 1].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                                # Add text label slightly above and to the right/left depending on position
                                axs[2, 1].text(label_x_position+2.5, label_y_position, 'IF:10yr,  OF:10yr', ha='center')
                        else:
                            # Calculate the x and y positions for the label and arrow
                            x = positions[2]
                            y = bottom[2]
                            label_y_position = y + (0.06 * y)
                            label_x_position = x + 0.3
                            # Draw an arrow pointing to the bar
                            axs[2, 1].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                            # Add text label slightly above and to the right/left depending on position
                            axs[2, 1].text(label_x_position + 2.4, label_y_position + (0.01 * y), 'IF:10yr,  OF:2yr', ha='center') # 'IF:30a,  OF:2a'
                        a += 1

        if yaxisinpercent != True:
            axs[2, 1].set_ylabel('Conventional Energy [TWh$_{th}$]', fontsize=12)
        else:
            axs[2, 1].set_ylabel('Carrier Mix of Conventional Heat Provided [%]', fontsize=12)
        axs[2, 1].set_title('Carrier Mix of Conv. Heat Generation')

        # # Adding legends and adjusting layout
        for ax in axs.flat:
            ax.legend().set_visible(False)
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 0.42), ncol=3)

        # This creates the first legend and attaches it to axs[0, 0]
        # Now, creating custom patches for the second legend
        legend_patches = [patches.Patch(color=color, label=label) for label, color in color_map.items()]
        # Creating the second legend at the figure level
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.53, 0.1), ncol=6)
        #
        # Adjust the layout
        # # Adjust the layout to make space for the legend outside the subplots
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        # Adjust spacing between subplots and increase the bottom margin
        fig.subplots_adjust(hspace=0.4, bottom=0.15)  # Increased bottom margin

        # Retrieve and adjust the positions of the last row subplots to create extra space
        # Manually adjusting the 'bottom' property of the third row's axes
        bottom_shift = 0.02  # Amount to move the last row down, adjust as needed
        # for ax in axs[2, :]:  # Only adjust the last row
        #     pos = ax.get_position()
        #     ax.set_position([pos.x0, pos.y0 - bottom_shift, pos.width, pos.height])

        i = 0
        for row in axs:
            j = 0
            for ax in row:
                # ax.grid(True, zorder=1)
                ax.grid(False)
                if i < 1:
                    ax.set_ylim(0, 105)
                    yticks_major = np.arange(0, 105, 25)  # Adjust 25 to your preferred y-interval
                    ax.set_yticks(yticks_major)

                    xticks_major = np.arange(0, 15, 2)
                    ax.set_xticks(xticks_major)
                    # Calculate new labels for these ticks
                    new_labels = [2022 + 2 * int(tick) for tick in xticks_major]
                    ax.set_xticklabels(new_labels, rotation=0)
                    for xtick in ax.get_xticks():
                        ax.plot([xtick, xtick], [0, 100], linestyle=':', linewidth=0.3,
                                color='grey')  # Adjust linestyle, linewidth, and color as needed

                    for ytick in yticks_major[1:]:  # Exclude first and last to draw lines between ticks
                        ax.axhline(y=ytick, color='lightgray', linestyle=':', linewidth=0.3)

                elif i < 2:
                    ax.set_ylim(-150, 750)
                    positive_ticks = np.arange(100, 751, 100)
                    negative_ticks = np.arange(0, -151, -100)
                    yticks_major = np.concatenate((negative_ticks[::-1], positive_ticks))
                    ax.set_yticks(yticks_major)
                    ax.yaxis.grid(True, which='major', linestyle=':', linewidth=0.3,
                                  color='grey')  # Adjust linestyle, linewidth, and color as needed

                    xticks_major = np.arange(0, 15, 2)
                    ax.set_xticks(xticks_major)
                    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.7)
                    ax.xaxis.grid(True, which='major', linestyle=':', linewidth=0.3,
                                  color='grey')  # Adjust linestyle, linewidth, and color as needed

                    # Calculate new labels for these ticks
                    new_labels = [2022 + 2 * int(tick) for tick in xticks_major]
                    ax.set_xticklabels(new_labels, rotation=0)
                else:
                    if yaxisinpercent == True:
                        ax.set_ylim(0, 105)
                        yticks_major = np.arange(0, 105, 25)  # Adjust 25 to your preferred y-interval
                    elif j > 0:
                        if model_name.startswith("PC_ct_"):
                            ax.set_ylim(0, 3100)
                            yticks_major = np.arange(0, 3100, 250)
                        else:
                            raise NotImplementedError("Check the scenarios and model chosen")
                    else:
                        if model_name.startswith("PC_ct_"):
                            ax.set_ylim(0, 2100)
                            yticks_major = np.arange(0, 2100, 250)
                        else:
                            raise NotImplementedError("Check the scenarios and model chosen")
                    ax.set_yticks(yticks_major)

                    xticks_major = np.arange(0, 15, 2)
                    ax.set_xticks(xticks_major)
                    # Calculate new labels for these ticks
                    new_labels = [2022 + 2 * int(tick) for tick in xticks_major]
                    ax.set_xticklabels(new_labels, rotation=0)
                j += 1
            i += 1

        plt.show()

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

model_name = "PC_ct_pass_tra_base" # , "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_ETS1and2", "PC_ct_pass_tra_gen_target"
# sectors = ["electricity", "heat", "passenger_mileage"] #  "electricity", "heat", "passenger_mileage"

flow_converstion_output_percent_plot(model_name)

