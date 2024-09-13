from plotting.helpers import *

def feedback_plot_progression_with_techs(model_name):
    print("plotting feedback_plot_progression")
    if type(model_name) is not str:
        model_name = str(model_name)
    r, config = read_in_results(model_name)

    projections =  ['0_inv', '3_inv', '4_inv', '14_inv'] # '2_inv','6_inv',

    df_emissions = r.get_total("carbon_emissions_cumulative", keep_raw=True)

    # indexing
    df_emissions = indexmapping(df_emissions, special_model_name=model_name)
    # Select only the desired rows
    if desired_scenarios is not None:
        df_emissions = df_emissions.loc[desired_scenarios]

    assert len(df_emissions.columns) == 15, "Number of timesteps not equal to 15"
    # Order for 'actual' and '_inv' rows separately as you requested
    order_op = [str(i) for i in range(15)]  # '0' to '14'
    order_inv = [f"{i}_inv" for i in range(15)]  # '0_inv' to '14_inv'

    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows_emissions = df_emissions[df_emissions.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows_emissions = inv_rows_emissions.reindex(order_inv, level=1)
    inv_rows_emissions = inv_rows_emissions.loc[(slice(None),projections), :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows_emissions = df_emissions[~df_emissions.index.get_level_values('mf').str.endswith('_inv')]
    op_rows_emissions = op_rows_emissions.reindex(order_op, level=1)
    # Identifying the first non-NaN value in each 'mf' row
    op_rows_emissions = op_rows_emissions.apply(
        lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1)
    op_rows_emissions = op_rows_emissions.values.flatten()

    initial_midpoint = 0
    # Incorporate the initial midpoint with the rest
    midpoints_op_emissions = [initial_midpoint] + [(op_rows_emissions[i] + op_rows_emissions[i + 1]) / 2 for i
                                                            in range(len(op_rows_emissions) - 1)]
    new_columns = np.arange(-0.5, inv_rows_emissions.columns.max() + 0.5, 0.5)
    all_columns = sorted(set(inv_rows_emissions.columns).union(new_columns))
    inv_rows_emissions = inv_rows_emissions.reindex(columns=all_columns)

    first_non_nan_indices = inv_rows_emissions.apply(lambda row: row.first_valid_index(), axis=1)

    for (indx, first_non_nan_index) in first_non_nan_indices.items():
        # Convert the index to an actual position; assuming it directly matches your description
        position = float(first_non_nan_index)
        # Find the corresponding value in midpoints_op_emissions
        value_to_add = midpoints_op_emissions[int(position)]
        # Determine the target column ("position - 0.5")
        target_column = position - 0.5
        assert pd.isna(inv_rows_emissions.at[indx, target_column]), "Expected NaN value not found"
        # Check if the target column exists in the DataFrame
        if target_column in inv_rows_emissions.columns:
            # Add the value to the specified position
            inv_rows_emissions.at[indx, target_column] = value_to_add

    # =================================================================================================================
    # do the same for capacity expansion:
    df_cap = r.get_total("capacity", keep_raw=True)
    assert len(df_cap.columns) == 15, "Number of timesteps not equal to 15"

    # Reset the index to work with it as columns
    df_cap.reset_index(inplace=True)
    print("reset complete")
    # The 'level_0' column contains the scenario information, so we will rename it
    df_cap.rename(columns={'level_0': 'scenario'}, inplace=True)

    # Now we set the multi-index again, this time including the 'scenario'
    df_cap.set_index(['scenario', 'technology', 'capacity_type', 'location', 'mf'], inplace=True)
    print("re-index complete")
    # Perform the aggregation by summing over the 'node' level of the index
    df_cap = df_cap.groupby(level=['scenario', 'technology', 'capacity_type', 'mf']).sum()
    print("aggregation complete")
    # indexing
    df_cap = indexmapping(df_cap, special_model_name=model_name)
    # reducing to the desired scenarios
    df_cap = df_cap.loc[df_cap.index.get_level_values('scenario').isin(desired_scenarios)]
    # get rid of capacit_type "energy"
    df_cap = df_cap[df_cap.index.get_level_values('capacity_type') != 'energy']
    df_cap.index = df_cap.index.droplevel('capacity_type')
    unique_technologies = df_cap.index.get_level_values('technology').unique()
    list_of_technologies = unique_technologies.tolist()
    # List of technologies to remove
    renewable_el_generation = ['biomass_plant',
                              'photovoltaics', 'reservoir_hydro', 'run-of-river_hydro',
                               'wind_offshore', 'wind_onshore'] # 'biomass_plant_CCS',
    renewable_heating = ['biomass_boiler', 'biomass_boiler_DH', 'district_heating_grid',
                         'heat_pump', 'heat_pump_DH', 'waste_boiler_DH',]
    # wind = ['wind_offshore', 'wind_onshore']
    biomass = ['biomass_plant'] # , 'biomass_plant_CCS'
    hydro = ['reservoir_hydro', 'run-of-river_hydro']
    pv = ['photovoltaics']
    wind_on = ['wind_onshore']
    wind_off = ['wind_offshore']
    # renewable_storage = ['battery', 'pumped_hydro', 'hydrogen_storage']
    technologies_to_keep = renewable_el_generation
    df_cap = df_cap.loc[(slice(None), technologies_to_keep), :]

    mapping_dict = {}
    for tech in biomass:
        mapping_dict[tech] = 'Biomass'
    for tech in hydro:
        mapping_dict[tech] = 'Hydro'
    for tech in pv:
        mapping_dict[tech] = 'Photovoltaics'
    for tech in wind_on:
        mapping_dict[tech] = 'Onshore Wind'
    for tech in wind_off:
        mapping_dict[tech] = 'Offshore Wind'

    # for tech in wind:
    #     mapping_dict[tech] = 'Wind'

    # for tech in renewable_storage:
    #     mapping_dict[tech] = 'renewable_storage'

    df_cap = df_cap.rename(index=mapping_dict, level='technology')
    df_cap = df_cap.groupby(level=['scenario', 'technology', 'mf']).sum()


    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows_cap = df_cap[df_cap.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows_cap = inv_rows_cap.reindex(order_inv, level=2)
    inv_rows_cap = inv_rows_cap.loc[idx[:, :, projections], :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows_cap = df_cap[~df_cap.index.get_level_values('mf').str.endswith('_inv')]
    op_rows_cap = op_rows_cap.reindex(order_op, level=2)
    # Identifying the first non-NaN value in each 'mf' row
    op_rows_cap = op_rows_cap.groupby(level='technology').sum()

    inv_cap_dataframes = {}
    for i, entry in enumerate(projections):
        num = int(entry.split('_')[0])
        # Create and store the DataFrame in the dictionary with a dynamic key
        df_temp = inv_rows_cap.loc[pd.IndexSlice[:, :, entry], :]

        # Replace columns up to that number from op_rows_cap, if num > 0
        if num > 0:
            df_temp.iloc[:, :num] = op_rows_cap.iloc[:, :num]

        inv_cap_dataframes[f'df_{num}'] = df_temp

    # ==================================================================================================================
    hatch = False
    something = True
    # plotting things
    def get_color_key(row_identifier):
        number_part = row_identifier.split('_')[0]
        return "df_" + number_part

    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # Choose a palette
    # palette = sns.color_palette("Set1", 7)  # 'Set2' for distinct, muted colors; change n_colors if more categories
    palette = sns.light_palette("seagreen", reverse=False, n_colors=256)
    palette2 = sns.dark_palette("seagreen", reverse=True, n_colors=256)
    # Create equally spaced indices to select colors
    indices = np.linspace(0, 255, num=15, dtype=int)
    selected_colors = np.array(palette)[indices]
    selected_colors2 = np.array(palette2)[indices]
    iterable_colors2 = list(map(tuple, selected_colors2))

    # Convert the palette to a list of RGB strings colors can be directly used in plotting functions
    # colors = [mcolors.to_hex(c) for c in palette]
    iterable_colors = list(map(tuple, selected_colors))
    iterable_colors.extend(iterable_colors2)

    colors = {
        'df_0': iterable_colors[4],
        'df_1': iterable_colors[3],
        'df_2': iterable_colors[14],
        'df_3': iterable_colors[14],
        'df_4': iterable_colors[14],
        'df_5': iterable_colors[8],
        'df_6': iterable_colors[9],
        'df_14': iterable_colors[21]
    }

    colors_emissions_green = {
    'df_0123': '#A8D5BA',  # Light Muted Green
    'df_0': iterable_colors[4],
    'df_41': '#8CBFA3',  # Light-Medium Muted Green
    'df_4': iterable_colors[14],
    'df_2': iterable_colors[14],
    'df_3': iterable_colors[14],
    'df_142': '#578776',  # Medium-Dark Muted Green
    'df_14': iterable_colors[21],
    'df_7': '#3E7061',  # Dark Muted Green
    'df_5': '#A8D5BA',  # Looping back to Light Muted Green
    'df_6': '#8CBFA3',  # Looping back to Light-Medium Muted Green
    'df_13': '#709F8C'  # Looping back to Medium Muted Green
    }

    colors_emissions = {
        'df_1': "#F2F2F2",  # Very Light Gray
        'df_0': "#CCCCCC",  # Light Gray
        'df_2': "#808080",  # Medium Gray
        'df_3': "#808080",  # Medium Gray
        'df_4': "#808080",  # Medium Gray
        'df_42': "#595959",  # Medium Dark Gray
        'df_5': "#333333",  # Dark Gray
        'df_14': "#000000"  # Black
    }

    def colors(index):
        if index == "Hydro":
            return "#0077BE"
        elif index == "Onshore Wind":
            return "#ADD8E6"
        elif index == "Offshore Wind":
            return "#6CA0DC"
        elif index == "Biomass":
            return "#19A519"
        elif index == "Photovoltaics":
            return "#FDFD96"
        elif index == "lignite_coal":
            return "#a65628"
        else:
            raise NotImplementedError(f"Technology-Color not implemented for '{index}'")

    # Plotting emissions on the primary y-axis
    legend_labels = [f"{2022 + 2 * int(row[1].split('_')[0])}" for row in inv_rows_emissions.index]
    legend_labels = legend_labels[:-1]
    i = 0
    if hatch == True:
        for row in inv_rows_emissions.index:
            color_key = get_color_key(row[1])
            if i < len(legend_labels):
                label = legend_labels[i]
            else:
                label = None
            ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors[color_key])
            i += 1
    else:
        for i, row in enumerate(inv_rows_emissions.index):
            if i < 5:
                color_key = get_color_key(row[1])
                if i < len(legend_labels):
                    label = legend_labels[i]
                else:
                    label = None
                ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors_emissions[color_key])
                i += 1

    first_y_value = op_rows_emissions[0] / 1000
    if hatch == True:
        ax1.plot(op_rows_emissions / 1000, label='Actual', color=iterable_colors[21], marker='', markersize=4,
                 markerfacecolor='black', lw=3)
        ax1.plot([-0.5, 0], [0, first_y_value], color=iterable_colors[21], lw=3)
    else:
        ax1.plot(op_rows_emissions / 1000, label='2050', color='black', marker='', markersize=4,
                 markerfacecolor='black', lw=3) # [:6]
        ax1.plot([-0.5, 0], [0, first_y_value], color='black', lw=3)
    # Setting the labels for the primary y-axis
    ax1.set_ylabel('Cumulative Carbon Emissions [Gt CO₂]', fontsize=12)

    # Carbon budget line on the primary y-axis
    carbon_budget = 16.58
    ax1.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0, xmax=0.95, zorder=1, alpha=0.5)
    # ax1.text(2.4, carbon_budget + 0.025 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='left', color='black',
    #          fontsize=10)
    # Define the label position
    label_x = 2.4
    label_y = carbon_budget + 0.025 * plt.ylim()[1]  # Position for the text above the carbon budget line

    # Place the text label
    ax1.text(label_x, label_y, "Carbon Emission Budget", va='bottom', ha='left', color='black', fontsize=10)

    # Draw a vertical connecting line from the label straight down to the carbon budget line
    ax1.plot([label_x, label_x], [carbon_budget, label_y], color='black', linestyle='-', linewidth=0.5)

    # Create secondary y-axis for the capacities
    ax2 = ax1.twinx()
    # Number of DataFrames (scenarios) and an example of time steps (assuming they're consistent across DataFrames)
    num_dfs = len(inv_cap_dataframes)
    time_steps = list(
        inv_cap_dataframes[next(iter(inv_cap_dataframes))].columns)  # Grab time steps from the first DataFrame
    bar_width = 0.8 / num_dfs  # Width of each bar, divided by the number of scenarios
    opacity = 0.5  # Adjust as needed for visibility

    # Define hatch patterns and edge color
    hatch_patterns = ['***', 'XX', '//////', r'\\\\\\', '']
    edge_color = 'black'  # Color for the edges of each bar segment

    for i, time_step in enumerate(time_steps):
        for j, (df_name, df) in enumerate(inv_cap_dataframes.items()):
            if j < 5:
                df_number = int(df_name.split('_')[1])
                bar_position = np.arange(len(time_steps)) + (j - len(inv_cap_dataframes) / 2) * bar_width + (bar_width / 2)
                data = df[time_step]

                bottom = np.zeros(len(data))
                edge_color = 'black'

                if hatch == True:
                    bar_color = colors.get(df_name, 'black')  # Fallback color
                    for k, category in enumerate(data.index):
                        # if time_step < df_number and df_number != 14:
                        #     bar_color = 'none'
                        #     edge_color = 'lightgrey' # 'dimgrey'
                        # Cycle through hatch patterns based on technology's position
                        hatch = hatch_patterns[k % len(hatch_patterns)]
                        ax2.bar(bar_position[i], data.loc[category]/1000, bottom=bottom, width=bar_width, alpha=opacity,
                                color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}', hatch=hatch)
                        bottom += data.loc[category]/1000
                else:
                    # Function to convert hex to RGB
                    def hex_to_rgb(hex_color):
                        hex_color = hex_color.lstrip('#')
                        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

                    # Function to convert RGB back to hex
                    def rgb_to_hex(rgb_color):
                        return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

                    # Function to darken the color
                    def adjust_color(color, factor):
                        rgb_color = hex_to_rgb(color)
                        darker_color = tuple(np.clip(np.array(rgb_color) * factor, 0, 255))
                        return rgb_to_hex(darker_color)

                    for k, category in enumerate(data.index):
                        original_color = colors(category[1])
                        # Adjust the color to be darker based on `j`
                        factor = 1 - 0.2 * j  # Adjust the factor according to your needs
                        bar_color = adjust_color(original_color, factor)
                        ax2.bar(bar_position[i], data.loc[category] / 1000, bottom=bottom, width=bar_width, alpha=opacity,
                                color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}')
                        bottom += data.loc[category] / 1000

    # Setting the label for the secondary y-axis
    ax2.set_ylabel('Projected Renewable Generation Capacity [TW]', fontsize=12, labelpad=15)
    ax2.set_ylim(0, 4.99)
    ax1.set_ylim(0, 18)

    # Legend and title
    # generate figure title:
    fig_title = model_name
    plt.title(fig_title)
    legend_techs = ["Biomass", "Hydro", "Offshore Wind", "Onshore Wind", "Photovoltaics"] #  , "renewable_storage"
    # # Create custom legend handles based on technologies and their corresponding hatch patterns
    if hatch == True:
        legend_handles = [patches.Patch(facecolor='white', edgecolor='black', hatch=hatch_patterns[i], label=legend_techs[i])
                          for i, technology in enumerate(legend_techs)]
    else:
        legend_handles = [patches.Patch(facecolor=colors(technology), edgecolor='black', label=technology)
                          for technology in legend_techs]

    # For better control, manually specify the legend to include labels from both plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    inv_horizon, op_horizon = desired_scenarios[0].split('_')
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1),
               title='Foresight Horizon\nInvestment: ' + str(int(inv_horizon) * 2) + 'yr' + '\nOperation: ' + str(
                   int(op_horizon) * 2) + 'yr\n\nDecision-Maker\nPerspective in:', framealpha=1)

    if hatch == True:
        # After all plotting commands, manually add the legend with custom handles
        plt.legend(handles=legend_handles, title="Renewable Electricity Generation Capacities", loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5) #
    else:
        plt.legend(handles=legend_handles, title="Renewable Electricity Generation Capacities", loc='upper center',
                   bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)

    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    ax1.spines['left'].set_position(('data', -0.5))
    ax1.set_xlim([-0.5, 14.5])

    # Modify x-axis tick labels if needed
    # This part may require adjustment depending on your data's time format
    new_labels = [2022 + 2 * int(label) for label in ax1.get_xticks()]
    whole_numbers = range(15)
    ax1.set_xticklabels(new_labels)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.show(block=True)
    return

def feedback_plot_progression_with_techs_heat(model_name):
    print("plotting feedback_plot_progression")
    if type(model_name) is not str:
        model_name = str(model_name)
    r, config = read_in_results(model_name)

    projections =  ['0_inv', '3_inv', '4_inv', '14_inv'] # '2_inv','6_inv',

    df_emissions = r.get_total("carbon_emissions_cumulative", keep_raw=True)

    # indexing
    df_emissions = indexmapping(df_emissions, special_model_name=model_name)
    # Select only the desired rows
    if desired_scenarios is not None:
        df_emissions = df_emissions.loc[desired_scenarios]

    assert len(df_emissions.columns) == 15, "Number of timesteps not equal to 15"
    # Order for 'actual' and '_inv' rows separately as you requested
    order_op = [str(i) for i in range(15)]  # '0' to '14'
    order_inv = [f"{i}_inv" for i in range(15)]  # '0_inv' to '14_inv'

    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows_emissions = df_emissions[df_emissions.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows_emissions = inv_rows_emissions.reindex(order_inv, level=1)
    inv_rows_emissions = inv_rows_emissions.loc[(slice(None),projections), :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows_emissions = df_emissions[~df_emissions.index.get_level_values('mf').str.endswith('_inv')]
    op_rows_emissions = op_rows_emissions.reindex(order_op, level=1)
    # Identifying the first non-NaN value in each 'mf' row
    op_rows_emissions = op_rows_emissions.apply(
        lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1)
    op_rows_emissions = op_rows_emissions.values.flatten()

    initial_midpoint = 0
    # Incorporate the initial midpoint with the rest
    midpoints_op_emissions = [initial_midpoint] + [(op_rows_emissions[i] + op_rows_emissions[i + 1]) / 2 for i
                                                            in range(len(op_rows_emissions) - 1)]
    new_columns = np.arange(-0.5, inv_rows_emissions.columns.max() + 0.5, 0.5)
    all_columns = sorted(set(inv_rows_emissions.columns).union(new_columns))
    inv_rows_emissions = inv_rows_emissions.reindex(columns=all_columns)

    first_non_nan_indices = inv_rows_emissions.apply(lambda row: row.first_valid_index(), axis=1)

    for (indx, first_non_nan_index) in first_non_nan_indices.items():
        # Convert the index to an actual position; assuming it directly matches your description
        position = float(first_non_nan_index)
        # Find the corresponding value in midpoints_op_emissions
        value_to_add = midpoints_op_emissions[int(position)]
        # Determine the target column ("position - 0.5")
        target_column = position - 0.5
        assert pd.isna(inv_rows_emissions.at[indx, target_column]), "Expected NaN value not found"
        # Check if the target column exists in the DataFrame
        if target_column in inv_rows_emissions.columns:
            # Add the value to the specified position
            inv_rows_emissions.at[indx, target_column] = value_to_add

    # =================================================================================================================
    # do the same for capacity expansion:
    df_cap = r.get_total("capacity", keep_raw=True)
    assert len(df_cap.columns) == 15, "Number of timesteps not equal to 15"

    # Reset the index to work with it as columns
    df_cap.reset_index(inplace=True)
    print("reset complete")
    # The 'level_0' column contains the scenario information, so we will rename it
    df_cap.rename(columns={'level_0': 'scenario'}, inplace=True)

    # Now we set the multi-index again, this time including the 'scenario'
    df_cap.set_index(['scenario', 'technology', 'capacity_type', 'location', 'mf'], inplace=True)
    print("re-index complete")
    # Perform the aggregation by summing over the 'node' level of the index
    df_cap = df_cap.groupby(level=['scenario', 'technology', 'capacity_type', 'mf']).sum()
    print("aggregation complete")
    # indexing
    df_cap = indexmapping(df_cap, special_model_name=model_name)
    # reducing to the desired scenarios
    df_cap = df_cap.loc[df_cap.index.get_level_values('scenario').isin(desired_scenarios)]
    # get rid of capacit_type "energy"
    df_cap = df_cap[df_cap.index.get_level_values('capacity_type') != 'energy']
    df_cap.index = df_cap.index.droplevel('capacity_type')
    unique_technologies = df_cap.index.get_level_values('technology').unique()
    list_of_technologies = unique_technologies.tolist()
    # List of technologies to remove
    renewable_heating = ['biomass_boiler', 'biomass_boiler_DH',
                         'heat_pump', 'heat_pump_DH']
    # wind = ['wind_offshore', 'wind_onshore']
    biomass = ['biomass_boiler', 'biomass_boiler_DH']
    heat_pump = ['heat_pump', 'heat_pump_DH']

    # renewable_storage = ['battery', 'pumped_hydro', 'hydrogen_storage']
    technologies_to_keep = renewable_heating # + renewable_heating# + renewable_storage
    df_cap = df_cap.loc[(slice(None), technologies_to_keep), :]

    mapping_dict = {}
    for tech in biomass:
        mapping_dict[tech] = 'Biomass'
    for tech in heat_pump:
        mapping_dict[tech] = 'Heat Pump'

    # for tech in wind:
    #     mapping_dict[tech] = 'Wind'

    # for tech in renewable_storage:
    #     mapping_dict[tech] = 'renewable_storage'

    df_cap = df_cap.rename(index=mapping_dict, level='technology')
    df_cap = df_cap.groupby(level=['scenario', 'technology', 'mf']).sum()


    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows_cap = df_cap[df_cap.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows_cap = inv_rows_cap.reindex(order_inv, level=2)
    inv_rows_cap = inv_rows_cap.loc[idx[:, :, projections], :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows_cap = df_cap[~df_cap.index.get_level_values('mf').str.endswith('_inv')]
    op_rows_cap = op_rows_cap.reindex(order_op, level=2)
    # Identifying the first non-NaN value in each 'mf' row
    op_rows_cap = op_rows_cap.groupby(level='technology').sum()

    inv_cap_dataframes = {}
    for i, entry in enumerate(projections):
        num = int(entry.split('_')[0])
        # Create and store the DataFrame in the dictionary with a dynamic key
        df_temp = inv_rows_cap.loc[pd.IndexSlice[:, :, entry], :]

        # Replace columns up to that number from op_rows_cap, if num > 0
        if num > 0:
            df_temp.iloc[:, :num] = op_rows_cap.iloc[:, :num]

        inv_cap_dataframes[f'df_{num}'] = df_temp

    # ==================================================================================================================
    hatch = False
    something = True
    # plotting things
    def get_color_key(row_identifier):
        number_part = row_identifier.split('_')[0]
        return "df_" + number_part

    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # Choose a palette
    # palette = sns.color_palette("Set1", 7)  # 'Set2' for distinct, muted colors; change n_colors if more categories
    palette = sns.light_palette("seagreen", reverse=False, n_colors=256)
    palette2 = sns.dark_palette("seagreen", reverse=True, n_colors=256)
    # Create equally spaced indices to select colors
    indices = np.linspace(0, 255, num=15, dtype=int)
    selected_colors = np.array(palette)[indices]
    selected_colors2 = np.array(palette2)[indices]
    iterable_colors2 = list(map(tuple, selected_colors2))

    # Convert the palette to a list of RGB strings colors can be directly used in plotting functions
    # colors = [mcolors.to_hex(c) for c in palette]
    iterable_colors = list(map(tuple, selected_colors))
    iterable_colors.extend(iterable_colors2)

    colors = {
        'df_0': iterable_colors[4],
        'df_1': iterable_colors[3],
        'df_2': iterable_colors[14],
        'df_3': iterable_colors[14],
        'df_4': iterable_colors[14],
        'df_5': iterable_colors[8],
        'df_6': iterable_colors[9],
        'df_14': iterable_colors[21]
    }

    colors_emissions_green = {
    'df_0123': '#A8D5BA',  # Light Muted Green
    'df_0': iterable_colors[4],
    'df_41': '#8CBFA3',  # Light-Medium Muted Green
    'df_4': iterable_colors[14],
    'df_2': iterable_colors[14],
    'df_3': iterable_colors[14],
    'df_142': '#578776',  # Medium-Dark Muted Green
    'df_14': iterable_colors[21],
    'df_7': '#3E7061',  # Dark Muted Green
    'df_5': '#A8D5BA',  # Looping back to Light Muted Green
    'df_6': '#8CBFA3',  # Looping back to Light-Medium Muted Green
    'df_13': '#709F8C'  # Looping back to Medium Muted Green
    }

    colors_emissions = {
        'df_1': "#F2F2F2",  # Very Light Gray
        'df_0': "#CCCCCC",  # Light Gray
        'df_2': "#808080",  # Medium Gray
        'df_3': "#808080",  # Medium Gray
        'df_4': "#808080",  # Medium Gray
        'df_42': "#595959",  # Medium Dark Gray
        'df_5': "#333333",  # Dark Gray
        'df_14': "#000000"  # Black
    }

    def colors(index):
        if index == "Hydro":
            return "#0077BE"
        elif index == "Onshore Wind":
            return "#ADD8E6"
        elif index == "Offshore Wind":
            return "#6CA0DC"
        elif index == "Biomass":
            return "#19A519"
        elif index == "Photovoltaics":
            return "#FDFD96"
        elif index == "Heat Pump":
            return "#ff7f00"
        else:
            raise NotImplementedError(f"Technology-Color not implemented for '{index}'")

    # Plotting emissions on the primary y-axis
    legend_labels = [f"{2022 + 2 * int(row[1].split('_')[0])}" for row in inv_rows_emissions.index]
    legend_labels = legend_labels[:-1]
    i = 0
    if hatch == True:
        for row in inv_rows_emissions.index:
            color_key = get_color_key(row[1])
            if i < len(legend_labels):
                label = legend_labels[i]
            else:
                label = None
            ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors[color_key])
            i += 1
    else:
        for i, row in enumerate(inv_rows_emissions.index):
            if i < 5:
                color_key = get_color_key(row[1])
                if i < len(legend_labels):
                    label = legend_labels[i]
                else:
                    label = None
                ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors_emissions[color_key])
                i += 1

    first_y_value = op_rows_emissions[0] / 1000
    if hatch == True:
        ax1.plot(op_rows_emissions / 1000, label='Actual', color=iterable_colors[21], marker='', markersize=4,
                 markerfacecolor='black', lw=3)
        ax1.plot([-0.5, 0], [0, first_y_value], color=iterable_colors[21], lw=3)
    else:
        ax1.plot(op_rows_emissions / 1000, label='2050', color='black', marker='', markersize=4,
                 markerfacecolor='black', lw=3) # [:6]
        ax1.plot([-0.5, 0], [0, first_y_value], color='black', lw=3)
    # Setting the labels for the primary y-axis
    ax1.set_ylabel('Cumulative Carbon Emissions [Gt CO₂]', fontsize=12)

    # Carbon budget line on the primary y-axis
    carbon_budget = 16.58
    ax1.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0, xmax=0.95, zorder=1, alpha=0.5)
    # ax1.text(2.4, carbon_budget + 0.025 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='left', color='black',
    #          fontsize=10)
    # Define the label position
    label_x = 2.4
    label_y = carbon_budget + 0.025 * plt.ylim()[1]  # Position for the text above the carbon budget line

    # Place the text label
    ax1.text(label_x, label_y, "Carbon Emission Budget", va='bottom', ha='left', color='black', fontsize=10)

    # Draw a vertical connecting line from the label straight down to the carbon budget line
    ax1.plot([label_x, label_x], [carbon_budget, label_y], color='black', linestyle='-', linewidth=0.5)

    # Create secondary y-axis for the capacities
    ax2 = ax1.twinx()
    # Number of DataFrames (scenarios) and an example of time steps (assuming they're consistent across DataFrames)
    num_dfs = len(inv_cap_dataframes)
    time_steps = list(
        inv_cap_dataframes[next(iter(inv_cap_dataframes))].columns)  # Grab time steps from the first DataFrame
    bar_width = 0.8 / num_dfs  # Width of each bar, divided by the number of scenarios
    opacity = 0.5  # Adjust as needed for visibility

    # Define hatch patterns and edge color
    hatch_patterns = ['***', 'XX', '//////', r'\\\\\\', '']
    edge_color = 'black'  # Color for the edges of each bar segment

    for i, time_step in enumerate(time_steps):
        for j, (df_name, df) in enumerate(inv_cap_dataframes.items()):
            if j < 5:
                df_number = int(df_name.split('_')[1])
                bar_position = np.arange(len(time_steps)) + (j - len(inv_cap_dataframes) / 2) * bar_width + (bar_width / 2)
                data = df[time_step]

                bottom = np.zeros(len(data))
                edge_color = 'black'

                if hatch == True:
                    bar_color = colors.get(df_name, 'black')  # Fallback color
                    for k, category in enumerate(data.index):
                        # if time_step < df_number and df_number != 14:
                        #     bar_color = 'none'
                        #     edge_color = 'lightgrey' # 'dimgrey'
                        # Cycle through hatch patterns based on technology's position
                        hatch = hatch_patterns[k % len(hatch_patterns)]
                        ax2.bar(bar_position[i], data.loc[category]/1000, bottom=bottom, width=bar_width, alpha=opacity,
                                color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}', hatch=hatch)
                        bottom += data.loc[category]/1000
                else:
                    # Function to convert hex to RGB
                    def hex_to_rgb(hex_color):
                        hex_color = hex_color.lstrip('#')
                        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

                    # Function to convert RGB back to hex
                    def rgb_to_hex(rgb_color):
                        return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

                    # Function to darken the color
                    def adjust_color(color, factor):
                        rgb_color = hex_to_rgb(color)
                        darker_color = tuple(np.clip(np.array(rgb_color) * factor, 0, 255))
                        return rgb_to_hex(darker_color)

                    for k, category in enumerate(reversed(data.index)):
                        original_color = colors(category[1])
                        # Adjust the color to be darker based on `j`
                        factor = 1 - 0.2 * j  # Adjust the factor according to your needs
                        bar_color = adjust_color(original_color, factor)
                        ax2.bar(bar_position[i], data.loc[category] / 1000, bottom=bottom, width=bar_width, alpha=opacity,
                                color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}')
                        bottom += data.loc[category] / 1000

    # Setting the label for the secondary y-axis
    ax2.set_ylabel('Projected Renewable Generation Capacity [TW]', fontsize=12, labelpad=15)
    ax2.set_ylim(0, 1.99)
    ax1.set_ylim(0, 18)

    # Legend and titlefee
    # generate figure title:
    fig_title = model_name
    plt.title(fig_title)
    legend_techs = ["Heat Pump", "Biomass"] #  , "renewable_storage"
    # # Create custom legend handles based on technologies and their corresponding hatch patterns
    if hatch == True:
        legend_handles = [patches.Patch(facecolor='white', edgecolor='black', hatch=hatch_patterns[i], label=legend_techs[i])
                          for i, technology in enumerate(legend_techs)]
    else:
        legend_handles = [patches.Patch(facecolor=colors(technology), edgecolor='black', label=technology)
                          for technology in legend_techs]

    # For better control, manually specify the legend to include labels from both plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    inv_horizon, op_horizon = desired_scenarios[0].split('_')
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1),
               title='Foresight Horizon\nInvestment: ' + str(int(inv_horizon) * 2) + 'yr' + '\nOperation: ' + str(
                   int(op_horizon) * 2) + 'yr\n\nDecision-Maker\nPerspective in:', framealpha=1)

    if hatch == True:
        # After all plotting commands, manually add the legend with custom handles
        plt.legend(handles=legend_handles, title="Renewable Heat Generation Capacities", loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5) #
    else:
        plt.legend(handles=legend_handles, title="Renewable Heat Generation Capacities", loc='upper center',
                   bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)

    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    ax1.spines['left'].set_position(('data', -0.5))
    ax1.set_xlim([-0.5, 14.5])

    # Modify x-axis tick labels if needed
    # This part may require adjustment depending on your data's time format
    new_labels = [2022 + 2 * int(label) for label in ax1.get_xticks()]
    whole_numbers = range(15)
    ax1.set_xticklabels(new_labels)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

    # test commit changes

    plt.show(block=True)
    return

def feedback_plot_progression_with_techs_transport(model_name):
    print("plotting feedback_plot_progression")
    if type(model_name) is not str:
        model_name = str(model_name)
    r, config = read_in_results(model_name)

    projections =  ['0_inv', '3_inv', '4_inv', '14_inv'] # '2_inv','6_inv',

    df_emissions = r.get_total("carbon_emissions_cumulative", keep_raw=True)

    # indexing
    df_emissions = indexmapping(df_emissions, special_model_name=model_name)
    # Select only the desired rows
    if desired_scenarios is not None:
        df_emissions = df_emissions.loc[desired_scenarios]

    assert len(df_emissions.columns) == 15, "Number of timesteps not equal to 15"
    # Order for 'actual' and '_inv' rows separately as you requested
    order_op = [str(i) for i in range(15)]  # '0' to '14'
    order_inv = [f"{i}_inv" for i in range(15)]  # '0_inv' to '14_inv'

    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows_emissions = df_emissions[df_emissions.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows_emissions = inv_rows_emissions.reindex(order_inv, level=1)
    inv_rows_emissions = inv_rows_emissions.loc[(slice(None),projections), :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows_emissions = df_emissions[~df_emissions.index.get_level_values('mf').str.endswith('_inv')]
    op_rows_emissions = op_rows_emissions.reindex(order_op, level=1)
    # Identifying the first non-NaN value in each 'mf' row
    op_rows_emissions = op_rows_emissions.apply(
        lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1)
    op_rows_emissions = op_rows_emissions.values.flatten()

    initial_midpoint = 0
    # Incorporate the initial midpoint with the rest
    midpoints_op_emissions = [initial_midpoint] + [(op_rows_emissions[i] + op_rows_emissions[i + 1]) / 2 for i
                                                            in range(len(op_rows_emissions) - 1)]
    new_columns = np.arange(-0.5, inv_rows_emissions.columns.max() + 0.5, 0.5)
    all_columns = sorted(set(inv_rows_emissions.columns).union(new_columns))
    inv_rows_emissions = inv_rows_emissions.reindex(columns=all_columns)

    first_non_nan_indices = inv_rows_emissions.apply(lambda row: row.first_valid_index(), axis=1)

    for (indx, first_non_nan_index) in first_non_nan_indices.items():
        # Convert the index to an actual position; assuming it directly matches your description
        position = float(first_non_nan_index)
        # Find the corresponding value in midpoints_op_emissions
        value_to_add = midpoints_op_emissions[int(position)]
        # Determine the target column ("position - 0.5")
        target_column = position - 0.5
        assert pd.isna(inv_rows_emissions.at[indx, target_column]), "Expected NaN value not found"
        # Check if the target column exists in the DataFrame
        if target_column in inv_rows_emissions.columns:
            # Add the value to the specified position
            inv_rows_emissions.at[indx, target_column] = value_to_add

    # =================================================================================================================
    # do the same for capacity expansion:
    df_cap = r.get_total("capacity", keep_raw=True)
    assert len(df_cap.columns) == 15, "Number of timesteps not equal to 15"

    # Reset the index to work with it as columns
    df_cap.reset_index(inplace=True)
    print("reset complete")
    # The 'level_0' column contains the scenario information, so we will rename it
    df_cap.rename(columns={'level_0': 'scenario'}, inplace=True)

    # Now we set the multi-index again, this time including the 'scenario'
    df_cap.set_index(['scenario', 'technology', 'capacity_type', 'location', 'mf'], inplace=True)
    print("re-index complete")
    # Perform the aggregation by summing over the 'node' level of the index
    df_cap = df_cap.groupby(level=['scenario', 'technology', 'capacity_type', 'mf']).sum()
    print("aggregation complete")
    # indexing
    df_cap = indexmapping(df_cap, special_model_name=model_name)
    # reducing to the desired scenarios
    df_cap = df_cap.loc[df_cap.index.get_level_values('scenario').isin(desired_scenarios)]
    # get rid of capacit_type "energy"
    df_cap = df_cap[df_cap.index.get_level_values('capacity_type') != 'energy']
    df_cap.index = df_cap.index.droplevel('capacity_type')
    unique_technologies = df_cap.index.get_level_values('technology').unique()
    list_of_technologies = unique_technologies.tolist()
    # List of technologies to remove
    renewable_heating = ['biomass_boiler', 'biomass_boiler_DH',
                         'heat_pump', 'heat_pump_DH']
    renewable_transport = ['BEV']
    # wind = ['wind_offshore', 'wind_onshore']
    biomass = ['biomass_boiler', 'biomass_boiler_DH']
    heat_pump = ['heat_pump', 'heat_pump_DH']

    # renewable_storage = ['battery', 'pumped_hydro', 'hydrogen_storage']
    technologies_to_keep = renewable_transport # + renewable_heating# + renewable_storage
    df_cap = df_cap.loc[(slice(None), technologies_to_keep), :]

    mapping_dict = {}
    for tech in biomass:
        mapping_dict[tech] = 'Biomass'
    for tech in heat_pump:
        mapping_dict[tech] = 'Heat Pump'

    # for tech in wind:
    #     mapping_dict[tech] = 'Wind'

    # for tech in renewable_storage:
    #     mapping_dict[tech] = 'renewable_storage'

    df_cap = df_cap.rename(index=mapping_dict, level='technology')
    df_cap = df_cap.groupby(level=['scenario', 'technology', 'mf']).sum()


    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows_cap = df_cap[df_cap.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows_cap = inv_rows_cap.reindex(order_inv, level=2)
    inv_rows_cap = inv_rows_cap.loc[idx[:, :, projections], :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows_cap = df_cap[~df_cap.index.get_level_values('mf').str.endswith('_inv')]
    op_rows_cap = op_rows_cap.reindex(order_op, level=2)
    # Identifying the first non-NaN value in each 'mf' row
    op_rows_cap = op_rows_cap.groupby(level='technology').sum()

    inv_cap_dataframes = {}
    for i, entry in enumerate(projections):
        num = int(entry.split('_')[0])
        # Create and store the DataFrame in the dictionary with a dynamic key
        df_temp = inv_rows_cap.loc[pd.IndexSlice[:, :, entry], :]

        # Replace columns up to that number from op_rows_cap, if num > 0
        if num > 0:
            df_temp.iloc[:, :num] = op_rows_cap.iloc[:, :num]

        inv_cap_dataframes[f'df_{num}'] = df_temp

    # ==================================================================================================================
    hatch = False
    something = True
    # plotting things
    def get_color_key(row_identifier):
        number_part = row_identifier.split('_')[0]
        return "df_" + number_part

    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # Choose a palette
    # palette = sns.color_palette("Set1", 7)  # 'Set2' for distinct, muted colors; change n_colors if more categories
    palette = sns.light_palette("seagreen", reverse=False, n_colors=256)
    palette2 = sns.dark_palette("seagreen", reverse=True, n_colors=256)
    # Create equally spaced indices to select colors
    indices = np.linspace(0, 255, num=15, dtype=int)
    selected_colors = np.array(palette)[indices]
    selected_colors2 = np.array(palette2)[indices]
    iterable_colors2 = list(map(tuple, selected_colors2))

    # Convert the palette to a list of RGB strings colors can be directly used in plotting functions
    # colors = [mcolors.to_hex(c) for c in palette]
    iterable_colors = list(map(tuple, selected_colors))
    iterable_colors.extend(iterable_colors2)

    colors = {
        'df_0': iterable_colors[4],
        'df_1': iterable_colors[3],
        'df_2': iterable_colors[14],
        'df_3': iterable_colors[14],
        'df_4': iterable_colors[14],
        'df_5': iterable_colors[8],
        'df_6': iterable_colors[9],
        'df_14': iterable_colors[21]
    }

    colors_emissions_green = {
    'df_0123': '#A8D5BA',  # Light Muted Green
    'df_0': iterable_colors[4],
    'df_41': '#8CBFA3',  # Light-Medium Muted Green
    'df_4': iterable_colors[14],
    'df_2': iterable_colors[14],
    'df_3': iterable_colors[14],
    'df_142': '#578776',  # Medium-Dark Muted Green
    'df_14': iterable_colors[21],
    'df_7': '#3E7061',  # Dark Muted Green
    'df_5': '#A8D5BA',  # Looping back to Light Muted Green
    'df_6': '#8CBFA3',  # Looping back to Light-Medium Muted Green
    'df_13': '#709F8C'  # Looping back to Medium Muted Green
    }

    colors_emissions = {
        'df_1': "#F2F2F2",  # Very Light Gray
        'df_0': "#CCCCCC",  # Light Gray
        'df_2': "#808080",  # Medium Gray
        'df_3': "#808080",  # Medium Gray
        'df_4': "#808080",  # Medium Gray
        'df_42': "#595959",  # Medium Dark Gray
        'df_5': "#333333",  # Dark Gray
        'df_14': "#000000"  # Black
    }

    def colors(index):
        if index == "Hydro":
            return "#0077BE"
        elif index == "Onshore Wind":
            return "#ADD8E6"
        elif index == "Offshore Wind":
            return "#6CA0DC"
        elif index == "Biomass":
            return "#19A519"
        elif index == "Photovoltaics":
            return "#FDFD96"
        elif index == "Heat Pump":
            return "#ff7f00"
        elif index == "BEV":
            return "#800080"
        else:
            raise NotImplementedError(f"Technology-Color not implemented for '{index}'")

    # Plotting emissions on the primary y-axis
    legend_labels = [f"{2022 + 2 * int(row[1].split('_')[0])}" for row in inv_rows_emissions.index]
    legend_labels = legend_labels[:-1]
    i = 0
    if hatch == True:
        for row in inv_rows_emissions.index:
            color_key = get_color_key(row[1])
            if i < len(legend_labels):
                label = legend_labels[i]
            else:
                label = None
            ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors[color_key])
            i += 1
    else:
        for i, row in enumerate(inv_rows_emissions.index):
            if i < 5:
                color_key = get_color_key(row[1])
                if i < len(legend_labels):
                    label = legend_labels[i]
                else:
                    label = None
                ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors_emissions[color_key])
                i += 1

    first_y_value = op_rows_emissions[0] / 1000
    if hatch == True:
        ax1.plot(op_rows_emissions / 1000, label='Actual', color=iterable_colors[21], marker='', markersize=4,
                 markerfacecolor='black', lw=3)
        ax1.plot([-0.5, 0], [0, first_y_value], color=iterable_colors[21], lw=3)
    else:
        ax1.plot(op_rows_emissions / 1000, label='2050', color='black', marker='', markersize=4,
                 markerfacecolor='black', lw=3) # [:6]
        ax1.plot([-0.5, 0], [0, first_y_value], color='black', lw=3)
    # Setting the labels for the primary y-axis
    ax1.set_ylabel('Cumulative Carbon Emissions [Gt CO₂]', fontsize=12)

    # Carbon budget line on the primary y-axis
    carbon_budget = 16.58
    ax1.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0, xmax=0.95, zorder=1, alpha=0.5)
    # ax1.text(2.4, carbon_budget + 0.025 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='left', color='black',
    #          fontsize=10)
    # Define the label position
    label_x = 2.4
    label_y = carbon_budget + 0.025 * plt.ylim()[1]  # Position for the text above the carbon budget line

    # Place the text label
    ax1.text(label_x, label_y, "Carbon Emission Budget", va='bottom', ha='left', color='black', fontsize=10)

    # Draw a vertical connecting line from the label straight down to the carbon budget line
    ax1.plot([label_x, label_x], [carbon_budget, label_y], color='black', linestyle='-', linewidth=0.5)

    # Create secondary y-axis for the capacities
    ax2 = ax1.twinx()
    # Number of DataFrames (scenarios) and an example of time steps (assuming they're consistent across DataFrames)
    num_dfs = len(inv_cap_dataframes)
    time_steps = list(
        inv_cap_dataframes[next(iter(inv_cap_dataframes))].columns)  # Grab time steps from the first DataFrame
    bar_width = 0.8 / num_dfs  # Width of each bar, divided by the number of scenarios
    opacity = 0.5  # Adjust as needed for visibility

    # Define hatch patterns and edge color
    hatch_patterns = ['***', 'XX', '//////', r'\\\\\\', '']
    edge_color = 'black'  # Color for the edges of each bar segment

    for i, time_step in enumerate(time_steps):
        for j, (df_name, df) in enumerate(inv_cap_dataframes.items()):
            if j < 5:
                df_number = int(df_name.split('_')[1])
                bar_position = np.arange(len(time_steps)) + (j - len(inv_cap_dataframes) / 2) * bar_width + (bar_width / 2)
                data = df[time_step]

                bottom = np.zeros(len(data))
                edge_color = 'black'

                if hatch == True:
                    bar_color = colors.get(df_name, 'black')  # Fallback color
                    for k, category in enumerate(data.index):
                        # if time_step < df_number and df_number != 14:
                        #     bar_color = 'none'
                        #     edge_color = 'lightgrey' # 'dimgrey'
                        # Cycle through hatch patterns based on technology's position
                        hatch = hatch_patterns[k % len(hatch_patterns)]
                        ax2.bar(bar_position[i], data.loc[category]/1000, bottom=bottom, width=bar_width, alpha=opacity,
                                color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}', hatch=hatch)
                        bottom += data.loc[category]/1000
                else:
                    # Function to convert hex to RGB
                    def hex_to_rgb(hex_color):
                        hex_color = hex_color.lstrip('#')
                        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

                    # Function to convert RGB back to hex
                    def rgb_to_hex(rgb_color):
                        return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))

                    # Function to darken the color
                    def adjust_color(color, factor):
                        rgb_color = hex_to_rgb(color)
                        darker_color = tuple(np.clip(np.array(rgb_color) * factor, 0, 255))
                        return rgb_to_hex(darker_color)

                    for k, category in enumerate(reversed(data.index)):
                        original_color = colors(category[1])
                        # Adjust the color to be darker based on `j`
                        factor = 1 - 0.2 * j  # Adjust the factor according to your needs
                        bar_color = adjust_color(original_color, factor)
                        ax2.bar(bar_position[i], data.loc[category] / 1000, bottom=bottom, width=bar_width, alpha=opacity,
                                color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}')
                        bottom += data.loc[category] / 1000

    # Setting the label for the secondary y-axis
    ax2.set_ylabel('Projected Renewable Generation Capacity [TW]', fontsize=12, labelpad=15)
    ax2.set_ylim(0, 1.39)
    ax1.set_ylim(0, 18)

    # Legend and titlefee
    # generate figure title:
    fig_title = model_name
    plt.title(fig_title)
    legend_techs = ["BEV"] #  , "renewable_storage"
    # # Create custom legend handles based on technologies and their corresponding hatch patterns
    if hatch == True:
        legend_handles = [patches.Patch(facecolor='white', edgecolor='black', hatch=hatch_patterns[i], label=legend_techs[i])
                          for i, technology in enumerate(legend_techs)]
    else:
        legend_handles = [patches.Patch(facecolor=colors(technology), edgecolor='black', label=technology)
                          for technology in legend_techs]

    # For better control, manually specify the legend to include labels from both plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    inv_horizon, op_horizon = desired_scenarios[0].split('_')
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1),
               title='Foresight Horizon\nInvestment: ' + str(int(inv_horizon) * 2) + 'yr' + '\nOperation: ' + str(
                   int(op_horizon) * 2) + 'yr\n\nDecision-Maker\nPerspective in:', framealpha=1)

    if hatch == True:
        # After all plotting commands, manually add the legend with custom handles
        plt.legend(handles=legend_handles, title="Renewable Transport Capacities", loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5) #
    else:
        plt.legend(handles=legend_handles, title="Renewable Transport Capacities", loc='upper center',
                   bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)

    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    ax1.spines['left'].set_position(('data', -0.5))
    ax1.set_xlim([-0.5, 14.5])

    # Modify x-axis tick labels if needed
    # This part may require adjustment depending on your data's time format
    new_labels = [2022 + 2 * int(label) for label in ax1.get_xticks()]
    whole_numbers = range(15)
    ax1.set_xticklabels(new_labels)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

    # test commit changes

    plt.show(block=True)
    return


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
model_name = "PC_ct_pass_tra_base"

# model_name = "PC_ct_pass_tra_ETS1and2"

# model_name = ("PC_ct_pass_tra_ETS1only")

# model_name = ("PC_ct_pass_tra_cap_add_target")

# model_name = ("PC_ct_pass_tra_gen_target")

# model_name = ("PC_ct_pass_tra_coal_cap_phaseout")

# model_name = ("PC_ct_pass_tra_coal_gen_phaseout")



desired_scenarios = ['15_1'] # ['5_1']
assert len(desired_scenarios) == 1, "too many scenarios for this plot"

# Plots: (electricity, heat, BEV)
feedback_plot_progression_with_techs(model_name)

feedback_plot_progression_with_techs_heat(model_name)

feedback_plot_progression_with_techs_transport(model_name)
