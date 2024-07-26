def feedback_pubplot_progression():
    global model_name
    print("plotting feedback_plot_progression")
    if type(model_name) is not str:
        model_name = str(model_name)
    r, config = read_in_results(model_name)

    projections =  ['0_inv', '4_inv', '6_inv', '14_inv'] # '2_inv',

    df_emissions = r.get_total("carbon_emissions_cumulative", keep_raw=True)

    # indexing
    df_emissions = indexmapping(df_emissions)
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
    renewable_el_generation = ['biomass_plant', 'biomass_plant_CCS',
                              'photovoltaics', 'reservoir_hydro', 'run-of-river_hydro',
                               'wind_offshore', 'wind_onshore']
    others = ['oil_plant', 'waste_plant']
    renewable_heating = ['biomass_boiler', 'biomass_boiler_DH', 'district_heating_grid',
                         'heat_pump', 'heat_pump_DH', 'waste_boiler_DH']
    # renewable_storage = ['battery', 'pumped_hydro', 'hydrogen_storage']
    technologies_to_keep = renewable_el_generation # + renewable_heating + renewable_storage
    df_cap = df_cap.loc[(slice(None), technologies_to_keep), :]
    #
    mapping_dict = {}
    for tech in others:
        mapping_dict[tech] = 'others'
    # for tech in renewable_heating:
    #     mapping_dict[tech] = 'renewable_heating'
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
    # plotting things

    # Create figure and axes for the two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), sharex=True)

    # Plotting the lineplot on ax1
    legend_labels = [f"{2022 + 2 * int(row[1].split('_')[0])}" for row in inv_rows_emissions.index]
    legend_labels = legend_labels[:-1]
    for i, row in enumerate(inv_rows_emissions.index):
        label = legend_labels[i] if i < len(legend_labels) else None
        ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index,
                 ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label)

    # Additional line plot settings
    ax1.plot(op_rows_emissions / 1000, label='Actual', color='black', marker='', markersize=4, markerfacecolor='black',
             lw=3)
    first_y_value = op_rows_emissions[0] / 1000
    ax1.plot([-0.5, 0], [0, first_y_value], color='black', lw=3)
    ax1.set_ylabel('Cumulative Carbon Emissions [Gt CO₂]', fontsize=12)
    carbon_budget = 12.2324
    ax1.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0, xmax=0.95, zorder=1, alpha=0.5)
    ax1.text(2.25, carbon_budget + 0.005 * ax1.get_ylim()[1], "Carbon Emission Budget", va='bottom', ha='left',
             color='black', fontsize=10)

    # Plotting the stacked barchart on ax2
    num_dfs = len(inv_cap_dataframes)
    time_steps = list(inv_cap_dataframes[next(iter(inv_cap_dataframes))].columns)
    bar_width = 0.8 / num_dfs
    opacity = 0.5
    scaling_factor = 0.1

    for i, time_step in enumerate(time_steps):
        for j, (df_name, df) in enumerate(inv_cap_dataframes.items()):
            df_number = int(df_name.split('_')[1])
            bar_position = np.arange(len(time_steps)) + (j - len(inv_cap_dataframes) / 2) * bar_width + (bar_width / 2)
            data = df[time_step] * scaling_factor
            bottom = np.zeros(len(data))
            for k, category in enumerate(reversed(data.index)):
                ax2.bar(bar_position[i], data.loc[category], bottom=bottom, width=bar_width, alpha=opacity)
                bottom += data.loc[category]

    ax2.set_ylabel('Projected Renewable Generation Capacity [TW]', fontsize=12, labelpad=15)
    ax2.set_ylim(0, 3500 * scaling_factor)  # Adjusted for the scaling factor

    # Adjusting plot layout
    plt.subplots_adjust(hspace=0.4)  # Adjust spacing between subplots
    new_labels = [2022 + 2 * int(label) for label in ax1.get_xticks()]
    ax2.set_xticklabels(new_labels, rotation=45)  # Ensure consistent x-axis labels
    plt.subplots_adjust(hspace=0)
    plt.show(block=True)
    return
