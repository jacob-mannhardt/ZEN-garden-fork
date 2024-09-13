from plotting.helpers import *

def generate_df():
    data = """
    cost_capex_total 15_5_PC_ct_pass_tra_ETS1and2          657.126544  652.392386  674.196947  710.836220  749.890395  785.639601   814.969855   856.475827   922.194585   976.248905   953.246023   936.824628   910.115784   894.612727   885.708408
    cost_capex_total 15_5_PC_ct_pass_tra_base              652.800563  645.451311  667.399040  705.696302  756.608183  806.788302   832.801246   900.111026   919.160077   928.005908   965.189134   976.904560   941.123773   904.540244   883.234181
    cost_capex_total 15_5_PC_ct_pass_tra_cap_add_target    652.557918  645.713261  667.987855  713.506493  757.441456  808.696728   827.898849   894.843262   920.552806   921.355856   952.229081   967.131916   932.757620   893.556826   883.368713
    cost_capex_total 15_5_PC_ct_pass_tra_gen_target        670.831581  655.103572  677.447004  716.210534  764.239399  779.024321   803.058441   827.593439   844.384339   852.995267   866.494145   866.027701   866.804775   867.334939   871.417596
    cost_capex_total 5_1_PC_ct_pass_tra_ETS1and2           650.140545  633.729126  634.215040  666.034897  723.776948  877.269574  1080.737389  1136.555502  1207.138009  1345.944308  1356.078366  1324.123978  1280.396185  1236.847108  1185.239551
    cost_capex_total 5_1_PC_ct_pass_tra_base               647.891686  632.254021  641.278297  680.817749  769.013649  949.023066  1127.078580  1160.495605  1232.635250  1372.264019  1373.264664  1340.783593  1294.168041  1247.499717  1174.785819
    cost_capex_total 5_1_PC_ct_pass_tra_cap_add_target     646.880332  629.480830  641.210457  696.126017  737.207659  813.395155   920.011359  1011.408475  1129.229163  1177.210860  1124.722721  1091.091710  1052.139056   996.148147   981.456703
    cost_capex_total 5_1_PC_ct_pass_tra_gen_target         669.100961  650.935518  657.535419  693.949389  753.516512  810.288600   934.006690  1078.931191  1083.072940  1053.326611   993.433491   982.791204   958.356247   939.839668   933.048474
    cost_opex_total  15_5_PC_ct_pass_tra_ETS1and2          451.727298  365.463099  341.337169  325.479980  309.622023  292.268390   276.333626   261.365388   241.747065   227.801918   222.093506   220.240191   219.580561   218.910640   217.871616
    cost_opex_total  15_5_PC_ct_pass_tra_base              450.310076  368.136227  345.404621  327.752076  311.648042  295.712374   283.367604   262.677788   247.367022   237.574007   226.173016   220.820109   220.079836   219.375830   218.018733
    cost_opex_total  15_5_PC_ct_pass_tra_cap_add_target    450.321684  367.941803  345.218984  326.822214  310.723863  294.434319   283.298035   262.633479   246.423418   237.332702   226.696210   221.054359   220.116549   219.369596   218.135085
    cost_opex_total  15_5_PC_ct_pass_tra_gen_target      11956.047657  379.270771  349.544880  329.243865  308.466938  296.052294   283.761456   269.832184   255.729511   243.161753   226.353614   224.155484   221.113087   218.699499   217.816754
    cost_opex_total  5_1_PC_ct_pass_tra_ETS1and2           454.569638  373.068465  357.771985  345.440198  324.144759  297.784199   282.500975   284.481360   267.134080   245.339257   236.378569   233.564214   232.772128   230.726342   228.001945
    cost_opex_total  5_1_PC_ct_pass_tra_base               452.893086  370.315431  353.061464  337.679173  319.821229  301.551472   299.785055   285.935237   268.145191   246.249971   237.596352   234.600267   232.543791   229.549615   226.996908
    cost_opex_total  5_1_PC_ct_pass_tra_cap_add_target     453.730434  372.543754  354.493825  338.085774  324.594338  313.762687   308.474929   286.452991   258.612844   246.625565   241.884475   238.432511   236.194128   232.560559   228.508546
    cost_opex_total  5_1_PC_ct_pass_tra_gen_target       11955.592198  376.720876  359.074826  339.807858  317.140118  299.032551   278.456873   261.145715   250.038955   244.218658   238.516052   235.934124   233.822313   232.949877   230.919757
    """
    # Split the data into rows
    data_lines = data.strip().split("\n")

    # Create a list for the multi-index and data
    multi_index = []
    data_values = []

    # Split each row into its components (multi-index and values)
    for line in data_lines:
        parts = line.split()
        multi_index.append((parts[0], parts[1]))  # Multi-index: first two parts
        data_values.append([float(x) for x in parts[2:]])  # Data: remaining parts

    # Now the multi_index should have the correct categories and sub_categories
    index = pd.MultiIndex.from_tuples(multi_index, names=["category", "sub_category"])

    # Ensure all rows have the same length (filling missing values with NaN if needed)
    max_len = max(len(row) for row in data_values)
    for row in data_values:
        while len(row) < max_len:
            row.append(np.nan)

    # Create the DataFrame with the multi-index and consistent data
    df = pd.DataFrame(data=np.array(data_values), index=index)

    # Define the column headers based on the data length
    df.columns = range(0, max_len)

    return df

def comparative_costs_over_time_5(model_name):
    discounted = False
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]

    dfs = []
    for name in model_name:
        if discounted == True:
            r, config = read_in_special_results(name)
            df_OPEX, df_CAPEX = r.get_npc()
            temp_dfs = []
            for i, df in enumerate([df_OPEX, df_CAPEX]):
                names = ['cost_opex_total', 'cost_capex_total']
                df = indexmapping(df, special_model_name=name)
                df_selected = df.iloc[:, ::2]
                new_column_names = range(15)
                df_selected.columns = new_column_names
                df_selected.index = pd.MultiIndex.from_product([[names[i]], df_selected.index])
                if i == 0:
                    df_OPEX = df_selected
                    temp_dfs.append(df_OPEX)
                else:
                    df_CAPEX = df_selected
                    temp_dfs.append(df_CAPEX)
            df = pd.concat(temp_dfs, axis=0, join='outer')
        else:
            r, config = read_in_results(name)

            costs = ["cost_opex_total", "cost_carrier_total", "cost_capex_total"]
            df = {}
            for c in costs:
                df[c] = r.get_total(c)
                df[c] = indexmapping(df[c], special_model_name=name)

            df = pd.concat(df, keys=df.keys())
            df.loc['cost_opex_total'] = (df.loc['cost_opex_total'].values + df.loc['cost_carrier_total'].values)
            df = df.drop('cost_carrier_total')
            df.index = pd.MultiIndex.from_tuples([(first, second + "_" + name) for first, second in df.index])

        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df = df /1000
    df = df.sort_index()
    df = df[~df.index.get_level_values(1).str.startswith(('15_1', '15_5'))]


    # df = generate_df()
    df.columns = rename_columns(df.columns)

    # Set the color palette
    colors = sns.color_palette('pastel')[:2]  # Choosing two colors from the 'pastel' palette
    saved_lines = {}
    saved_lines_capex = {}

    # Set up a 2x2 grid of plots with no space between them
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # axs = axs.flatten()

    colors = ['#1f77b4', '#ff7f0e']

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, name in enumerate(model_name):
        # For each subplot, filter df for the current scenario across both cost types
        model_data_5_5 = {cost_type: df.loc[cost_type, "5_5_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        model_data_5_1 = {cost_type: df.loc[cost_type, "5_1_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        capex_5_1 = model_data_5_1['cost_capex_total']
        opex_5_1 = model_data_5_1['cost_opex_total'] + capex_5_1

        capex_5_5 = model_data_5_5['cost_capex_total']
        opex_5_5 = model_data_5_5['cost_opex_total'] + capex_5_5

        # Plot CAPEX (fill between x-axis and CAPEX data using the original color)
        axs[i].fill_between(capex_5_1.index, 0, capex_5_1, label='CAPEX 5_1', color=colors[0], alpha=0.5)

        # Plot OPEX (fill between CAPEX and OPEX using the original color)
        axs[i].fill_between(capex_5_1.index, capex_5_1, opex_5_1, label='OPEX 5_1', color=colors[1], alpha=0.5)

        # Plot CAPEX for 15_5 with a dotted line
        axs[i].plot(capex_5_5.index, capex_5_5, label='CAPEX 5_5', linestyle=':', color="black", linewidth=2)

        # Plot OPEX for 15_5 with a dashed line
        axs[i].plot(opex_5_5.index, opex_5_5, label='OPEX 5_5', linestyle='--', color="black", linewidth=2)

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 2000)

        # Set x-axis ticks every two years from 2022 to 2050
        axs[i].set_xticks(np.arange(2022, 2052, 2))
        axs[i].set_xlim([2020, 2052])

        # Get the current tick labels and only display every second one
        tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(np.arange(2022, 2052, 2))]
        axs[i].set_xticklabels(tick_labels)

        # Enable minor ticks
        axs[i].minorticks_on()

        # Define the position of the minor ticks (every year)
        axs[i].set_xticks(np.arange(2021, 2051, 2), minor=True)

        # Customize the appearance of minor ticks
        axs[i].tick_params(axis='x', which='minor', length=2)

        if name == "PC_ct_pass_tra_base":
            model_name_str = "Base Scenario"
        elif name == "PC_ct_pass_tra_ETS1and2":
            model_name_str = "Emission Policy"
        elif name == "PC_ct_pass_tra_cap_add_target":
            model_name_str = "Investment Policy"
        elif name == "PC_ct_pass_tra_gen_target":
            model_name_str = "Operation Policy"
        axs[i].set_title(model_name_str + ', TSA:20', loc='left', x=0.05, y=0.9, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 1800)

    # Add a common x and y label to the entire figure
    fig.text(0.5, 0.04, 'Years', ha='center')
    fig.text(0.04, 0.5, 'Cost [billion €]', va='center', rotation='vertical')

    # Add legends
    for ax in axs:
        ax.legend(loc='lower left',bbox_to_anchor=(0.05, 0))

    # Adjust layout to make the subplots fit well together
    plt.subplots_adjust(hspace=0, wspace=0)

    # Display the plot
    plt.show(block=True)

def comparative_costs_over_time_15(model_name):
    discounted = False
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]

    dfs = []
    for name in model_name:
        if discounted == True:
            r, config = read_in_special_results(name)
            df_OPEX, df_CAPEX = r.get_npc()
            temp_dfs = []
            for i, df in enumerate([df_OPEX, df_CAPEX]):
                names = ['cost_opex_total', 'cost_capex_total']
                df = indexmapping(df, special_model_name=name)
                df_selected = df.iloc[:, ::2]
                new_column_names = range(15)
                df_selected.columns = new_column_names
                df_selected.index = pd.MultiIndex.from_product([[names[i]], df_selected.index])
                if i == 0:
                    df_OPEX = df_selected
                    temp_dfs.append(df_OPEX)
                else:
                    df_CAPEX = df_selected
                    temp_dfs.append(df_CAPEX)
            df = pd.concat(temp_dfs, axis=0, join='outer')
        else:
            r, config = read_in_results(name)

            costs = ["cost_opex_total", "cost_carrier_total", "cost_capex_total"]
            df = {}
            for c in costs:
                df[c] = r.get_total(c)
                df[c] = indexmapping(df[c], special_model_name=name)

            df = pd.concat(df, keys=df.keys())
            df.loc['cost_opex_total'] = (df.loc['cost_opex_total'].values + df.loc['cost_carrier_total'].values)
            df = df.drop('cost_carrier_total')
            df.index = pd.MultiIndex.from_tuples([(first, second + "_" + name) for first, second in df.index])

        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df = df /1000
    df = df.sort_index()
    df = df[~df.index.get_level_values(1).str.startswith(('5_5', '5_1'))]


    # df = generate_df()
    df.columns = rename_columns(df.columns)

    # Set the color palette
    colors = sns.color_palette('pastel')[:2]  # Choosing two colors from the 'pastel' palette
    saved_lines = {}
    saved_lines_capex = {}

    # Set up a 2x2 grid of plots with no space between them
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # axs = axs.flatten()

    colors = ['#1f77b4', '#ff7f0e']

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, name in enumerate(model_name):
        # For each subplot, filter df for the current scenario across both cost types
        model_data_15_5 = {cost_type: df.loc[cost_type, "15_5_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        model_data_15_1 = {cost_type: df.loc[cost_type, "15_1_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        capex_15_1 = model_data_15_1['cost_capex_total']
        opex_15_1 = model_data_15_1['cost_opex_total'] + capex_15_1

        capex_15_5 = model_data_15_5['cost_capex_total']
        opex_15_5 = model_data_15_5['cost_opex_total'] + capex_15_5

        # Plot CAPEX (fill between x-axis and CAPEX data using the original color)
        axs[i].fill_between(capex_15_1.index, 0, capex_15_1, label='CAPEX 15_1', color=colors[0], alpha=0.5)

        # Plot OPEX (fill between CAPEX and OPEX using the original color)
        axs[i].fill_between(capex_15_1.index, capex_15_1, opex_15_1, label='OPEX 15_1', color=colors[1], alpha=0.5)

        # Plot CAPEX for 15_5 with a dotted line
        axs[i].plot(capex_15_5.index, capex_15_5, label='CAPEX 15_5', linestyle=':', color="black", linewidth=2)

        # Plot OPEX for 15_5 with a dashed line
        axs[i].plot(opex_15_5.index, opex_15_5, label='OPEX 15_5', linestyle='--', color="black", linewidth=2)

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 2000)

        # Set x-axis ticks every two years from 2022 to 2050
        axs[i].set_xticks(np.arange(2022, 2052, 2))
        axs[i].set_xlim([2020, 2052])

        # Get the current tick labels and only display every second one
        tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(np.arange(2022, 2052, 2))]
        axs[i].set_xticklabels(tick_labels)

        # Enable minor ticks
        axs[i].minorticks_on()

        # Define the position of the minor ticks (every year)
        axs[i].set_xticks(np.arange(2021, 2051, 2), minor=True)

        # Customize the appearance of minor ticks
        axs[i].tick_params(axis='x', which='minor', length=2)

        if name == "PC_ct_pass_tra_base":
            model_name_str = "Base Scenario"
        elif name == "PC_ct_pass_tra_ETS1and2":
            model_name_str = "Emission Policy"
        elif name == "PC_ct_pass_tra_cap_add_target":
            model_name_str = "Investment Policy"
        elif name == "PC_ct_pass_tra_gen_target":
            model_name_str = "Operation Policy"
        axs[i].set_title(model_name_str + ', TSA:20', loc='left', x=0.05, y=0.9, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 1800)

    # Add a common x and y label to the entire figure
    fig.text(0.5, 0.04, 'Years', ha='center')
    fig.text(0.04, 0.5, 'Cost [billion €]', va='center', rotation='vertical')

    # Add legends
    for ax in axs:
        ax.legend(loc='lower left',bbox_to_anchor=(0.05, 0))

    # Adjust layout to make the subplots fit well together
    plt.subplots_adjust(hspace=0, wspace=0)

    # Display the plot
    plt.show(block=True)

def comparative_costs_over_time_extreme(model_name):
    discounted = False
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]

    dfs = []
    for name in model_name:
        if discounted == True:
            r, config = read_in_special_results(name)
            df_OPEX, df_CAPEX = r.get_npc()
            temp_dfs = []
            for i, df in enumerate([df_OPEX, df_CAPEX]):
                names = ['cost_opex_total', 'cost_capex_total']
                df = indexmapping(df, special_model_name=name)
                df_selected = df.iloc[:, ::2]
                new_column_names = range(15)
                df_selected.columns = new_column_names
                df_selected.index = pd.MultiIndex.from_product([[names[i]], df_selected.index])
                if i == 0:
                    df_OPEX = df_selected
                    temp_dfs.append(df_OPEX)
                else:
                    df_CAPEX = df_selected
                    temp_dfs.append(df_CAPEX)
            df = pd.concat(temp_dfs, axis=0, join='outer')
        else:
            r, config = read_in_results(name)

            costs = ["cost_opex_total", "cost_carrier_total", "cost_capex_total"]
            df = {}
            for c in costs:
                df[c] = r.get_total(c)
                df[c] = indexmapping(df[c], special_model_name=name)

            df = pd.concat(df, keys=df.keys())
            df.loc['cost_opex_total'] = (df.loc['cost_opex_total'].values + df.loc['cost_carrier_total'].values)
            df = df.drop('cost_carrier_total')
            df.index = pd.MultiIndex.from_tuples([(first, second + "_" + name) for first, second in df.index])

        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df = df /1000
    df = df.sort_index()
    df = df[~df.index.get_level_values(1).str.startswith(('15_1', '5_5'))]


    # df = generate_df()
    df.columns = rename_columns(df.columns)

    # Set the color palette
    colors = sns.color_palette('pastel')[:2]  # Choosing two colors from the 'pastel' palette
    saved_lines = {}
    saved_lines_capex = {}

    # Set up a 2x2 grid of plots with no space between them
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # axs = axs.flatten()

    colors = ['#1f77b4', '#ff7f0e']

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, name in enumerate(model_name):
        # For each subplot, filter df for the current scenario across both cost types
        model_data_15 = {cost_type: df.loc[cost_type, "15_5_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        model_data_5 = {cost_type: df.loc[cost_type, "5_1_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        capex_5_1 = model_data_5['cost_capex_total']
        opex_5_1 = model_data_5['cost_opex_total'] + capex_5_1

        capex_15_5 = model_data_15['cost_capex_total']
        opex_15_5 = model_data_15['cost_opex_total'] + capex_15_5

        # Plot CAPEX (fill between x-axis and CAPEX data using the original color)
        axs[i].fill_between(capex_5_1.index, 0, capex_5_1, label='CAPEX 5_1', color=colors[0], alpha=0.5)

        # Plot OPEX (fill between CAPEX and OPEX using the original color)
        axs[i].fill_between(capex_5_1.index, capex_5_1, opex_5_1, label='OPEX 5_1', color=colors[1], alpha=0.5)

        # Plot CAPEX for 15_5 with a dotted line
        axs[i].plot(capex_15_5.index, capex_15_5, label='CAPEX 15_5', linestyle=':', color="black", linewidth=2)

        # Plot OPEX for 15_5 with a dashed line
        axs[i].plot(opex_15_5.index, opex_15_5, label='OPEX 15_5', linestyle='--', color="black", linewidth=2)

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 2000)

        # Set x-axis ticks every two years from 2022 to 2050
        axs[i].set_xticks(np.arange(2022, 2052, 2))
        axs[i].set_xlim([2020, 2052])

        # Get the current tick labels and only display every second one
        tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(np.arange(2022, 2052, 2))]
        axs[i].set_xticklabels(tick_labels)

        # Enable minor ticks
        axs[i].minorticks_on()

        # Define the position of the minor ticks (every year)
        axs[i].set_xticks(np.arange(2021, 2051, 2), minor=True)

        # Customize the appearance of minor ticks
        axs[i].tick_params(axis='x', which='minor', length=2)

        if name == "PC_ct_pass_tra_base":
            model_name_str = "Base Scenario"
        elif name == "PC_ct_pass_tra_ETS1and2":
            model_name_str = "Emission Policy"
        elif name == "PC_ct_pass_tra_cap_add_target":
            model_name_str = "Investment Policy"
        elif name == "PC_ct_pass_tra_gen_target":
            model_name_str = "Operation Policy"
        axs[i].set_title(model_name_str + ', TSA:20', loc='left', x=0.05, y=0.9, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 1800)

    # Add a common x and y label to the entire figure
    fig.text(0.5, 0.04, 'Years', ha='center')
    fig.text(0.04, 0.5, 'Cost [billion €]', va='center', rotation='vertical')

    # Add legends
    for ax in axs:
        ax.legend(loc='lower left',bbox_to_anchor=(0.05, 0))

    # Adjust layout to make the subplots fit well together
    plt.subplots_adjust(hspace=0, wspace=0)

    # Display the plot
    plt.show(block=True)

def comparative_costs_over_time_policy_impact(model_name, scenario):
    discounted = False
    scenarios = ["15_5", "15_1", "5_5", "5_1"]
    other_scenarios = [s for s in scenarios if s != scenario]

    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]

    dfs = []
    for name in model_name:
        if discounted == True:
            r, config = read_in_special_results(name)
            df_OPEX, df_CAPEX = r.get_npc()
            temp_dfs = []
            for i, df in enumerate([df_OPEX, df_CAPEX]):
                names = ['cost_opex_total', 'cost_capex_total']
                df = indexmapping(df, special_model_name=name)
                df_selected = df.iloc[:, ::2]
                new_column_names = range(15)
                df_selected.columns = new_column_names
                df_selected.index = pd.MultiIndex.from_product([[names[i]], df_selected.index])
                if i == 0:
                    df_OPEX = df_selected
                    temp_dfs.append(df_OPEX)
                else:
                    df_CAPEX = df_selected
                    temp_dfs.append(df_CAPEX)
            df = pd.concat(temp_dfs, axis=0, join='outer')
        else:
            r, config = read_in_results(name)

            costs = ["cost_opex_total", "cost_carrier_total", "cost_capex_total"]
            df = {}
            for c in costs:
                df[c] = r.get_total(c)
                df[c] = indexmapping(df[c], special_model_name=name)

            df = pd.concat(df, keys=df.keys())
            df.loc['cost_opex_total'] = (df.loc['cost_opex_total'].values + df.loc['cost_carrier_total'].values)
            df = df.drop('cost_carrier_total')
            df.index = pd.MultiIndex.from_tuples([(first, second + "_" + name) for first, second in df.index])

        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df = df /1000
    df = df.sort_index()
    df = df[~df.index.get_level_values(1).str.startswith(tuple(other_scenarios))]

    # df = generate_df()
    df.columns = rename_columns(df.columns)

    # Set the color palette
    colors = sns.color_palette('pastel')[:2]  # Choosing two colors from the 'pastel' palette
    saved_lines = {}
    saved_lines_capex = {}

    # Set up a 2x2 grid of plots with no space between them
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # axs = axs.flatten()

    colors = ['#1f77b4', '#ff7f0e']

    # make sure the first element in the for loop is the base scenario
    if "PC_ct_pass_tra_base" in model_name:
        model_name.remove("PC_ct_pass_tra_base")
        model_name.insert(0, "PC_ct_pass_tra_base")

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    for i, name in enumerate(model_name):
        # For each subplot, filter df for the current scenario across both cost types
        model_data = {cost_type: df.loc[cost_type, scenario + "_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}


        if name == "PC_ct_pass_tra_base":
            base_scenario_capex = model_data['cost_capex_total']
            base_scenario_opex = model_data['cost_opex_total'] + base_scenario_capex


        capex = model_data['cost_capex_total']
        opex = model_data['cost_opex_total'] + capex


        # Plot CAPEX (fill between x-axis and CAPEX data using the original color)
        axs[i].fill_between(capex.index, 0, capex, label='CAPEX', color=colors[0], alpha=0.5)

        # Plot OPEX (fill between CAPEX and OPEX using the original color)
        axs[i].fill_between(capex.index, capex, opex, label='OPEX', color=colors[1], alpha=0.5)

        if name != "PC_ct_pass_tra_base":
            # Plot CAPEX for 15_5 with a dotted line
            axs[i].plot(base_scenario_capex.index, base_scenario_capex, label='Base Scenario CAPEX', linestyle=':', color="black", linewidth=2)

            # Plot OPEX for 15_5 with a dashed line
            axs[i].plot(base_scenario_opex.index, base_scenario_opex, label='Base Scenario OPEX', linestyle='--', color="black", linewidth=2)

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 2000)

        # Set x-axis ticks every two years from 2022 to 2050
        axs[i].set_xticks(np.arange(2022, 2052, 2))
        axs[i].set_xlim([2020, 2052])

        # Get the current tick labels and only display every second one
        tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(np.arange(2022, 2052, 2))]
        axs[i].set_xticklabels(tick_labels)

        # Enable minor ticks
        axs[i].minorticks_on()

        # Define the position of the minor ticks (every year)
        axs[i].set_xticks(np.arange(2021, 2051, 2), minor=True)

        # Customize the appearance of minor ticks
        axs[i].tick_params(axis='x', which='minor', length=2)

        if name == "PC_ct_pass_tra_base":
            model_name_str = "Base Scenario"
        elif name == "PC_ct_pass_tra_ETS1and2":
            model_name_str = "Emission Policy"
        elif name == "PC_ct_pass_tra_cap_add_target":
            model_name_str = "Investment Policy"
        elif name == "PC_ct_pass_tra_gen_target":
            model_name_str = "Operation Policy"
        axs[i].set_title(model_name_str + ', TSA:20', loc='left', x=0.05, y=0.9, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        # Set y-limits (adjust based on your data)
        axs[i].set_ylim(0, 1800)

    # Add a common x and y label to the entire figure
    fig.text(0.5, 0.04, 'Years', ha='center')
    fig.text(0.04, 0.5, 'Cost [billion €]', va='center', rotation='vertical')

    # Add legends
    for ax in axs:
        ax.legend(loc='lower left',bbox_to_anchor=(0.05, 0))

    # Adjust layout to make the subplots fit well together
    plt.subplots_adjust(hspace=0, wspace=0)

    # Display the plot
    plt.show(block=True)




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

model_name = ["PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_ETS1and2", "PC_ct_pass_tra_gen_target"]
sectors = ["electricity", "heat", "passenger_mileage"] #  "electricity", "heat", "passenger_mileage"



# # extreme comparison plot
# comparative_costs_over_time_extreme(model_name)

# # perfect foresight investment planning plot
# comparative_costs_over_time_15(model_name)

# # myopic investment planning plot
# comparative_costs_over_time_5(model_name)

# # policy impact plot
# scenario = "5_1"
# comparative_costs_over_time_policy_impact(model_name, scenario)
#
