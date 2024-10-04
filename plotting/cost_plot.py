from plotting.helpers import *
import matplotlib.patches as mpatches

def bar_comparative_costs_over_time(model_name):
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
    df = pd.concat(dfs, axis=0, join='outer')
    df = df / 1000
    df = df.sort_index()
    df = df[~df.index.get_level_values(1).str.startswith(('15_1', '5_5'))]

    # ==============================================
    # to develop the figure without the data carrier:
    # df = generate_df()
    # ==============================================

    df.columns = rename_columns(df.columns)

    # Set up a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    # Define colors
    colors = ['#1f77b4', '#ff7f0e']  # Two colors for CAPEX and OPEX
    darker_colors = ['#104e8b', '#b25a00']  # Darker shades for diamonds

    capex_patch = mpatches.Patch(color=colors[0], label='CAPEX', alpha=0.8)
    opex_patch = mpatches.Patch(color=colors[1], label='OPEX', alpha=0.8)

    # Loop over each model and generate the bar charts
    for i, name in enumerate(model_name):

        # For each subplot, filter df for the current scenario across both cost types
        model_data_15 = {cost_type: df.loc[cost_type, "15_5_" + name] for cost_type in
                         ['cost_opex_total', 'cost_capex_total']}

        model_data_5 = {cost_type: df.loc[cost_type, "5_1_" + name] for cost_type in
                        ['cost_opex_total', 'cost_capex_total']}

        capex_5_1 = model_data_5['cost_capex_total']
        opex_5_1 = model_data_5['cost_opex_total']

        capex_15_5 = model_data_15['cost_capex_total']
        opex_15_5 = model_data_15['cost_opex_total']

        x_ticks = capex_5_1.index  # Assuming the x-axis remains years from 2022 to 2050

        # Width for the bars
        width = 0.6

        if name == "PC_ct_pass_tra_base":
            diamond_5_1_opex = opex_5_1
            diamond_5_1_capex = capex_5_1

            diamond_15_5_opex =opex_15_5
            diamond_15_5_capex = capex_15_5

        # Plot stacked bars for 5_1 scenario (left of each tick)
        axs[i].bar(x_ticks - width/2, capex_5_1, width=width, label='CAPEX 5_1', color=colors[0], alpha=0.5)
        axs[i].bar(x_ticks - width/2, opex_5_1, width=width, bottom=capex_5_1, label='OPEX 5_1', color=colors[1], alpha=0.5)

        # Plot stacked bars for 15_5 scenario (right of each tick)
        axs[i].bar(x_ticks + width/2, capex_15_5, width=width, label='CAPEX 15_5', color=colors[0], alpha=0.8)
        axs[i].bar(x_ticks + width/2, opex_15_5, width=width, bottom=capex_15_5, label='OPEX 15_5', color=colors[1], alpha=0.8)

        # Add diamonds for base case data on non-base subplots
        if name != "PC_ct_pass_tra_base":
            axs[i].scatter(x_ticks - width/2, diamond_5_1_opex + diamond_5_1_capex, marker='x', color="black", s=40, label='Base OPEX 5_1', alpha=0.5)
            axs[i].scatter(x_ticks + width/2, diamond_15_5_opex + diamond_15_5_capex, marker='x', color="black", s=40, label='Base OPEX 15_5', alpha=0.8)
            # axs[i].scatter(x_ticks - width / 2, diamond_5_1_capex, marker='x', color=darker_colors[0], s=20, label='Base OPEX 5_1', alpha=0.5)
            # axs[i].scatter(x_ticks + width / 2, diamond_15_5_capex, marker='x', color=darker_colors[0], s=20, label='Base OPEX 15_5', alpha=0.8)

            # Create dummy diamonds for the legend
            diamond_legend_1 = axs[i].scatter([], [], marker='x', color="black", s=30,
                                               label='Base Scenario Total Cost')
            # diamond_legend_2 = axs[i].scatter([], [], marker='x', color=darker_colors[1], s=15,
            #                                    label='Base Scenario OPEX')

            # Add a legend to the bottom right of the last subplot (axs[-1])
            axs[i].legend([diamond_legend_1], [], loc='lower right', fontsize=10)


        # Set the x-axis ticks and labels
        axs[i].set_xticks(np.arange(2022, 2052, 2))
        axs[i].set_xlim([2020, 2052])

        # Get the current tick labels and only display every second one
        tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(np.arange(2022, 2052, 2))]
        axs[i].set_xticklabels(tick_labels)

        # Set titles for each subplot
        if name == "PC_ct_pass_tra_base":
            model_name_str = "Base Scenario"
        elif name == "PC_ct_pass_tra_ETS1and2":
            model_name_str = "Emission Policy"
        elif name == "PC_ct_pass_tra_cap_add_target":
            model_name_str = "Investment Policy"
        elif name == "PC_ct_pass_tra_gen_target":
            model_name_str = "Operation Policy"
        axs[i].set_title(model_name_str, loc='left', x=0.05, y=0.9, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        axs[i].set_ylim(0, 1750)

        # Label bars in the first subplot
        if i == 0:
            capex_label = capex_5_1.iloc[0] + opex_5_1.iloc[0]
            axs[0].annotate('Most Myopic', xy=(x_ticks[0] - width / 2, capex_label),
                            xytext=(2020.5, capex_label + 300),
                            arrowprops=dict(facecolor='black', arrowstyle='-'), fontsize=10)

            capex_label = capex_15_5.iloc[0] + opex_15_5.iloc[0]
            axs[0].annotate('Least Myopic', xy=(x_ticks[0] + width / 2, capex_label),
                            xytext=(x_ticks[0] + 2 * width, capex_label + 150),
                            arrowprops=dict(facecolor='black', arrowstyle='-'), fontsize=10)

    # Loop over subplots
    for i, ax in enumerate(axs):
        if i == 0:
            # First subplot gets only CAPEX and OPEX patches
            ax.legend(handles=[capex_patch, opex_patch], loc='lower right', fontsize=10)
        else:
            # Other subplots get both patches and diamond markers
            ax.legend(handles=[capex_patch, opex_patch, diamond_legend_1], loc='lower right', # , diamond_legend_2
                      fontsize=10)

    # Add a common x and y label to the entire figure
    fig.text(0.5, 0.04, 'Years', ha='center')
    fig.text(0.04, 0.5, 'Cost [billion €]', va='center', rotation='vertical')

    # Adjust layout to make the subplots fit well together
    plt.subplots_adjust(hspace=0, wspace=0)

    # Display the plot
    plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# barchart of the cost progression
model_name = ["PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_gen_target", "PC_ct_pass_tra_ETS1and2"]
bar_comparative_costs_over_time(model_name)
