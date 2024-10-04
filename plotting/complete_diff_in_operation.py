from plotting.helpers import *
import matplotlib.patches as mpatches
import matplotlib.lines as mlines



def complete_diff_in_operation_plot(model_name):
    for model in model_name:
        r, config = read_in_special_results(model)
        if "base" in model:
            desired_scenarios = ['5_5', '5_1']

            df_output = r.get_total("flow_conversion_output")
            df_output.reset_index(inplace=True)
            df_output.rename(columns={'level_0': 'scenario'}, inplace=True)
            df_output.set_index(['scenario', 'technology', 'carrier', 'node'], inplace=True)
            # Perform the aggregation by summing over the 'node' level of the index
            df_output = df_output.groupby(level=['scenario', 'technology', 'carrier']).sum()
            # indexing
            df_output = indexmapping(df_output, special_model_name=model)
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

            mask_conventional = electricity_df['category'] == 'conventional'
            conv_el_df = electricity_df[mask_conventional]
            conv_el_df = conv_el_df.drop(['category'], axis=1)
            conv_el_df = conv_el_df.reset_index('carrier', drop=True)

            # Apply the renaming function to the 'technology' part of the index
            new_index = [(item, rename_technology(tech)) for item, tech in conv_el_df.index]
            conv_el_df.index = pd.MultiIndex.from_tuples(new_index, names=conv_el_df.index.names)

            percentage_renewable_electricity = calculate_renewable_percentage(electricity_df)

        if "_pass_tra" in model:
            e_5_1, em_5_1 = r.get_all_sector_emissions("scenario_2")
            e_5_5, em_5_5 = r.get_all_sector_emissions("scenario_3")
        else:
            raise NotImplementedError("Check the scenarios and model chosen")

        tech_em_el_5_5 = em_5_5.loc["electricity"]
        tech_em_el_5_5.loc['natural_gas'] += tech_em_el_5_5.loc['lng']
        tech_em_el_5_5 = tech_em_el_5_5.drop('lng')
        tech_em_el_5_5 = tech_em_el_5_5.rename(index={'tech': 'CCS'})

        tech_em_el_5_1 = em_5_1.loc["electricity"]
        tech_em_el_5_1.loc['natural_gas'] += tech_em_el_5_1.loc['lng']
        tech_em_el_5_1 = tech_em_el_5_1.drop('lng')
        tech_em_el_5_1 = tech_em_el_5_1.rename(index={'tech': 'CCS'})

        order = ["CCS", "hard_coal", "lignite", "oil", "natural_gas", "waste"]

        # Reorder tech_em_el_5_1
        complete_order = order + [tech for tech in tech_em_el_5_1.index if tech not in order]
        tech_em_el_5_1['order'] = pd.Categorical(tech_em_el_5_1.index, categories=complete_order, ordered=True)
        tech_em_el_5_1 = tech_em_el_5_1.sort_values('order')

        # Reorder tech_em_el_5_5
        tech_em_el_5_5['order'] = pd.Categorical(tech_em_el_5_5.index, categories=complete_order, ordered=True)
        tech_em_el_5_5 = tech_em_el_5_5.sort_values('order')

        # Remove the temporary 'order' column
        tech_em_el_5_1.drop(columns='order', inplace=True)
        tech_em_el_5_5.drop(columns='order', inplace=True)

        if "base" in model:
            base_tech_em_el_5_1 = tech_em_el_5_1
            base_tech_em_el_5_5 = tech_em_el_5_5
        elif "cap_add" in model:
            cap_add_tech_em_el_5_1 = tech_em_el_5_1
            cap_add_tech_em_el_5_5 = tech_em_el_5_5
        elif "gen" in model:
            gen_tech_em_el_5_1 = tech_em_el_5_1
            gen_tech_em_el_5_5 = tech_em_el_5_5
        elif "ETS" in model:
            ETS_tech_em_el_5_1 = tech_em_el_5_1
            ETS_tech_em_el_5_5 = tech_em_el_5_5

    # plotting stuff

    def colors(index):
        if index == "hard_coal":
            color = "black"
        elif index == "nuclear":
            color = "#eed202"
        elif index == "gasoline":
            color = "#f0c702"
        elif index == "diesel":
            color = "#f4b102"
        elif index == "natural_gas":
            color = "#377eb8"
        elif index == "oil":
            color = "#66c2a5"
        elif index == "waste":
            color = "#999999"
        elif index == "lignite":
            color = "#a65628"
        elif index == "CCS":
            color = "#6a0dad"
        else:
            raise NotImplementedError("Technology-Color not implemented")
        return color

    def transform_label(label):
        if label != "CCS":
            # Capitalize the first letter of the label
            label = label.capitalize()

        # Replace underscores with spaces and capitalize the subsequent letter
        if label != "CCS":
            if '_' in label:
                parts = label.split('_')
                label = ' '.join(word.capitalize() for word in parts)

        return label

    base_colors_1 = [colors(idx) for idx in base_tech_em_el_5_1.index]
    base_colors_5 = [colors(idx) for idx in base_tech_em_el_5_5.index]
    cap_add_colors_1 = [colors(idx) for idx in cap_add_tech_em_el_5_1.index]
    cap_add_colors_5 = [colors(idx) for idx in cap_add_tech_em_el_5_5.index]
    gen_colors_1 = [colors(idx) for idx in gen_tech_em_el_5_1.index]
    gen_colors_5 = [colors(idx) for idx in gen_tech_em_el_5_5.index]
    ETS_colors_1 = [colors(idx) for idx in ETS_tech_em_el_5_1.index]
    ETS_colors_5 = [colors(idx) for idx in ETS_tech_em_el_5_5.index]


    # for legend:
    unique_colors_labels = {}
    all_colors = {
        **dict(zip(base_tech_em_el_5_1.index, base_colors_1)),
        **dict(zip(base_tech_em_el_5_5.index, base_colors_5)),
        **dict(zip(cap_add_tech_em_el_5_1.index, cap_add_colors_1)),
        **dict(zip(cap_add_tech_em_el_5_5.index, cap_add_colors_5)),
        **dict(zip(gen_tech_em_el_5_1.index, gen_colors_1)),
        **dict(zip(gen_tech_em_el_5_5.index, gen_colors_5)),
        **dict(zip(ETS_tech_em_el_5_1.index, ETS_colors_1)),
        **dict(zip(ETS_tech_em_el_5_5.index, ETS_colors_5)),
    }
    for tech, color in all_colors.items():
        label = transform_label(tech)
        unique_colors_labels[label] = color

    legend_elements = []
    # Add line styles at the top
    legend_elements.append(mlines.Line2D([], [], color='black', linestyle=':', label='Scenario D (IF:10a,  OF:2a)'))
    legend_elements.append(mlines.Line2D([], [], color='black', linestyle='-', label='Scenario C (IF:10a,  OF:10a)'))
    # Add color patches for each unique color
    for label, color in unique_colors_labels.items():
        legend_elements.append(mpatches.Patch(color=color, label=label))

    fig, axs = plt.subplots(3, 2, figsize=(9, 13))  # Creating a 2x3 grid of subplots
    width = 0.4
    # Top-left plot (Percentage renewable electricity)
    axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['5_1'],
                   label='Scenario D (IF:10a,  OF:2a)', linestyle=':', color='black', zorder=2)
    axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['5_5'],
                   label='Scenario C (IF:10a,  OF:10a)', linestyle='-', color='black', zorder=2)

    axs[0, 0].set_title('Electricity from Renewable Generation')
    axs[0, 0].set_ylabel('Renewable Electricity [%]', fontsize=12)
    axs[0, 0].grid(True, zorder=1)

    # Clear out unwanted elements in the top-right subplot
    axs[0, 1].plot([], [], ' ')  # Dummy plot to attach the legend
    axs[0, 1].axis('off')
    # Adding a dummy plot to the top-right subplot to anchor the legend
    axs[0, 1].legend(handles=legend_elements, loc='center', frameon=False, fontsize=10)

    # Middle-left plot (Base Emissions)
    base_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=base_colors_1,
                               position=0.0, ax=axs[1, 0], width=width)
    base_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=base_colors_5, ax=axs[1, 0],
                               position=1, width=width)
    axs[1, 0].set_title('Electricity Sector Emissions over Time')
    axs[1, 0].set_ylabel('Annual Emissions [Mt CO₂]')
    axs[1, 0].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Middle-right plot (Capacity Additions Emissions)
    cap_add_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=cap_add_colors_1,
                                  position=0.0, ax=axs[1, 1], width=width)
    cap_add_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=cap_add_colors_5,
                                  ax=axs[1, 1], position=1, width=width)
    axs[1, 1].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Bottom-left plot (Generation Emissions)
    gen_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=gen_colors_1, position=0.0,
                              ax=axs[2, 0], width=width)
    gen_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=gen_colors_5, ax=axs[2, 0],
                              position=1, width=width)
    axs[2, 0].set_ylabel('Annual Emissions [Mt CO₂]')
    axs[2, 0].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Bottom-right plot (ETS Emissions)
    ETS_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=ETS_colors_1, position=0.0,
                              ax=axs[2, 1], width=width)
    ETS_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=ETS_colors_5, ax=axs[2, 1],
                              position=1, width=width)
    axs[2, 1].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)
    # List of target subplots
    target_axes = [axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]

    # # Apply settings to each subplot
    for ax in target_axes:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + 0.45)
        ax.set_ylim(-100, 800)
        ax.grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Remove legends from all subplots except the top-right (axs[0, 1])
    for ax in axs.flat:
        if ax != axs[0, 1] and ax.get_legend() is not None:  # Only hide legends if they exist
            ax.get_legend().set_visible(False)

    subplot_labels = [
        "Base Scenario", "", "Base Scenario", "Investment Policy",
        "Operation Policy", "Emission Policy"
    ]
    x_pos = 0.03
    y_pos = 0.93

    for i, ax in enumerate(axs.flat):
        ax.text(x=x_pos, y=y_pos, s=subplot_labels[i], transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    i = 0
    for row in axs:
        j = 0
        for ax in row:
            # ax.grid(True, zorder=1)
            ax.grid(False)
            if i < 1 and j < 1:
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
                            color='grey')

            elif i > 0:
                ax.set_ylim(0, 850)
                # Define positive ticks
                positive_ticks = np.arange(0, 851, 100)

                # Set y-axis ticks
                ax.set_yticks(positive_ticks)
                ax.yaxis.grid(True, which='major', linestyle=':', linewidth=0.3,
                              color='grey')

                xticks_major = np.arange(0, 16, 2)
                ax.set_xticks(xticks_major)
                ax.axhline(y=0, color='black', linestyle=':', linewidth=0.7)
                ax.xaxis.grid(True, which='major', linestyle=':', linewidth=0.3,
                              color='grey')

                # Calculate new labels for these ticks
                new_labels = [2022 + 2 * int(tick) for tick in xticks_major]
                ax.set_xticklabels(new_labels, rotation=0)

            if i == 1 and j == 0:
                # Calculate sum for the bars at the first x-tick
                y_coord_5_5 = base_tech_em_el_5_5.sum(axis=0).iloc[0]
                y_coord_5_1 = base_tech_em_el_5_1.sum(axis=0).iloc[0]

                # X-coordinates based on the adjusted positions of bars
                x_coord_5_5 = -0.2  # Adjust this value if the left bar shifts
                x_coord_5_1 = 0.3  # Adjust this value if the right bar shifts

                # Annotations for the bars
                ax.text(x_coord_5_5 + 8.2, y_coord_5_5 + 80, 'Scenario C (IF:10a,  OF:10a)', ha='center', va='bottom')
                ax.plot([x_coord_5_5, x_coord_5_5 + 3], [y_coord_5_5 + 100, y_coord_5_5 + 100], color='black')
                ax.plot([x_coord_5_5, x_coord_5_5], [y_coord_5_5, y_coord_5_5 + 100], color='black')

                ax.text(x_coord_5_1 + 7.7, y_coord_5_1 - 20, 'Scenario D (IF:10a,  OF:2a)', ha='center', va='bottom')
                ax.plot([x_coord_5_1, x_coord_5_1 + 3], [y_coord_5_1, y_coord_5_1], color='black')

            j += 1
        i += 1

    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


model_name = ["PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_gen_target", "PC_ct_pass_tra_ETS1and2"]

complete_diff_in_operation_plot(model_name)

