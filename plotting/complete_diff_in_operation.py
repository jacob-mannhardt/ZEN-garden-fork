from plotting.helpers import *



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


    fig, axs = plt.subplots(3, 2, figsize=(9, 13))  # Creating a 2x3 grid of subplots

    # Top-left plot (Percentage renewable electricity)
    axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['5_1'],
                   label='IF:10yr,  OF:2yr', linestyle=':', color='black', zorder=2) # 'IF:30a,  OF:2a'
    axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['5_5'],
                   label='IF:10yr,  OF:10yr', linestyle='-', color='black', zorder=2)

    axs[0, 0].set_title('Electricity from Renewable Generation')
    axs[0, 0].set_ylabel('Renewable Electricity [%]', fontsize=12)
    axs[0, 0].grid(True, zorder=1)

    axs[0, 1].axis('off')

    # Middle-left plot (Base Emissions)
    base_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=base_colors_1,
                               position=-0.15, ax=axs[1, 0], width=0.3)
    base_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=base_colors_5, ax=axs[1, 0],
                               width=0.3)
    axs[1, 0].set_title('Electricity Sector Emissions over Time')
    axs[1, 0].set_ylabel('Annual Emissions [Mt CO₂]')
    axs[1, 0].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Middle-right plot (Capacity Additions Emissions)
    cap_add_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=cap_add_colors_1,
                                  position=-0.5, ax=axs[1, 1], width=0.3)
    cap_add_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=cap_add_colors_5,
                                  ax=axs[1, 1], width=0.3)
    axs[1, 1].set_title('Capacity Additions Emissions')
    axs[1, 1].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Bottom-left plot (Generation Emissions)
    gen_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=gen_colors_1, position=-0.5,
                              ax=axs[2, 0], width=0.3)
    gen_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=gen_colors_5, ax=axs[2, 0],
                              width=0.3)
    axs[2, 0].set_title('Generation Emissions')
    axs[2, 0].set_ylabel('Annual Emissions [Mt CO₂]')
    axs[2, 0].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Bottom-right plot (ETS Emissions)
    ETS_tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=ETS_colors_1, position=-0.5,
                              ax=axs[2, 1], width=0.3)
    ETS_tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=ETS_colors_5, ax=axs[2, 1],
                              width=0.3)
    axs[2, 1].set_title('ETS Emissions')
    axs[2, 1].grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    axs[2, 1].set_title('Title 2, 1')

    # List of target subplots
    target_axes = [axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]

    # Apply settings to each subplot
    for ax in target_axes:
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min - 0.5, x_max)
        ax.set_ylim(-100, 800)
        ax.grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # # Adding legends and adjusting layout
    for ax in axs.flat:
        ax.legend().set_visible(False)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, 0.42), ncol=3)


    # Adjust the layout
    # # Adjust the layout to make space for the legend outside the subplots
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Adjust spacing between subplots and increase the bottom margin
    fig.subplots_adjust(hspace=0.4, bottom=0.15)  # Increased bottom margin

    # Retrieve and adjust the positions of the last row subplots to create extra space
    # Manually adjusting the 'bottom' property of the third row's axes
    bottom_shift = 0.02  # Amount to move the last row down, adjust as needed


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
                            color='grey')  # Adjust linestyle, linewidth, and color as needed

            else:
                ax.set_ylim(0, 750)
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
            j += 1
        i += 1

    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


model_name = ["PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_gen_target", "PC_ct_pass_tra_ETS1and2"]

complete_diff_in_operation_plot(model_name)

