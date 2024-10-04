from plotting.helpers import *

def sector_emissions_plot(el_emissions_5_5, heat_emissions_5_5, transport_emissions_5_5, el_emissions_5_1, heat_emissions_5_1, transport_emissions_5_1):
    # Create figure and axes for subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot for 5_5 data
    ax1.plot(el_emissions_5_5, label='Electricity Emissions')
    ax1.plot(heat_emissions_5_5, label='Heat Emissions')
    ax1.plot(transport_emissions_5_5, label='Transport Emissions')
    ax1.set_title('5_5')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Emissions [Mt CO2]')
    ax1.legend()

    # Plot for 5_1 data
    ax2.plot(el_emissions_5_1, label='Electricity Emissions')
    ax2.plot(heat_emissions_5_1, label='Heat Emissions')
    ax2.plot(transport_emissions_5_1, label='Transport Emissions')
    ax2.set_title('5_1')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Emissions [Mt CO2]')
    ax2.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

def emissions_techs_plot(sector):
    print("getting all sector emissions")
    if "_pass_tra" in model_name:
        e_5_1, em_5_1 = r.get_all_sector_emissions("scenario_2")
        e_5_5, em_5_5 = r.get_all_sector_emissions("scenario_3")
    else:
        raise NotImplementedError("Check the scenarios and model chosen")

    # an overview plot to see what the different sectors are doing
    # set to true to plot the overview plot
    if False:
        update_indices(e_5_1)
        update_indices(e_5_5)

        el_emissions_5_5 = sum_series_in_nested_dict(e_5_5['electricity'])
        heat_emissions_5_5 = sum_series_in_nested_dict(e_5_5['heat'])
        transport_emissions_5_5 = sum_series_in_nested_dict(e_5_5['passenger_mileage'])
        el_emissions_5_1 = sum_series_in_nested_dict(e_5_1['electricity'])
        heat_emissions_5_1 = sum_series_in_nested_dict(e_5_1['heat'])
        transport_emissions_5_1 = sum_series_in_nested_dict(e_5_1['passenger_mileage'])

        sector_emissions_plot(el_emissions_5_5, heat_emissions_5_5, transport_emissions_5_5, el_emissions_5_1, heat_emissions_5_1, transport_emissions_5_1)

    tech_em_el_5_5 = em_5_5.loc[sector]
    tech_em_el_5_5.loc['natural_gas'] += tech_em_el_5_5.loc['lng']
    tech_em_el_5_5 = tech_em_el_5_5.drop('lng')
    tech_em_el_5_5 = tech_em_el_5_5.rename(index={'tech': 'CCS'})

    tech_em_el_5_1 = em_5_1.loc[sector]
    tech_em_el_5_1.loc['natural_gas'] += tech_em_el_5_1.loc['lng']
    tech_em_el_5_1 = tech_em_el_5_1.drop('lng')
    tech_em_el_5_1 = tech_em_el_5_1.rename(index={'tech': 'CCS'})

    order = [ "CCS", "hard_coal", "lignite", "oil", "natural_gas", "waste"]
    if sector == "passenger_mileage":
        order = ["gasoline", "diesel", "CCS", "hard_coal", "lignite", "oil", "natural_gas", "waste"]

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

    colors_1 = [colors(idx) for idx in tech_em_el_5_1.index]
    colors_5 = [colors(idx) for idx in tech_em_el_5_5.index]

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    # Width of the bars in the bar plot
    width = 0.3

    # Locations for the groups on the x-axis
    x = range(len(tech_em_el_5_1.columns))

    # Plotting data from the first dataframe
    tech_em_el_5_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=colors_1, position=-0.5, ax=ax, width=width) # position=0.5,

    # Plotting data from the second dataframe
    tech_em_el_5_5.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=colors_5, ax=ax, width=width) # position=-0.5,

    # Setting x-axis labels
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(tech_em_el_5_1.columns)

    plt.title('Carrier Emissions over Time: ' + sector + '\n' + model_name)
    plt.ylabel('Anual Emissions [Mt CO₂]')

    # Get all handles (patches) and labels from the current axes
    handles, labels = ax.get_legend_handles_labels()

    # Transform each label using the defined function
    transformed_labels = [transform_label(label) for label in labels]

    unique_legend = {label: handle for handle, label in zip(handles, transformed_labels)}
    handles_ordered = list(unique_legend.values())[::-1]
    labels_ordered = list(unique_legend.keys())[::-1]

    # Set the legend with the unique handles and labels
    legend = ax.legend(handles=handles_ordered, labels=labels_ordered, title='Carrier')
    legend.set_title('Carrier', prop={'weight': 'bold'})

    current_ticks = ax.get_xticks()
    new_labels = [2022 + 2 * int(tick) if i % 2 == 0 else '' for i, tick in enumerate(current_ticks)]
    ax.set_xticklabels(new_labels, rotation=0)
    ax.axhline(y=0, color='black', linewidth=1)

    # Shift the x_min to the left by subtracting a small value
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min - 0.5, x_max)
    ax.set_ylim(-100, 800)
    ax.grid(True, which='both', linestyle='-', linewidth=0.2, color='gray', alpha=0.4)

    # Annotation for the leftmost bar ('5_5')
    x_coord_5_5 = -0.05  # x-coordinate for the first '5_5' bar (adjust if needed)
    y_coord_5_5 = tech_em_el_5_5.sum(axis=0).iloc[0]  # sum of the first column for y-coordinate
    ax.text(x_coord_5_5, y_coord_5_5 + 60, '5_5', ha='center', va='bottom')

    # Draw a short line pointing to the '5_5' label
    ax.plot([x_coord_5_5, x_coord_5_5], [y_coord_5_5, y_coord_5_5 + 55], color='black')

    # Annotation for the right bar ('5_1')
    x_coord_5_1 = 0.35  # x-coordinate for the first '5_1' bar (adjust if needed)
    y_coord_5_1 = tech_em_el_5_1.sum(axis=0).iloc[0]  # sum of the first column for y-coordinate
    ax.text(x_coord_5_1 + 0.3, y_coord_5_1 + 20, '5_1', ha='center', va='bottom')

    # Draw a short line pointing to the '5_1' label
    ax.plot([x_coord_5_1, x_coord_5_1], [y_coord_5_1, y_coord_5_1 + 15], color='black')

    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

model_names = ["PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_gen_target", "PC_ct_pass_tra_ETS1and2"] # "PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target", "PC_ct_pass_tra_gen_target", "PC_ct_pass_tra_ETS1and2"
sectors = ["electricity"] #  "electricity", "heat", "passenger_mileage"

for model_name in model_names:
    r, config = read_in_special_results(model_name)

    for sector in sectors:
        emissions_techs_plot(sector)
