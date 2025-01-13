"""
This script is used to plot the results of the uniform vs.variable CoC comparison and
for the technology vs. country convergence comparison.
It has the following structure:
1. Define paths, colors, labels etc. - Important select here which plots to plot (uniform vs. variable CoC or technology vs. country convergence)
2. Collection of all plotting functions
3. Generate plots for uniform vs. variable CoC (if results_path == results_path_annuity)
4. Generate plots for technology vs. country convergence (if results_path == results_path_yearly)

"""

from zen_garden.postprocess.results.results import Results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import eth_colors


#paths
results_path_yearly = 'outputs/PI_yearly_new/'
results_path_annuity = 'outputs/PI_annuity/'

#scenario names of policy scenarios
scenario_1_yearly = "scenario_country_convergence"
scenario_2_yearly = "scenario_technology_convergence"

#ToDO:choose here which results to plot!!!!!!!
#results_path_yearly corresponds to country vs. technology convergence
#results_path_annuity corresponds to variable CoC vs. uniform CoC
results_path = results_path_yearly

#read results
r_yearly = Results(path=results_path_yearly)
r_annuity = Results(path=results_path_annuity)

#Define technology sets
RE_technologies = ["photovoltaics","wind_offshore", "wind_onshore"]
fossil_technologies = ["hard_coal_plant", "hard_coal_plant_CCS", "lignite_coal_plant","natural_gas_turbine","natural_gas_turbine_CCS","nuclear","oil_plant","waste_plant"]

#Define regions
NWE = ["AT", "BE", "CH", "DE", "DK","EE","FI", "FR", "IE", "LU", "NL", "NO","SE", "UK", "LT", "LV"]
SE = ["ES", "IT","PT","HR", "EL"]
EE = ["BG", "CZ", "HU", "PL", "RO", "SI", "SK"]

#Define colors and labels
colors_technologies = {"biomass_plant": ("Biomass", "olivedrab"),"biomass_plant_CCS": ("Biomass CCS", "yellowgreen"),"hard_coal_plant": ("Hard Coal", "black"),"hard_coal_plant_CCS": ("Hard Coal CCS", "dimgray"),
                       "lignite_coal_plant": ("Lignite", "saddlebrown"),"natural_gas_turbine": ("Natural Gas", "darkorange"),"natural_gas_turbine_CCS": ("Natural Gas CCS", "orange"),"nuclear": ("Nuclear", "purple"),
                          "oil_plant": ("Oil", "rosybrown"),"photovoltaics": ("Photovoltaics", "gold"),"reservoir_hydro": ("Reservoir Hydro", "turquoise"),"run-of-river_hydro": ("Run-of-River Hydro", "lightseagreen"),
                       "waste_plant": ("Waste", "darkkhaki"),"wind_offshore": ("Wind Offshore", "royalblue"),"wind_onshore": ("Wind Onshore", "skyblue")}
labels_technologies = {key: value[0] for key, value in colors_technologies.items()}
labels_technologies["Others"] = "Others"
labels_technologies["Coal"] = "Coal"
labels_technologies["Hydro"] = "Hydro"
labels_technologies["battery"] = "Battery"
labels_technologies["hydrogen_storage"] = "Hydrogen Storage"
labels_technologies["pumped_hydro"] = "Pumped Hydro"
colors = eth_colors.ETHColors()
colors.setColorsTechs()
colors_dict = colors.colors["techs"]
colors_country = {
    'AT': (0.0, 0.47058823529411764, 0.5803921568627451),
    'BE': (0.3411764705882353, 0.3411764705882353, 0.3411764705882353),
    'BG': (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
    'CH': (0.4, 0.6862745098039216, 0.7529411764705882),
    'CZ': (0.6627450980392157, 0.6627450980392157, 0.6627450980392157),
    'DE': (0.469568, 0.392932, 0.167627, 1.0),
    'DK': (0.68609, 0.300996, 0.587036, 1.0),
    'EE': (0.774878, 0.634273, 0.472535, 1.0),
    'GR': (0.3843137254901961, 0.45098039215686275, 0.07450980392156863),
    'ES': (0.7176470588235294, 0.20784313725490197, 0.17647058823529413),
    'FI': (0.42485, 0.474498, 0.674725, 1.0),
    'FR': (0.551067, 0.600867, 0.800939, 1.0),
    'HR': (0.7725490196078432, 0.36470588235294116, 0.3411764705882353),
    'HU': (0.30196078431372547, 0.49019607843137253, 0.7490196078431373),
    'IE': (0.47843137254901963, 0.615686274509804, 0.8117647058823529),
    'IT': (0.6313725490196078, 0.6705882352941176, 0.44313725490196076),
    'LT': (0.5058823529411764, 0.5607843137254902, 0.25882352941176473),
    'LU': (0.8313725490196079, 0.5254901960784314, 0.5058823529411764),
    'LV': (0.253046, 0.301978, 0.50237, 1.0),
    'NL': (0.707907, 0.504161, 0.290477, 1.0),
    'NO': (0.81380667, 0.73582667, 0.35353583, 1.0),
    'PL': (0.819205, 0.496265, 0.735705, 1.0),
    'PT': (0.774878, 0.634273, 0.472535, 1.0),
    'RO': (0.400, 0.6862745098039216, 0.7529411764705882),
    'SE': (0.47843137254901963, 0.615686274509804, 0.8117647058823529),
    'SI': (0.551067, 0.600867, 0.800939, 1.0),
    'SK': (0.42485, 0.474498, 0.674725, 1.0),
    'GB': (0.7725490196078432, 0.36470588235294116, 0.3411764705882353),
}


######### plot functions used for both uniform vs. variable and technology vs. country convergence ###############

def gini_plot_multiple(data_tuples, labels,save_path = None):
    """
    Plot B.11 in the Appendix of the report.
    Plots cumulative distribution lines for multiple datasets.

    Parameters:
    data_tuples (list of tuples): Each tuple contains two dictionaries:
                                  (x_axis_dict, y_axis_dict).
                                  - x_axis_dict: Dictionary with x-axis values (widths).
                                  - y_axis_dict: Dictionary with y-axis values (heights).
    save_path (str): Path to save the plot.
    labels (list of str): List of labels for each line corresponding to the datasets.
    """
    if len(data_tuples) != len(labels):
        raise ValueError("Number of datasets must match the number of labels.")

    plt.figure(figsize=(12, 8))

    # Loop through each dataset and plot cumulative distribution
    for (x_axis_dict, y_axis_dict), label in zip(data_tuples, labels):
        # Validate input
        if not isinstance(x_axis_dict, dict) or not isinstance(y_axis_dict, dict):
            raise ValueError("Each dataset must consist of two dictionaries: x_axis_dict and y_axis_dict.")

        # Sort by x/y ratio
        dic_to_sort = {key: y_axis_dict[key] / x_axis_dict[key] for key in x_axis_dict.keys()}
        sorted_keys = sorted(dic_to_sort, key=dic_to_sort.get, reverse=False)
        sorted_x_values = [x_axis_dict[key] for key in sorted_keys]
        sorted_y_values = [y_axis_dict[key] for key in sorted_keys]

        # Compute cumulative percentages
        x_cumulative = np.cumsum(sorted_x_values) #/ sum(sorted_x_values)
        y_cumulative = np.cumsum(sorted_y_values) #/ sum(sorted_y_values)

        # Add line to the plot
        plt.plot(
            x_cumulative,
            y_cumulative,
            label=label,
            linestyle='-',
            marker=None
        )

    # Add diagonal line for ideal distribution
    plt.plot([0, 1], [0, 1], label="Ideal Fair Distribution", linestyle='--', color='red')

    # Set x and y axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    #change size of x ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Axis labels and title
    plt.xlabel("Cumulative share of total demand", fontsize=14)
    plt.ylabel("Cumulative share of total production", fontsize=14)
    plt.title("Lorenz curve", fontsize=14)

    # Add legend (with size adjustment) and grid
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f"../../../Plots_presentation/{save_path}", dpi=300)

def gini_plot(x_axis_dict, y_axis_dict,save_path = None, additional_line = None):
    """
    Figure 4.8 and 4.9 in the report.
    Plots a cumulative distribution with bars where the width corresponds to the x-values
    and the height corresponds to the cumulative y-values. The observed distribution line
    starts at the origin (0, 0) and touches the upper-right corner of each bar.

    Parameters:
    x_axis_dict (dict): Dictionary with x-axis values (widths of bars); demand.
    y_axis_dict (dict): Dictionary with y-axis values (heights of bars); production.
    save_path (str): Path to save the plot.
    additional_line (tuple): Tuple with three elements, which allows to manually plot the lorenz curve of other scenarios:
                                - x_values (list): List of x-values
                                - y_values (list): List of y-values
                                - label (str): Label for the additional line

    """

    # Sort by x-axis values in descending order
    dic_to_sort = {key:y_axis_dict[key]/x_axis_dict[key] for key in x_axis_dict.keys()}
    sorted_x = dict(sorted(dic_to_sort.items(), key=lambda item: item[1],reverse=False))
    #sorted_x = sorted(x_axis_dict.items()/y_axis_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_keys = [key for key in sorted_x.keys()]
    sorted_x_values = [x_axis_dict[key] for key in sorted_keys]
    sorted_y_values = [y_axis_dict[key] for key in sorted_keys]

    # Compute cumulative percentages
    x_cumulative = np.cumsum(sorted_x_values) #/ sum(sorted_x_values)
    y_cumulative = np.cumsum(sorted_y_values) #/ sum(sorted_y_values)

    # Add (0, 0) to the cumulative distributions
    x_cumulative = np.insert(x_cumulative, 0, 0)  # Insert 0 at the beginning
    y_cumulative = np.insert(y_cumulative, 0, 0)  # Insert 0 at the beginning

    # Ensure the plot fills the entire x-axis range
    x_positions = np.insert(x_cumulative[:-1], 0, 0)  # Start at 0 for the first bar

    # Plot setup
    plt.figure(figsize=(12, 8))

    # Plot bars
    for i, (key, x_start, x_end, y_val) in enumerate(
            zip(sorted_keys, x_positions[1:], x_cumulative[1:], y_cumulative[1:])):
        plt.bar(
            x=x_start,  # Start of the bar
            height=y_val,
            width=(x_end - x_start),  # Width of the bar
            align='edge',
            alpha=0.6,
            color = colors_country[key]
            #label=f"{key}" if i == 0 else None  # Label the first bar for legend
        )
        # Add labels above the bar
        plt.text(
            x=x_end - (x_end - x_start) / 2,  # Center of the bar
            y=y_val + 0.02,
            s=key,
            ha='center',
            va='bottom',
            fontsize=14,
            color='blue'
        )

    # Plot the line for observed cumulative distribution
    plt.plot(
        x_cumulative,
        y_cumulative,
        color='black',
        linestyle='-',
        marker=None  # Only show the line
    )

    if additional_line != None:
        plt.plot(
            additional_line[0],
            additional_line[1],
            label=additional_line[2],
            color='blue',
            linestyle='--',
            marker=None  # Only show the line
        )

    # Add diagonal line for ideal distribution
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')

    # Set x and y axis limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    #change size of x ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # Axis labels and title
    plt.xlabel("Cumulative share of total demand", fontsize=14)
    plt.ylabel("Cumulative sahre of total production", fontsize=14)
    plt.title("Lorenz curve", fontsize=14)

    # Add legend and grid
    if additional_line != None:
        plt.legend(fontsize=12)
    #plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f"../../../Plots_presentation/{save_path}", dpi=300)




def storage_plot(data, years, save_path = None):
    """
    Plot 4.4 and B.3 in the report.
    Creates a stacked bar plot for a dictionary of scenarios and technologies.

    Parameters:
    data: dict, structured as {'scenario_name': {'technology': [values_per_year], ...}, ...}
    years: list of years for the x-axis.
    save_path: str, path to save the plot.
    """
    technologies = list(next(iter(data.values())).keys())  # Extract technology names
    patterns = ['', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # Patterns for scenarios

    n_scenarios = len(data)
    n_years = len(years)
    bar_width = 1.0 / (n_scenarios + 1)  # Narrower width for multiple scenarios
    x_positions = np.arange(n_years)  # Position of years on x-axis

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (scenario, tech_data) in enumerate(data.items()):
        bottom_stack = np.zeros(n_years)  # Initialize the bottom stack for each scenario
        offset = (i - n_scenarios / 2) * bar_width  # Adjust bar positions

        for j, tech in enumerate(technologies):
            values = tech_data[tech]
            x_scenario = x_positions + offset
            ax.bar(
                x_scenario,
                values,
                bottom=bottom_stack,
                width=bar_width,
                label=f"{scenario} - {labels_technologies[tech]}" if i == 0 else "",
                color=colors_dict[tech],
                edgecolor='black',
                hatch=patterns[i % len(patterns)]
            )
            bottom_stack += np.array(values)  # Update the bottom stack

    # Add legend
    tech_legend = [
        mpatches.Patch(color=colors_dict[tech], label=labels_technologies[tech])
        for j, tech in enumerate(technologies)
    ]
    scenario_legend = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch=patterns[i % len(patterns)], label=scenario_labels[i])
        for i, scenario in enumerate(data.keys())
    ]
    plt.legend(handles=tech_legend + scenario_legend, loc='upper left')

    # Formatting
    ax.set_xticks(x_positions)
    plt.yticks(fontsize=14)
    ax.set_xticklabels(years, rotation=45, ha='right',fontsize = 14)
    ax.set_xlabel("Year", fontsize = 14)
    ax.set_ylabel("Capacity in GWh", fontsize = 14)
    ax.set_title("Stacked Bar Plot of Scenarios and Technologies")
    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def plot_stacked_bars_with_patterns_aggregated(
        scenario_0, scenario_1, years, aggregation_groups,
        scenario_labels=("Scenario 0", "Scenario 1"), y_label="Capacity",
        title="Plot", save_path=None):
    """
    Figure B.8 in the report.
    Create stacked bar plots for two scenarios with optional patterns for distinction.

    Aggregates technologies based on the aggregation_groups dictionary and assigns colors/labels to groups.

    Parameters:
    - scenario_0 (dict): First scenario, keys are technologies, values are capacities.
    - scenario_1 (dict): Second scenario, keys are technologies, values are capacities.
    - years (list): List of year labels for the x-axis.
    - aggregation_groups (dict): Dictionary defining groups of technologies to aggregate.
    - scenario_labels (tuple): Tuple of two strings for scenario labels in the legend.
    """

    if len(next(iter(scenario_0.values()))) != len(years):
        raise ValueError("Mismatch between number of timesteps in scenarios and provided years.")

    # Aggregate technologies
    def aggregate_scenario(scenario, aggregation_groups):
        aggregated = {key: [0] * len(years) for key in aggregation_groups.keys()}
        ungrouped = {key: value for key, value in scenario.items()}

        for group, techs in aggregation_groups.items():
            for tech in techs:
                if tech in ungrouped:
                    for i in range(len(years)):
                        aggregated[group][i] += ungrouped[tech][i]
                    ungrouped.pop(tech)
        aggregated.update(ungrouped)  # Add remaining technologies
        return aggregated

    scenario_0 = aggregate_scenario(scenario_0, aggregation_groups)
    scenario_1 = aggregate_scenario(scenario_1, aggregation_groups)

    # Configuration for bar plots
    n_timesteps = len(years)
    bar_width = 0.35
    x_positions = np.arange(n_timesteps) * (bar_width * 3)  # Space between groups of bars

    # Colors and patterns
    colors = plt.cm.tab10.colors
    patterns = [None, '/']  # No pattern for the first scenario, striped for the second
    keys = list(scenario_0.keys())
    n_keys = len(keys)

    # Assign colors to groups and individual technologies
    colors_technologies = {key: (key, colors[i % len(colors)]) for i, key in enumerate(keys)}

    fig, ax = plt.subplots(figsize=(14, 8))

    for t, year in enumerate(years):
        x_base = x_positions[t]  # Position for the current group of bars

        # Initialize the bottom positions for stacking
        bottom_0, bottom_1 = 0, 0

        for idx, key in enumerate(keys):
            value_0 = scenario_0[key][t]
            value_1 = scenario_1[key][t]

            # Bar for scenario_0
            ax.bar(
                x_base,
                value_0,
                bar_width,
                bottom=bottom_0,
                color=colors_dict[key],
                edgecolor="black",
                label=labels_technologies[key] if t == 0 else None  # Add legend only for the first timestep
            )
            bottom_0 += value_0

            # Bar for scenario_1
            ax.bar(
                x_base + bar_width,
                value_1,
                bar_width,
                bottom=bottom_1,
                color=colors_dict[key],
                edgecolor="black",
                hatch=patterns[1],  # Striped pattern for scenario_1
            )
            bottom_1 += value_1

    # Add year labels under the grouped bars
    ax.set_xticks(x_positions + bar_width / 2)
    ax.set_xticklabels(years,fontsize=12)
    ax.set_xlabel("Year",fontsize=14)
    ax.set_ylabel(y_label,fontsize=14)
    #set font size of y ticks
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(title)
    ax.set_ylim(0, 3000)  # Set y-limit to ensure bars touch the upper frame

    # Custom legend
    scenario_0_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch=patterns[0], label=scenario_labels[0]
    )
    scenario_1_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch=patterns[1], label=scenario_labels[1]
    )
    key_patches = [mpatches.Patch(color=colors_dict[key], label=labels_technologies[key]) for key in
                   keys]
    ax.legend(
        handles=[scenario_0_patch, scenario_1_patch] + key_patches,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize = 12
    )
    #title="Legend",
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def europe_plot(set_nodes, values,title = "Uniform vs. variable CoC", y_axis = "Capacity difference",center=0,color = "Wistia",save_path = None,vmin = None,vmax=None):
    """
    Plot 3.1, 4.3, 4.10, 4.11, B.1, B.4, B.5, B.6, B.7, B.9, B.10 in the report.
    Creates a heatmap of the values for the European countries.

    Parameters:
    set_nodes (list): List of countries to plot.
    values (list): List of values to plot.
    title (str): Title of the plot.
    y_axis (str): Label for the colorbar.
    center (float): Center value for the colormap.
    color (str): Colormap to use.
    save_path (str): Path to save the plot.
    vmin (float): Minimum value for the colormap.
    vmax (float): Maximum value for the colormap.
    """
    #set max and min value of heatmap
    if vmin==None:
        vmin = min(values)
    if vmax==None:
        vmax=max(values)
    # Create a DataFrame with country codes and values
    data = pd.DataFrame({'ISO_A2_EH': set_nodes, 'value': values})

    # Load Europe shapefile
    world = gpd.read_file("countries.zip") # path to shapefile

    # Filter for Europe
    europe = world[(world['CONTINENT'] == 'Europe') & (world['ISO_A2_EH'].isin(set_nodes))]

    # STEP 1 #
    # Make polygon from bbox coordinates https://stackoverflow.com/a/68741143/18253502
    def make_bbox(long0, lat0, long1, lat1):
        return Polygon([[long0, lat0],
                        [long1, lat0],
                        [long1, lat1],
                        [long0, lat1]])


    # Coords covering Europe made with http://bboxfinder.com
    bbox = make_bbox(-10.811309,35.344767,32.255098,71.535549)

    # Convert to gdf
    bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox])

    # Use bbox as clipping border for Europe
    europe = europe.overlay(bbox_gdf, how="intersection")

    # Merge the geometries with the data
    europe_data = europe.merge(data, left_on='ISO_A2_EH', right_on='ISO_A2_EH', how='left')

    # Plot the data
    if center!=None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        europe_data.plot(
            column='value',  # Use the 'value' column for coloring
            cmap=color,  # Choose a colormap 'RdYlGn'
            legend=True,  # Add a legend
            edgecolor='black',  # Add borders to countries
            linewidth=0.5,
            norm=TwoSlopeNorm(center,vmin=vmin,vmax=vmax),
            ax=ax
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        europe_data.plot(
            column='value',  # Use the 'value' column for coloring
            cmap=color,  # Choose a colormap 'RdYlGn'
            legend=True,  # Add a legend
            edgecolor='black',  # Add borders to countries
            linewidth=0.5,
            ax=ax,
            vmax=vmax,
            vmin=vmin
        )

    ax.set_title(title, fontsize=16)
    ax.set_axis_off()  # Turn off axis

    #give the colorbar a title
    cbar = ax.get_figure().get_axes()[1]
    cbar.set_title(y_axis, fontsize=14)
    #cbar.axhline(center, c='b')
    cbar.set_yscale('linear')
    cbar.tick_params(labelsize=14)

    # Show the plot
    plt.tight_layout()
    if save_path!=None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()

def plot_sorted_wacc(avg_wacc_tech, colors_technologies,labels, x_tick_size=10, y_label_size=12,save_path=None):
    """
    Figure B.2 in the report.
    Creates a bar plot of WACC values sorted from highest to lowest.

    Parameters:
    avg_wacc_tech (dict): Dictionary containing WACC values for each technology.
    colors_technologies (dict): Dictionary with technology names as keys, and tuples as values.
                                The first element of the tuple is the label for the x-axis,
                                and the second element is the bar color.
    x_tick_size (int): Font size for x-axis tick labels.
    y_label_size (int): Font size for y-axis label.
    save_path (str): Path to save the plot.

    """
    # Sort the dictionary by WACC values in descending order
    sorted_wacc = sorted(avg_wacc_tech.items(), key=lambda x: x[1], reverse=True)

    # Extract sorted technology names, WACC values, labels, and colors
    technologies = [tech for tech, _ in sorted_wacc]
    wacc_values = [value for _, value in sorted_wacc]
    labels = [labels[tech] for tech in technologies]
    colors = [colors_technologies[tech] for tech in technologies]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, wacc_values, color=colors)

    #add grey horizontal line at 0.584
    plt.axhline(y=0.0584, color='black', linestyle='--', alpha=0.7,label="Uniform CoC")

    # Add labels and title
    plt.ylabel('Average CoC', fontsize=y_label_size)
    plt.xticks(rotation=45, fontsize=x_tick_size)
    plt.yticks(fontsize=x_tick_size)
    plt.legend(fontsize = 10)
    # Show the plot
    plt.tight_layout()
    if save_path!=None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()

def plot_bar_plots_region(data_dict,save_path=None,y_label="LCOE in €/kWh",labelsize = 12,external_sorting = None):
    """
    Plot 4.6 in the report.
    Plots LCOE per country ordered into the three regions for two scenarios: uniform CoC and variable.
    Regions are separated by vertical dotted lines.

    Parameters:
    data_dict (dict): A dictionary where each key is a country code, and values are dictionaries with
                      'scenario_country_convergence', 'scenario_variable_CoC', and 'scenario_technology_convergence'.
    save_path (str): Path to save the plot.
    y_label (str): Label for the y-axis.
    labelsize (int): Size of the x and y-axis labels.
    external_sorting (list): List of sorted regions to be used instead of the default sorting.
    """

    # Define regions
    NWE = ["AT", "BE", "CH", "DE", "DK", "EE", "FI", "FR", "IE", "LU", "NL", "NO", "SE", "GB", "LT", "LV"]
    SE = ["ES", "IT", "PT", "HR", "GR"]
    EE = ["BG", "CZ", "HU", "PL", "RO", "SI", "SK"]

    # Group countries by region and sort by scenario_country_convergence within each region
    def sort_region(region):
        return sorted(region, key=lambda country: data_dict["scenario_0"][country], reverse=True)

    NWE_sorted = sort_region(NWE)
    SE_sorted = sort_region(SE)
    EE_sorted = sort_region(EE)

    # Combine sorted regions
    sorted_regions = [NWE_sorted, SE_sorted, EE_sorted]
    if external_sorting!=None:
        sorted_regions = external_sorting
    region_labels = ["North-West Europe", "South Europe", "East Europe"]

    # Extract normalized values and x positions
    normalized_country = []
    normalized_technology = []
    x_positions = []
    bar_width = 0.4
    space_between_bars = 0.2
    space_between_regions = 1.5
    current_x = 0

    for region in sorted_regions:
        for country in region:
            normalized_country.append(data_dict["scenario_0"][country])
            normalized_technology.append(data_dict["scenario_1"][country])
            x_positions.append(current_x)
            current_x += 2 * bar_width + space_between_bars
        current_x += space_between_regions  # Add extra space between regions

    # Plot setup
    plt.figure(figsize=(15, 8))

    # Plot bars
    plt.bar(x_positions, normalized_country, width=bar_width, label='Uniform CoC', alpha=0.7)
    plt.bar([x + bar_width for x in x_positions], normalized_technology, width=bar_width,
            label='Variable CoC', alpha=0.7)

    # Add vertical lines and headers
    region_midpoints = []
    region_start = 0
    counter = 0
    for region, label in zip(sorted_regions, region_labels):
        region_end = region_start + len(region) * (2 * bar_width + space_between_bars)
        region_midpoints.append((region_start + region_end - space_between_bars) / 2)
        if counter<2:
            plt.axvline(x=region_end + space_between_bars/2 + 0.2, color='grey', linestyle='--', alpha=0.7)
        counter += 1
        region_start = region_end + space_between_regions

    # Add labels and ticks
    plt.xticks([x + bar_width / 2 for x in x_positions], sum(sorted_regions, []), rotation=90)
    plt.tick_params(axis='x', labelsize=labelsize)
    plt.tick_params(axis='y', labelsize=labelsize)
    plt.ylabel(y_label,fontsize=16)
    plt.ylim(0,18)
    plt.legend(fontsize=labelsize)

    # Add grid and layout adjustments
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show plot
    if save_path!=None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()


def plot_normalized_scenarios(data_dict):
    """
        Figure 4.7 in the report.
        Plots normalized values (normalized in respect to variable CoC scenario) for 'scenario_technology_convergence' and 'scenario_country_convergence'
        for each country, sorted by region and by 'scenario_country_convergence' within each region.
        Regions are separated by vertical dotted lines with headers above them.

        Parameters:
        data_dict (dict): A dictionary where keys are countries, and values are dictionaries with
                          'scenario_country_convergence', 'scenario_variable_CoC', and 'scenario_technology_convergence' as keys.
        """

    # Define regions
    NWE = ["AT", "BE", "CH", "DE", "DK", "EE", "FI", "FR", "IE", "LU", "NL", "NO", "SE", "GB", "LT", "LV"]
    SE = ["ES", "IT", "PT", "HR", "GR"]
    EE = ["BG", "CZ", "HU", "PL", "RO", "SI", "SK"]

    # Group countries by region and sort by scenario_country_convergence within each region
    def sort_region(region):
        return sorted(region, key=lambda country: data_dict[country]['scenario_country_convergence']/np.round(data_dict[country]['scenario_variable_CoC'], decimals=4), reverse=True)

    NWE_sorted = sort_region(NWE)
    SE_sorted = sort_region(SE)
    EE_sorted = sort_region(EE)

    # Combine sorted regions
    sorted_regions = [NWE_sorted, SE_sorted, EE_sorted]
    region_labels = ["Northern & Western Europe", "Southern Europe", "Eastern Europe"]

    # Extract normalized values and x positions
    normalized_country = []
    normalized_technology = []
    x_positions = []
    bar_width = 0.4
    space_between_bars = 0.2
    space_between_regions = 1.5
    current_x = 0

    for region in sorted_regions:
        for country in region:
            values = data_dict[country]
            scenario_variable_CoC = np.round(values['scenario_variable_CoC'], decimals=4)
            if scenario_variable_CoC == 0:
                scenario_variable_CoC = max(values['scenario_country_convergence'],
                                            values['scenario_technology_convergence'])
            normalized_country.append(values['scenario_country_convergence'] / scenario_variable_CoC-1)
            normalized_technology.append(values['scenario_technology_convergence'] / scenario_variable_CoC-1)
            x_positions.append(current_x)
            current_x += 2 * bar_width + space_between_bars
        current_x += space_between_regions  # Add extra space between regions

    # Plot setup
    plt.figure(figsize=(15, 8))

    # Plot bars
    plt.bar(x_positions, normalized_country, width=bar_width, label='Country Convergence', alpha=0.7)
    plt.bar([x + bar_width for x in x_positions], normalized_technology, width=bar_width,
            label='Technology Convergence', alpha=0.7)

    # Add vertical lines and headers
    region_midpoints = []
    region_start = 0
    counter = 0
    for region, label in zip(sorted_regions, region_labels):
        region_end = region_start + len(region) * (2 * bar_width + space_between_bars)
        region_midpoints.append((region_start + region_end - space_between_bars) / 2)
        if counter<2:
            plt.axvline(x=region_end + space_between_bars/2 + 0.2, color='grey', linestyle='--', alpha=0.7)
        counter += 1
        region_start = region_end + space_between_regions

    # Add region headers
    for midpoint, label in zip(region_midpoints, region_labels):
        plt.text(midpoint, plt.ylim()[1] * 1.02, label, ha='center', va='bottom', fontsize=15, fontweight='bold')

    # Add labels and ticks
    plt.xticks([x + bar_width / 2 for x in x_positions], sum(sorted_regions, []), rotation=90,fontsize=16)
    plt.ylabel('Renewable capacity deployed relative to variable CoC scenario',fontsize=14)
    plt.legend(fontsize=14,loc="upper right")
    plt.yticks(fontsize=16)

    # Add grid and layout adjustments
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show plot
    plt.show()
    #plt.savefig("../../../Plots_presentation/normalized_capacity_tech_contry.svg",dpi=300)



######## From here all plots are computed ###############
# 1. Uniform vs. variable CoC
if results_path == 'outputs/PI_annuity/':
    r = r_annuity
    scenario_labels = ["Uniform CoC", "Variable CoC"]
    # Figure 4.4
    #######--------- Storage capacity over time ----------############
    capacity = r.get_total("capacity")
    storage_technologies = ["battery", "hydrogen_storage", "pumped_hydro"]
    scenarios = ["scenario_", "scenario_variable_CoC"]
    results = {scenario: {tech: [] for tech in storage_technologies} for scenario in scenarios}
    for time in range(15):
        for scenario in scenarios:
            for technology in storage_technologies:
                results[scenario][technology].append(capacity[time][scenario][technology]["energy"].sum())

    years = [str(year) for year in range(2022, 2051, 2)]
    storage_plot(results, years)

    #Figure B.3
    ########- Heating capacity over time -##########
    capacity = r.get_total("capacity")
    heating_technologies = ['biomass_boiler', 'biomass_boiler_DH', 'district_heating_grid',
       'electrode_boiler', 'electrode_boiler_DH', 'hard_coal_boiler_DH',
       'heat_pump', 'heat_pump_DH',
       'natural_gas_boiler', 'natural_gas_boiler_DH', 'oil_boiler', 'oil_boiler_DH',
       'waste_boiler_DH']
    labels_technologies["biomass_boiler"] = "Biomass Boiler"
    labels_technologies["biomass_boiler_DH"] = "Biomass Boiler DH"
    labels_technologies["district_heating_grid"] = "District Heating Grid"
    labels_technologies["electrode_boiler"] = "Electrode Boiler"
    labels_technologies["electrode_boiler_DH"] = "Electrode Boiler DH"
    labels_technologies["hard_coal_boiler_DH"] = "Hard Coal Boiler DH"
    labels_technologies["heat_pump"] = "Heat Pump"
    labels_technologies["heat_pump_DH"] = "Heat Pump DH"
    labels_technologies["natural_gas_boiler"] = "Natural Gas Boiler"
    labels_technologies["natural_gas_boiler_DH"] = "Natural Gas Boiler DH"
    labels_technologies["oil_boiler"] = "Oil Boiler"
    labels_technologies["oil_boiler_DH"] = "Oil Boiler DH"
    labels_technologies["waste_boiler_DH"] = "Waste Boiler DH"
    scenarios = ["scenario_", "scenario_variable_CoC"]
    results = {scenario: {tech: [] for tech in heating_technologies} for scenario in scenarios}
    for time in range(15):
        for scenario in scenarios:
            for technology in heating_technologies:
                results[scenario][technology].append(capacity[time][scenario][technology]["power"].sum())
    years = [str(year) for year in range(2022, 2051, 2)]
    storage_plot(results, years)



    #Figure 4.5
    ##### ------------ Bar plot relative difference in RE per region ---------
    capacity = r.get_total("capacity")
    time_steps = [0, 4, 9, 14]
    scenarios = list(capacity.index.levels[0])
    capacity_type = "power"
    regions = ["NWE", "SE", "EE"]
    # iterate over the relevant time steps
    scenario_0 = {"NWE": {}, "SE": {}, "EE": {}}
    scenario_1 = {"NWE": {}, "SE": {}, "EE": {}}
    for time in time_steps:
        data = capacity[time]
        for scenario in scenarios:
            data_scenario = data[scenario]
            for technology in RE_technologies:
                data_scenario_technology = data_scenario[technology][capacity_type]
                for region in regions:
                    sum_capacity = data_scenario_technology[eval(region)].sum()
                    if scenario == "scenario_":
                        try:
                            scenario_0[region][technology].append(sum_capacity)
                        except KeyError:
                            scenario_0[region][technology] = [sum_capacity]
                    elif scenario == "scenario_variable_CoC":
                        try:
                            scenario_1[region][technology].append(sum_capacity)
                        except KeyError:
                            scenario_1[region][technology] = [sum_capacity]
    technologies = RE_technologies
    years = ['2022', '2030', '2040', '2050']
    #compute relative difference in capacity deployments
    relative_diff = {
        region: {
            tech: np.nan_to_num((np.round(np.array(scenario_1[region][tech]), decimals=1) -
                                 np.round(np.array(scenario_0[region][tech]), decimals=1)) /
                                np.round(np.array(scenario_0[region][tech]), decimals=1), nan=0) * 100
            for tech in technologies
        }
        for region in regions
    }
    # Prepare for plotting
    bar_width = 0.2
    x_positions = np.arange(len(years)) * (len(regions) * bar_width + 0.2)  # Extra spacing for groups
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'cyan']
    fig, ax = plt.subplots(figsize=(14, 8))
    # Create stacked bars
    for region_idx, region in enumerate(regions):
        for year_idx, year in enumerate(years):
            x = x_positions[year_idx] + region_idx * bar_width
            bottom = 0  # Track the stack
            for tech_idx, tech in enumerate(technologies):
                value = relative_diff[region][tech][year_idx]
                ax.bar(x, value, bar_width, label=labels_technologies[tech] if year_idx == 0 and region_idx == 0 else "",
                       bottom=bottom if value >= 0 else None,
                       color=colors_dict[tech],edgecolor='black')
                if value >= 0:
                    bottom += value
            # Add region label at the top of the stacked bar
            total_height = bottom if bottom > 0 else 0  # Adjust for positive bars only
            ax.text(x, total_height, region, ha='center', va='bottom', fontsize=12, color='black')
    # Add labels and legend
    ax.set_xticks(x_positions + (len(regions) - 1) * bar_width / 2)
    ax.set_xticklabels(years, fontsize = 14)
    ax.set_xlabel("Year", fontsize = 14)
    ax.set_ylabel("Relative Difference (%)", fontsize = 14)
    ax.legend(title="Technologies", loc="upper left", bbox_to_anchor=(1, 1))
    ax.axhline(0, color='black', linewidth=0.8)  # X-axis at y=0
    ax.set_title("Relative Difference in Capacity by Region and Year")
    plt.yticks(fontsize=14)
    plt.tight_layout()
    #plt.savefig("../../../Plots_presentation/relative_difference_RE.svg", dpi=300)
    plt.show()


    #Figure B.8
    ###### ------ bar plot of total capacity installed ---------########
    elec_conv_technologies = [
        "biomass_plant",
        "biomass_plant_CCS",
        "hard_coal_plant",
        "hard_coal_plant_CCS",
        "lignite_coal_plant",
        "natural_gas_turbine",
        "natural_gas_turbine_CCS",
        "nuclear",
        "oil_plant",
        "photovoltaics",
        "reservoir_hydro",
        "run-of-river_hydro",
        "waste_plant",
        "wind_offshore",
        "wind_onshore"
    ]
    capacity = r.get_total("capacity")
    time_steps = [i for i in range(15)]
    scenarios = list(capacity.index.levels[0])
    capacity_type = "power"
    # iterate over the relevant time steps
    scenario_0 = {tech: [] for tech in elec_conv_technologies}
    scenario_1 = {tech: [] for tech in elec_conv_technologies}
    for time in time_steps:
        data = capacity[time]
        for scenario in scenarios:
            data_scenario = data[scenario]
            total_capacity = 0
            for technology in elec_conv_technologies:
                sum_capacity = data_scenario[technology][capacity_type].sum()
                if scenario == "scenario_":
                    scenario_0[technology].append(sum_capacity)
                elif scenario == "scenario_variable_CoC":
                    scenario_1[technology].append(sum_capacity)

    #define aggregation groups
    aggregation_groups = {"biomass_plant": ["biomass_plant", "biomass_plant_CCS"], "Others": ["oil_plant","waste_plant"],"natural_gas_turbine": ["natural_gas_turbine","natural_gas_turbine_CCS"],"Hydro":["reservoir_hydro","run-of-river_hydro"],"Coal":["hard_coal_plant","hard_coal_plant_CCS","lignite_coal_plant"]}
    years = [str(year) for year in range(2022, 2051, 2)]
    plot_stacked_bars_with_patterns_aggregated(scenario_0, scenario_1, years,aggregation_groups,scenario_labels=("Uniform CoC", "Variable CoC"),title="Total capacity installed per technology",y_label="Capacity in GW")



    #Figure B.9 and B.10
    ### --- Eruope plot PV, Wind potential -----#########
    #PV potential
    pv = r.get_total("max_load")[14]["scenario_"]["photovoltaics"].xs(key="power", level="capacity_type").groupby(
        level="location").sum()
    pv.index = pv.index.str.replace("UK", "GB").str.replace("EL", "GR")
    values = list(pv.values)
    set_nodes = list(pv.index)
    europe_plot(set_nodes, values, title="PV potential", y_axis="Full load hours",color = 'Oranges' ,center=None)
    #Wind potential
    wind = r.get_total("max_load")[14]["scenario_"]["wind_onshore"].xs(key="power", level="capacity_type").groupby(
        level="location").sum()
    wind.index = wind.index.str.replace("UK", "GB").str.replace("EL", "GR")
    values = list(wind.values)
    set_nodes = list(wind.index)
    europe_plot(set_nodes, values, title="Wind potential", y_axis="Full load hours", color = 'Blues',center=None)

    #Figure B.1 and 3.1
    ###### Average CoC per country ########
    avg_WACC ={'AT': 0.04739185185185186, 'BE': 0.04559185185185185, 'BG': 0.06735407407407408, 'CH': 0.04856370370370372,
     'CZ': 0.0770637037037037, 'DE': 0.04248222222222223, 'DK': 0.04644962962962963, 'EE': 0.057925185185185184,
     'GR': 0.09844000000000001, 'ES': 0.060840000000000005, 'FI': 0.04718222222222221, 'FR': 0.04706888888888888,
     'HR': 0.0792962962962963, 'HU': 0.07588666666666667, 'IE': 0.054696296296296305, 'IT': 0.05781111111111111,
     'LT': 0.0556962962962963, 'LU': 0.04225407407407408, 'LV': 0.053696296296296304, 'NL': 0.04806888888888889,
     'NO': 0.051649629629629636, 'PL': 0.06815407407407406, 'PT': 0.06389629629629628, 'RO': 0.0773251851851852,
     'SE': 0.0496111111111111, 'SI': 0.06209629629629629, 'SK': 0.05011555555555556, 'GB': 0.058068888888888885}
    values = list(avg_WACC.values())
    set_nodes = list(avg_WACC.keys())
    europe_plot(set_nodes, values, title="Average CoC per country", y_axis="Average CoC",color='RdBu_r',center=np.mean(list(avg_WACC.values())))

    #Figure B.2
    ###### Average CoC per technology ########
    avg_WACC_renewables = {'photovoltaics': 0.05953928571428572, 'reservoir_hydro': 0.050564285714285714,
     'biomass_plant': 0.06547857142857143, 'wind_onshore': 0.09282142857142858,
     'wind_offshore': 0.10317857142857143}
    plot_sorted_wacc(avg_WACC_renewables, colors_dict,labels_technologies ,x_tick_size=10, y_label_size=10)

    #Figure B.4,B.5,B.6,B.7 and 4.3
    #########################renewable capacity difference europe map (2022,2030,2040,2050) ########################
    RE_technologies = ["biomass_plant_CCS", "biomass_plant", "photovoltaics", "wind_offshore", "wind_onshore",
                       "reservoir_hydro", "run-of-river_hydro"]
    capacity = r.get_total("capacity")[14]["scenario_"]["photovoltaics"]["power"]
    capacity.index = capacity.index.str.replace("UK", "GB").str.replace("EL", "GR")
    countries = list(capacity.index)
    capacity = r.get_total("capacity")
    time_steps = [0,4,9,14]
    years = ["2022","2030","2040","2050"]
    for i,time in enumerate(time_steps):
        scenario_0 = {key: 0 for key in countries}
        scenario_1 = {key: 0 for key in countries}
        data = capacity[time]
        for scenario in scenarios:
            data_scenario = data[scenario]
            sum_technologies = {key: 0 for key in countries}
            for technology in RE_technologies:
                data_scenario_technology = data_scenario[technology][capacity_type]
                data_scenario_technology.index = data_scenario_technology.index.str.replace("UK", "GB").str.replace("EL", "GR")
                for region in countries:
                    sum_capacity = data_scenario_technology[region]
                    sum_technologies[region] += sum_capacity
            if scenario == "scenario_":
                for region in countries:
                    scenario_0[region] = sum_technologies[region]
            elif scenario == "scenario_variable_CoC":
                for region in countries:
                    scenario_1[region] = sum_technologies[region]
        relative_diff = {
            region: np.nan_to_num((np.round(scenario_1[region], decimals=1) -
                                     np.round(scenario_0[region], decimals=1)) /
                                    np.round(scenario_0[region], decimals=1), nan=0) * 100
            for region in countries
        }
        #get all values from relative_diff
        values = list(relative_diff.values())
        #get all keys from relative_diff
        set_nodes = list(relative_diff.keys())
        #plt map
        europe_plot(set_nodes, values,title = f"{years[i]}" ,y_axis = "Difference in capacity in %", center=0, color="RdBu",vmin = -100,vmax =90)


    #Figure 4.6 + plots in presentation
    ################## LCOE per country ############################
    capacity = r.get_total("capacity")[14]["scenario_"]["photovoltaics"]["power"]
    capacity.index = capacity.index.str.replace("UK", "GB").str.replace("EL", "GR")
    countries = list(capacity.index)
    npc = r.get_total("net_present_cost_yearly_technology")
    technologies = [
        "biomass_plant",
        "biomass_plant_CCS",
        "hard_coal_plant",
        "hard_coal_plant_CCS",
        "lignite_coal_plant",
        "natural_gas_turbine",
        "natural_gas_turbine_CCS",
        "nuclear",
        "oil_plant",
        "photovoltaics",
        "reservoir_hydro",
        "run-of-river_hydro",
        "waste_plant",
        "wind_offshore",
        "wind_onshore"
    ]
    time_steps = [14]
    years = ["2050"]
    production = r.get_total("flow_conversion_output")
    for i, time in enumerate(time_steps):
        scenario_0 = {key: 0 for key in countries}
        scenario_1 = {key: 0 for key in countries}
        data = npc.sum(axis=1)
        for scenario in scenarios:
            production_scenario = production.apply(lambda row: sum(row[col] / ((1+r.get_total("discount_rate")[0][scenario]) ** (col * 2)) for col in list(production.columns)), axis=1)[scenario]
            data_scenario = data[scenario]
            sum_technologies = {key: 0 for key in countries}
            sum_production = {region:0 for region in countries}
            for technology in technologies:
                data_scenario_technology = data_scenario[technology]
                data_scenario_technology.index = data_scenario_technology.index.str.replace("UK", "GB").str.replace(
                    "EL", "GR")
                production_technology = production_scenario[technology]["electricity"]
                production_technology.index = production_technology.index.str.replace("UK", "GB").str.replace(
                    "EL", "GR")
                for region in countries:
                    costs = data_scenario_technology[region]
                    sum_technologies[region] += costs
                    sum_production[region] += production_technology[region]
            if scenario == "scenario_":
                for region in countries:
                    scenario_0[region] = sum_technologies[region]
                #norm all entries of the dictionaries with total production
                normed_scenario_0 = {key: value / sum_production[key]*100 for key, value in scenario_0.items()}
                normed_DE_0 = {key: (value / sum_production[key]*100 if key in ["DE", "HU"] else 0)for key, value in scenario_0.items()}
            elif scenario == "scenario_variable_CoC":
                for region in countries:
                    scenario_1[region] = sum_technologies[region]
                normed_scenario_1_0 = {key: 0 / sum_production[key] for key, value in scenario_1.items()} #value
                normed_scenario_1 = {key: value / sum_production[key]*100 for key, value in scenario_1.items()}
                normed_DE_1 = {key: (value / sum_production[key]*100 if key in ["DE", "HU"] else 0)for key, value in scenario_1.items()}

        sorted_countries = [['LV', 'FR', 'LU', 'SE', 'GB', 'DE', 'DK', 'NL', 'CH', 'BE', 'EE', 'FI', 'IE', 'AT', 'NO', 'LT'], ['HR', 'GR', 'IT', 'ES', 'PT'], ['SI', 'RO', 'SK', 'CZ', 'BG', 'HU', 'PL']]


        #plots -> divided into regions
        #1. Just uniform DE and HU
        data_1 = {"scenario_0":normed_DE_0, "scenario_1":normed_scenario_1_0}
        #2. Uniform and variable DE and HU
        data_2 = {"scenario_0": normed_DE_0, "scenario_1": normed_DE_1}
        #3. Uniform all countries and variable of DE and HU
        data_3 = {"scenario_0": normed_scenario_0, "scenario_1": normed_DE_1}
        #4. Uniform all countries and variable all countries
        data_4 = {"scenario_0": normed_scenario_0, "scenario_1": normed_scenario_1}
        #Plots for presentation
        plot_bar_plots_region(data_1,y_label="LCOE in cent/kWh",external_sorting=sorted_countries)
        plot_bar_plots_region(data_2,y_label="LCOE in cent/kWh",external_sorting=sorted_countries)
        plot_bar_plots_region(data_3,y_label="LCOE in cent/kWh",external_sorting=sorted_countries)
        #Plot 4.6
        plot_bar_plots_region(data_4,y_label="LCOE in cent/kWh",external_sorting=sorted_countries,labelsize=16)




elif results_path == 'outputs/PI_yearly_new/':
    #assign right settings
    r = r_yearly
    scenario_1 = scenario_1_yearly
    scenario_2 = scenario_2_yearly

    #Figure 4.7
    #######-------- Renewable energy capacity addition per country compared to variable CoC --------#########
    capacity = r.get_total("capacity")[14]["scenario_"]["photovoltaics"]["power"]
    capacity.index = capacity.index.str.replace("UK", "GB").str.replace("EL", "GR")
    countries = list(capacity.index)
    capacity_addition = r.get_total("capacity_addition").sum(axis=1)
    capacity_addition_var = r_annuity.get_total("capacity_addition").sum(axis=1)
    green_technologies = [
        "biomass_plant",
        "biomass_plant_CCS",
        "photovoltaics",
        "reservoir_hydro",
        "run-of-river_hydro",
        "waste_plant",
        "wind_offshore",
        "wind_onshore"
    ]
    scenarios = ["scenario_country_convergence", "scenario_variable_CoC","scenario_technology_convergence"]
    results = {country: {scenario: 0 for scenario in scenarios} for country in
               countries}
    for scenario in scenarios:
        for tech in green_technologies:
            if scenario == "scenario_variable_CoC":
                capacity_addition_tech = capacity_addition_var[scenario][tech]["power"]
            else:
                capacity_addition_tech = capacity_addition[scenario][tech]["power"]
            capacity_addition_tech.index = capacity_addition_tech.index.str.replace("UK", "GB").str.replace("EL", "GR")
            for country in countries:
                results[country][scenario] += np.round(capacity_addition_tech[country], decimals=4)
    # Plotting
    plot_normalized_scenarios(results)

    #Figure 4.8 and 4.9
    ##### ------------ Gini plot --------------------------------------------
    capacity = r.get_total("capacity")[14]["scenario_"]["photovoltaics"]["power"]
    capacity.index = capacity.index.str.replace("UK", "GB").str.replace("EL", "GR")
    countries = list(capacity.index)
    demand = r.get_total("demand").sum(axis=1)
    capacity_addition = r.get_total("capacity_addition").sum(axis=1)
    production = r.get_total("flow_conversion_output").sum(axis=1)
    demand_tech = r.get_total("flow_conversion_input").sum(axis=1)
    scenarios = ["scenario_country_convergence", "scenario_technology_convergence"]
    elec_conv_technologies = [
        "biomass_plant",
        "biomass_plant_CCS",
        "hard_coal_plant",
        "hard_coal_plant_CCS",
        "lignite_coal_plant",
        "natural_gas_turbine",
        "natural_gas_turbine_CCS",
        "nuclear",
        "oil_plant",
        "photovoltaics",
        "reservoir_hydro",
        "run-of-river_hydro",
        "waste_plant",
        "wind_offshore",
        "wind_onshore"
    ]
    time_steps = [14]
    years = ["2050"]
    for i, time in enumerate(time_steps):
        production_0 = {key: 0 for key in countries}
        production_1 = {key: 0 for key in countries}
        demand_0 = {key: 0 for key in countries}
        demand_1 = {key: 0 for key in countries}
        for scenario in scenarios:
            demand_scenario = demand[scenario]["electricity"]
            demand_scenario.index = demand_scenario.index.str.replace("UK", "GB").str.replace(
                "EL", "GR")
            demand_tech_scenario = demand_tech[scenario].xs(key="electricity", level="carrier").groupby(
                level="node").sum()
            demand_tech_scenario.index = demand_tech_scenario.index.str.replace("UK", "GB").str.replace("EL", "GR")
            production_scenario = production[scenario].xs(key="electricity", level="carrier").loc[
                green_technologies].groupby(level="node").sum()
            production_scenario.index = production_scenario.index.str.replace("UK", "GB").str.replace(
                "EL", "GR")
            # for technology in green_technologies:
            for region in countries:
                # sum_capacity = data_scenario_technology[region]
                if scenario == "scenario_country_convergence":
                    demand_0[region] = demand_scenario[region] + demand_tech_scenario[region]
                    production_0[region] = production_scenario[region]
                elif scenario == "scenario_technology_convergence":
                    demand_1[region] = demand_scenario[region] + demand_tech_scenario[region]
                    production_1[region] = production_scenario[region]

        # normalize scenario_0 dictionaries
        # get total production over all countries
        total_production_0 = sum(production_0.values())
        # get total demand over all countries
        total_demand_0 = sum(demand_0.values())
        # normalize production
        production_0 = {key: value / total_production_0 for key, value in production_0.items()}
        # normalize demand
        demand_0 = {key: value / total_demand_0 for key, value in demand_0.items()}

        # normalize scenario_1 dictionaries
        # get total production over all countries
        total_production_1 = sum(production_1.values())
        # get total demand over all countries
        total_demand_1 = sum(demand_1.values())
        # normalize production
        production_1 = {key: value / total_production_1 for key, value in production_1.items()}
        # normalize demand
        demand_1 = {key: value / total_demand_1 for key, value in demand_1.items()}

        #for additional line
        x_country = [0.0, 0.002779176400044366, 0.004456858456113045, 0.008865053474450504, 0.03811835567033309, 0.19957987576081826, 0.21799228199996135, 0.2263480646599333, 0.3673145302759623, 0.40495372452869927, 0.4422943663266637, 0.5521268418697408, 0.6542814229503495, 0.6745723840811002, 0.7353728445641277, 0.748085491871732, 0.7586625047743732, 0.7683580412071909, 0.7921600347808772, 0.867393942415517, 0.873589944623974, 0.8877863686080852, 0.9038644304327126, 0.919894559716499, 0.9392596323294098, 0.9734459259428223, 0.992909352201738, 0.9954689264364197, 1.0]
        y_country = [0.0, 6.9503429681333e-05, 0.0006394463952595175, 0.002575706734148516, 0.02027384067160739, 0.1275592341235946, 0.13984258959078333, 0.1454472400570347, 0.25248091042784193, 0.28241050392509365, 0.3147007231821116, 0.4175286625102954, 0.5182222443354256, 0.5387031597450367, 0.6041026522690381, 0.6187288306470666, 0.6313983386027785, 0.6430793997580677, 0.6723163764943488, 0.7708539736809727, 0.7791669540521157, 0.7984308295693593, 0.8206787521671517, 0.8434138436986247, 0.8737087373904204, 0.9287946183002582, 0.9778388126146615, 0.9845192552445693, 1.0]
        x_tech = [0.0, 0.0016880853442756264, 0.00436081374917132, 0.01268129922970935, 0.01698932077591688, 0.03502512546118684, 0.07278397408646713, 0.10215066847263153, 0.2434887985681784, 0.28141843966331426, 0.4465356940813208, 0.5480270388261814, 0.5536372546882914, 0.5560915617590216, 0.6657627516699469, 0.7235147955879183, 0.7331847989173965, 0.7432812875564319, 0.8172676711578916, 0.8386399895324658, 0.8527075264479012, 0.8770207845141275, 0.8927846978821765, 0.9078645441534734, 0.9210208401151718, 0.9563482848349958, 0.9759498287530305, 0.9956993196697097, 1.0]
        y_tech = [0.0, 0.0001400793722146212, 0.0007947034380091865, 0.003678775745576247, 0.005467786815623113, 0.015157518382980465, 0.043795203066996656, 0.0668064239270793, 0.17892829436094798, 0.2108553228280489, 0.35886505480699193, 0.45103851147319984, 0.45625365430982573, 0.45860397073353165, 0.5643044969682272, 0.6211167075958789, 0.6319759658141518, 0.6434832546063831, 0.7296705930333693, 0.7550725030717385, 0.7725316504811722, 0.8028018996892722, 0.8230411751741291, 0.8431233315260861, 0.8609705337770249, 0.9185596252931233, 0.9512019544712241, 0.9889810322821458, 0.9999999999999999]
        #plot gini
        gini_plot(demand_0, production_0, additional_line=(x_tech, y_tech,"Technology convergence"))
        gini_plot(demand_1, production_1, additional_line=(x_country, y_country,"Country convergence"))
        gini_plot_multiple([(demand_0, production_0), (demand_1, production_1)],
                           ["Country convergence", "Technology convergence"])

        # Calculate Gini coefficient
        def gini_coefficient(demand, production):
            countries = demand.keys()
            demand = np.array([demand[country] for country in countries])
            production = np.array([production[country] for country in countries])
            # Calculate the fairness ratio
            fairness_ratio = production / demand

            # Sort the fairness ratios
            sorted_indices = np.argsort(fairness_ratio)
            sorted_fairness = fairness_ratio[sorted_indices]
            sorted_demand = demand[sorted_indices]

            # Compute the cumulative proportions
            cumulative_production = np.cumsum(sorted_fairness * sorted_demand)
            cumulative_demand = np.cumsum(sorted_demand)

            # Normalize the cumulative proportions
            cumulative_production /= cumulative_production[-1]
            cumulative_demand /= cumulative_demand[-1]

            # Compute the Gini coefficient using the trapezoidal rule
            gini = 1 - 2 * np.trapz(cumulative_production, cumulative_demand)
            return gini


        # Calculate and print the Gini coefficient
        gini = gini_coefficient(demand_0, production_0)
        print(f"Gini Coefficient of country convergence: {gini:.4f}")
        gini = gini_coefficient(demand_1, production_1)
        print(f"Gini Coefficient of technology convergence: {gini:.4f}")

    #Figure 4.10 and 4.11
    ############ ------------- LCOE with variable LCOE ##### ---------------
    capacity = r.get_total("capacity")[14]["scenario_"]["photovoltaics"]["power"]
    capacity.index = capacity.index.str.replace("UK", "GB").str.replace("EL", "GR")
    countries = list(capacity.index)
    npc = r.get_total("net_present_cost_yearly_technology")
    technologies = [
        "biomass_plant",
        "biomass_plant_CCS",
        "hard_coal_plant",
        "hard_coal_plant_CCS",
        "lignite_coal_plant",
        "natural_gas_turbine",
        "natural_gas_turbine_CCS",
        "nuclear",
        "oil_plant",
        "photovoltaics",
        "reservoir_hydro",
        "run-of-river_hydro",
        "waste_plant",
        "wind_offshore",
        "wind_onshore"
    ]
    time_steps = [14]
    years = ["2050"]
    production = r.get_total("flow_conversion_output")
    for i, time in enumerate(time_steps):
        scenario_0 = {key: 0 for key in countries}
        scenario_1 = {key: 0 for key in countries}
        data = npc.sum(axis=1)
        for scenario in scenarios:
            production_scenario = production.apply(lambda row: sum(row[col] / ((1+r.get_total("discount_rate")[0][scenario]) ** (col * 2)) for col in list(production.columns)), axis=1)[scenario]
            data_scenario = data[scenario]
            sum_technologies = {key: 0 for key in countries}
            sum_production = {region:0 for region in countries}
            for technology in technologies:
                data_scenario_technology = data_scenario[technology]
                data_scenario_technology.index = data_scenario_technology.index.str.replace("UK", "GB").str.replace(
                    "EL", "GR")
                production_technology = production_scenario[technology]["electricity"]
                production_technology.index = production_technology.index.str.replace("UK", "GB").str.replace(
                    "EL", "GR")
                for region in countries:
                    costs = data_scenario_technology[region]
                    sum_technologies[region] += costs
                    sum_production[region] += production_technology[region]
            if scenario == "scenario_country_convergence":
                for region in countries:
                    scenario_0[region] = sum_technologies[region]
                normed_scenario_0 = {key: value / sum_production[key] for key, value in scenario_0.items()}
            elif scenario == "scenario_technology_convergence":
                for region in countries:
                    scenario_1[region] = sum_technologies[region]
                normed_scenario_1 = {key: value / sum_production[key] for key, value in scenario_1.items()}

    LCOE_variable = {'AT': 0.08643204562876534, 'BE': 0.08499295661965142, 'BG': 0.11988311597913531, 'CH': 0.09685947742276296,
     'CZ': 0.10028231556204634, 'DE': 0.08793590009515483, 'DK': 0.10229933310830452, 'EE': 0.09787377327819424,
     'GR': 0.09984856891560256, 'ES': 0.09620895639365189, 'FI': 0.09732870817894128, 'FR': 0.13017941177693038,
     'HR': 0.10429030378585381, 'HU': 0.13992896473949626, 'IE': 0.09058866105808701, 'IT': 0.09574056021008868,
     'LT': 0.0871477102540566, 'LU': 0.11168550643338335, 'LV': 0.1708603328283931, 'NL': 0.10426167385667151,
     'NO': 0.07731915582845089, 'PL': 0.11737300980382576, 'PT': 0.08331637731390601, 'RO': 0.15179907928451752,
     'SE': 0.10496807866083367, 'SI': 0.13459581829476883, 'SK': 0.12510800872223124, 'GB': 0.11048051165795297}

    LCOE_uniform={'AT': 0.09029321954518431, 'BE': 0.10089011386687052, 'BG': 0.11219733622107884, 'CH': 0.10680677914858791,
     'CZ': 0.11604853276255865, 'DE': 0.10884738340098232, 'DK': 0.10881393923650892, 'EE': 0.098571461307816,
     'GR': 0.10037772429083057, 'ES': 0.09515082548244293, 'FI': 0.09748012504677717, 'FR': 0.1347389505399738,
     'HR': 0.1043909480491487, 'HU': 0.10714311949257149, 'IE': 0.0937683656578389, 'IT': 0.09753313186182874,
     'LT': 0.0857627546457811, 'LU': 0.12096574682877523, 'LV': 0.17027369408990647, 'NL': 0.10825313761223634,
     'NO': 0.09005596159215946, 'PL': 0.1050892546456718, 'PT': 0.08304056104402163, 'RO': 0.1291838083567891,
     'SE': 0.1147210147696452, 'SI': 0.13354637546791617, 'SK': 0.12471136574350845, 'GB': 0.10909907912807415}

    diff_tech = {
            region: np.nan_to_num((np.round(normed_scenario_1[region], decimals=5) -
                     np.round(LCOE_variable[region], decimals=5))/(np.round(LCOE_variable[region], decimals=5)), nan=0) * 100
            for region in countries
        }
    set_nodes = countries
    values = diff_tech.values()
    europe_plot(set_nodes, values, title=f"Difference in LCOE",y_axis = "Difference in LCOE in %",color='RdBu_r',vmin=-30,vmax=50)

    diff_country = {
        region: np.nan_to_num((np.round(normed_scenario_0[region], decimals=5) -
                               np.round(LCOE_variable[region], decimals=5)) / (
                                  np.round(LCOE_variable[region], decimals=5)), nan=0) * 100
        for region in countries
    }

    set_nodes = countries
    values = diff_country.values()
    europe_plot(set_nodes, values, title=f"Difference in LCOE",y_axis="Difference in LCOE in %",color='RdBu_r',vmin=-30,vmax=50)

