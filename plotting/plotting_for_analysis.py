import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
import importlib
import os
import json
import pandas as pd

from zen_garden.postprocess.results.results import Results
from zen_garden.model.default_config import Config,Analysis,Solver,System

# read in results
def read_in_results(model_name):
    data_path = "/Users/lukashegner/PycharmProjects/ZEN-garden/data/"
    outputs_path = data_path + "outputs"
    # data_path = "/Volumes/Elements10T/RRE_Myopic_Operation_Project/Model_Outputs/"
    # outputs_path = data_path + "/outputs"
    try:
        spec = importlib.util.spec_from_file_location("module", data_path + "config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    except FileNotFoundError:
        with open("config.json") as f:
            config = Config(json.load(f))

    if os.path.exists(
            out_folder := os.path.join(outputs_path, model_name)
    ):
        r = Results(out_folder)
    return r, config


# functions used in plotting functions:
def reformat_prompt(prompt):
    # Step 1: Replace underscores with spaces
    prompt_with_spaces = prompt.replace("_", " ")

    # Step 2: Capitalize the first letter of every word
    formatted_prompt = prompt_with_spaces.title()

    return formatted_prompt

def indexmapping(df, special_model_name=None):
    global model_name
    if special_model_name is not None:
        model_name = special_model_name

    if model_name == 'PI_1701_1to6':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '6_1',
            'scenario_1': '6_2',
            'scenario_10': '5_5',
            'scenario_11': '4_1',
            'scenario_15': '3_1',
            'scenario_16': '3_2',
            'scenario_17': '3_3',
            'scenario_18': '2_1',
            'scenario_19': '2_2',
            'scenario_2': '6_3',
            'scenario_20': '1_1',
            'scenario_3': '6_4',
            'scenario_4': '6_5',
            'scenario_5': '6_6',
            'scenario_6': '5_1',
            'scenario_7': '5_2',
            'scenario_9': '5_4'
        }
    elif model_name == 'PC_1701_1to6':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '6_1',
            'scenario_1': '6_2',
            'scenario_11': '4_1',
            'scenario_12': '4_2',
            'scenario_15': '3_1',
            'scenario_16': '3_2',
            'scenario_17': '3_3',
            'scenario_18': '2_1',
            'scenario_19': '2_2',
            'scenario_20': '1_1',
            'scenario_5': '6_6',
            'scenario_6': '5_1',
            'scenario_7': '5_2',
        }
    elif model_name == 'euler_no_lead':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '6_1',
            'scenario_1': '6_2',
            'scenario_2': '6_3',
            'scenario_3': '6_4',
            'scenario_4': '6_5',
            'scenario_5': '6_6',
            'scenario_6': '5_1',
            'scenario_7': '5_2',
            'scenario_8': '5_3',
            'scenario_9': '5_4',
            'scenario_10': '5_5',
            'scenario_11': '4_1',
            'scenario_12': '4_2',
            'scenario_13': '4_3',
            'scenario_14': '4_4',
            'scenario_15': '3_1',
            'scenario_16': '3_2',
            'scenario_17': '3_3',
            'scenario_18': '2_1',
            'scenario_19': '2_2',
            'scenario_20': '1_1'
        }
    elif model_name == 'cons_nolead_15_testing' \
            or model_name == 'uncons_lead_15_testing'\
            or model_name == 'cons_lead_15_testing_nodiscounting_higherovershootcost'\
            or model_name == 'uncons_lead_15_testing_nodiscounting'\
            or model_name == 'uncons_nolead_15_testing_nodiscounting':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '15_15',
            'scenario_1': '15_14'
        }
    elif model_name == 'uncons_lead_15_testing_nodiscounting':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '15_15'
        }
    elif model_name == 'cons_lead_15':
        # Define a dictionary mapping the current index values to the desired ones, 'scenario_11': '15_12',
        index_mapping = {
            'scenario_': '15_1',
            'scenario_1': '15_2',
            'scenario_2': '15_3',
            'scenario_3': '15_4',
            'scenario_4': '15_5',
            'scenario_5': '15_6',
            'scenario_6': '15_7',
            'scenario_7': '15_8',
            'scenario_8': '15_9',
            'scenario_9': '15_10',
            'scenario_10': '15_11',
            'scenario_12': '15_13',
            'scenario_13': '15_14',
            'scenario_14': '15_15'
        }
    elif model_name == 'cons_lead_15_testing_nodiscounting':
        # Define a dictionary mapping the current index values to the desired ones, 'scenario_11': '15_12',
        index_mapping = {
            'scenario_': '15_1',
            'scenario_1': '15_2',
            'scenario_2': '15_3',
            'scenario_3': '15_4',
            'scenario_4': '15_5',
            'scenario_5': '15_6',
            'scenario_6': '15_7',
            'scenario_7': '15_8',
            'scenario_8': '15_9',
            'scenario_9': '15_10',
            'scenario_10': '15_11',
            'scenario_11': '15_12',
            'scenario_12': '15_13',
            'scenario_13': '15_14',
            'scenario_14': '15_15'
        }
    elif model_name == 'cons_lead_1to7':
        index_mapping = {
            'scenario_': '7_1',
            'scenario_1': '7_2',
            'scenario_2': '7_3',
            'scenario_3': '7_4',
            'scenario_4': '7_5',
            'scenario_5': '7_6',
            'scenario_6': '7_7',
            'scenario_7': '6_1',
            'scenario_8': '6_2',
            'scenario_9': '6_3',
            'scenario_10': '6_4',
            'scenario_11': '6_5',
            'scenario_12': '6_6',
            'scenario_13': '5_1',
            'scenario_14': '5_2',
            'scenario_15': '5_3',
            'scenario_16': '5_4',
            'scenario_17': '5_5',
            'scenario_18': '4_1',
            'scenario_19': '4_2',
            'scenario_20': '4_3',
            'scenario_21': '4_4',
            'scenario_22': '3_1',
            'scenario_23': '3_2',
            'scenario_24': '3_3',
            'scenario_25': '2_1',
            'scenario_26': '2_2',
            'scenario_27': '1_1'
        }
    elif model_name == 'cons_lead_1to4':
        index_mapping = {
            'scenario_': '4_1',
            'scenario_1': '4_2',
            'scenario_2': '4_3',
            'scenario_3': '4_4',
            'scenario_4': '3_1',
            'scenario_5': '3_2',
            'scenario_6': '3_3',
            'scenario_7': '2_1',
            'scenario_8': '2_2',
            'scenario_9': '1_1'
        }
    elif model_name == 'cons_lead_5to7': # 'scenario_6': '7_7',
        index_mapping = {
            'scenario_': '7_1',
            'scenario_1': '7_2',
            'scenario_2': '7_3',
            'scenario_3': '7_4',
            'scenario_4': '7_5',
            'scenario_5': '7_6',
            'scenario_7': '6_1',
            'scenario_6': '7_7',
            'scenario_8': '6_2',
            'scenario_9': '6_3',
            'scenario_10': '6_4',
            'scenario_11': '6_5',
            'scenario_12': '6_6',
            'scenario_13': '5_1',
            'scenario_14': '5_2',
            'scenario_15': '5_3',
            'scenario_16': '5_4',
            'scenario_17': '5_5'
        }
    else:
        print("dataset not defined correctly")
        raise ValueError("Invalid dataset configuration: {}".format(model_name))

    df = df.rename(index=index_mapping)
    return df

def draw_lines_around_threshold(ax, threshold_matrix):
    n = len(threshold_matrix)  # Assuming square matrix
    for i in range(n):
        for j in range(n):
            if threshold_matrix[i][j] == 1:
                # Adjust lines to be half a step up and right
                half_step = 0.5
                # For cells on or below the diagonal from bottom left to top right (j >= n - i - 1)
                # Check below if not on the last row and within the interest area
                if i < n - 1 and threshold_matrix[i + 1][j] == 0 and j >= n - i - 2:
                    ax.plot([j + half_step, j + 1 + half_step], [n - i - 1 + half_step, n - i - 1 + half_step],
                            color='black', linewidth=2)
                # Check right if not on the last column
                if j < n - 1 and threshold_matrix[i][j + 1] == 0:
                    ax.plot([j + 1 + half_step, j + 1 + half_step], [n - i - 1 + half_step, n - i + half_step],
                            color='black', linewidth=2)
                # Check above if not in the first row and to the right of the diagonal
                if i > 0 and threshold_matrix[i - 1][j] == 0 and j >= n - i:
                    ax.plot([j + half_step, j + 1 + half_step], [n - i + half_step, n - i + half_step], color='black',
                            linewidth=2)
                # Check left if not in the first column and strictly below the diagonal
                if j > 0 and threshold_matrix[i][j - 1] == 0 and j >= n - i:
                    ax.plot([j + half_step, j + half_step], [n - i - 1 + half_step, n - i + half_step], color='black',
                            linewidth=2)


# Function to add data to the matrix, modified to ignore squares above the diagonal
def add_data_from_bottom(matrix, df, n):
    for key, value in df.items():
        col, row = map(int, key.split('_'))
        # Adjust indices for 0-based indexing and start from the bottom row
        adjusted_row = n - row
        adjusted_col = col - 1
        matrix[adjusted_row][adjusted_col] = value

def threshold_matrix_diagonal_bl_tr(matrix, threshold):
    """
    Returns a new matrix (list of lists) where values on and below the diagonal from bottom left to top right
    and above the threshold are set to 1, and all other values are set to 0.

    :param matrix: A list of lists representing the matrix.
    :param threshold: The threshold value to compare against.
    :return: A new list of lists with values set based on the condition.
    """
    n = len(matrix)  # Assuming the matrix is square
    new_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            # Including values on and below the diagonal from bottom left to top right (j >= n - i - 1)
            if j >= n - i - 1 and matrix[i][j] > threshold:
                new_matrix[i][j] = 1

    return new_matrix

def emissions_threshold_matrix_creation(model_list):
    global n
    print("creating emissions threshold matrix")
    # Initialize an empty list to store DataFrames from each model
    emissions_dfs = []
    # Iterate over each model name in the list
    for emissions_name in model_list:
        r, config = read_in_results(emissions_name)
        temp_emissions_df = r.get_total("carbon_emissions_cumulative")
        temp_emissions_df= temp_emissions_df.iloc[:, -1]
        # indexing
        temp_emissions_df = indexmapping(temp_emissions_df, special_model_name=emissions_name)
        # Append the processed DataFrame to the list
        emissions_dfs.append(temp_emissions_df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    emissions_df = pd.concat(emissions_dfs, axis=0, join='outer')
    carbon_budget = 10076

    max_value = emissions_df.max()
    # Normalize the dataset values to a range of 0 to 1 for color mapping
    normalized_emissions_df = (emissions_df - carbon_budget) / (max_value - carbon_budget)

    # Sample matrix for emissions line
    emissions_matrix = []

    # Extending the pattern to the rest of the matrix
    for i in range(n):
        row = [1] * (n - i - 1) + [0] * (i + 1)
        emissions_matrix.append(row)

    # Add the dataset to the matrix, filling from the bottom
    add_data_from_bottom(emissions_matrix, normalized_emissions_df, n)

    threshold = 1  # % by which cumulative emissions may be above the carbon budget
    emission_threshold_matrix = threshold_matrix_diagonal_bl_tr(emissions_matrix, threshold/100)

    return emission_threshold_matrix

def plot_stairs(normalized_df, threshold_matrix, max_value, min_value, fig_title, cbar_label):
    print("plotting stairs")
    # Sample matrix to color the squares
    matrix = []
    # Extending the pattern to the rest of the matrix
    for i in range(n):
        row = [1] * (n - i - 1) + [0] * (i + 1)
        matrix.append(row)

    # Add the dataset to the matrix, filling from the bottom
    add_data_from_bottom(matrix, normalized_df, n)

    # Creating a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Setting the ticks at the edges of each grid square
    ax.set_xticks(np.arange(0.5, n + 2, 1))
    ax.set_yticks(np.arange(0.5, n + 2, 1))

    # Setting the minor ticks at the center of each grid square for labeling
    ax.set_xticks(np.arange(1, n + 1, 1), minor=True, )
    ax.set_yticks(np.arange(1, n + 1, 1), minor=True)

    # Labeling the axes
    ax.set_xlabel("Investment Foresight Horizon", fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("Operation Foresight Horizon", fontsize=14)

    # Adding a grid
    ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)

    # Setting axis limits
    ax.set_xlim(0, n + 1)
    ax.set_ylim(0, n + 1)

    # Generate labels [2, 4, 6, ..., 2*n] for both axes
    labels = [str(2 * i) for i in range(1, n + 1)]

    # Setting labels for the minor ticks (centered on the squares)
    ax.set_xticklabels(labels, minor=True)
    ax.set_yticklabels(labels, minor=True)
    ax.tick_params(axis='x', which='minor', length=0)  # Hiding minor tick marks for X-axis
    ax.tick_params(axis='y', which='minor', length=0)  # Hiding minor tick marks for Y-axis

    # Moving Y-axis numbering to the right-hand side of the plot
    ax.tick_params(axis='y', which='minor', labelright=True, labelleft=False)
    # Removing labels from the major ticks
    ax.tick_params(axis='both', which='major', labelbottom=False, labelright=False)

    # Setting aspect of the plot to be equal
    ax.set_aspect('equal', adjustable='box')

    # Create a red colormap with a gradient from strong red for high values to light red for low values
    cmap = sns.light_palette("red", as_cmap=True)

    # Coloring the squares based on the matrix
    for i in range(-1, n + 1):
        for j in range(-1, n + 1):
            if i == -1 or i == n or j == -1 or j == n:
                border_square = patches.Rectangle((j + 0.5, n - i - .5), 1, 1, color='grey', alpha=0.5)
                ax.add_patch(border_square)
            if 0 <= i < n and 0 <= j < n:
                if j < n - 1 - i:
                    # Adding a colored square at the specified location
                    square = patches.Rectangle((j + 0.5, n - i - .5), 1, 1, color='grey', alpha=0.9)  # light red color
                    ax.add_patch(square)
                elif matrix[i][j] != 0:
                    # Get color from color map using the normalized value
                    color = cmap(matrix[i][j])
                    # Adding a colored square at the specified location
                    square = patches.Rectangle((j + 0.5, n - i - .5), 1, 1, facecolor=color)
                    ax.add_patch(square)

    # Call the function to draw lines around 1s adjacent to 0s
    draw_lines_around_threshold(ax, threshold_matrix)

    # Create a ScalarMappable with our red color map and the normalization
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=((min_value / min_value) - 1) * 100,
                                                  vmax=((max_value / min_value) - 1) * 100))

    # plt.title(fig_title)

    # Create the colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
    cbar.set_label(cbar_label, fontsize=12)
    # Reverse the direction of the colorbar
    cbar.ax.invert_xaxis()

    # Display the plot
    plt.show(block=True)
    return

def get_fig_title(plot_name, model_list=None, feature=None, formatted_tech=None, costs=None, granularity=None, year_str=None, technology=None):
    fig_title = None
    if [plot_name, feature] == ["stair_plot", "cumulative_emissions"]:
        if len(model_list) == 1:
            if model_name == 'PI_1701_1to6':
                fig_title = 'Total Cumulative Eimssions above Carbon Budget' + '\nUnconstrained Technology Deployment' + '\nNo Lead Times'
            elif model_name == 'PC_1701_1to6':
                fig_title = 'Total Cumulative Eimssions above Carbon Budget' + '\nCoanstrained Technology Deployment' + '\nNo Lead Times'
            elif model_name == 'euler_no_lead':
                fig_title = 'Total Cumulative Eimssions above Carbon Budget' + '\nConstrained Technology Deployment' + '\nNo Lead Times'
            else:
                print("dataset not defined correctly")
                raise ValueError("Invalid dataset configuration: {}".format(model_name))
    elif [plot_name, feature] == ["stair_plot", "costs"]:
        if len(model_list) == 1:
            if model_name == 'PI_1701_1to6':
                fig_title = 'Total System Costs above Cheapest Option' + '\nUnconstrained Technology Deployment' + '\nNo Lead Times'
            elif model_name == 'PC_1701_1to6':
                fig_title = 'Total System Costs above Cheapest Option' + '\nConstrained Technology Deployment' + '\nNo Lead Times'
            elif model_name == 'euler_no_lead':
                fig_title = 'Total System Costs above Cheapest Option' + '\nConstrained Technology Deployment' + '\nNo Lead Times'
            else:
                print("dataset not defined correctly")
                raise ValueError("Invalid dataset configuration: {}".format(model_name))
    elif [plot_name, feature] == ["stair_plot", "capacity_addition"]:
        tech_set = set(technology)
        renewable_gen_set = set(["nuclear", "wind_offshore", "wind_onshore",
                                    "photovoltaics", "run-of-river_hydro", "reservoir_hydro",
                                    "biomass_plant", "biomass_plant_CCS"])
        if tech_set == renewable_gen_set:
            tech = "Renewable El. Generation Capacity"
        else:
            tech = ", ".join(technology)
        if len(model_list) == 1:
            if model_name == 'PI_1701_1to6':
                fig_title = 'Sum of Total Capacity Built by 2050 of:\n' + tech + '\nUnconstrained Technology Deployment' + '\nNo Lead Times'
            elif model_name == 'PC_1701_1to6':
                fig_title = 'Sum of Total Capacity Built by 2050 of:\n' + tech + '\nConstrained Technology Deployment' + '\nNo Lead Times'
            elif model_name == 'euler_no_lead':
                fig_title = 'Sum of Total Capacity Built by 2050 of:\n' + tech + '\nConstrained Technology Deployment' + '\nNo Lead Times'
            else:
                print("dataset not defined correctly")
                raise ValueError("Invalid dataset configuration: {}".format(model_name))
    elif plot_name == "feedback_plot":
        if model_name == 'PI_1701_1to6':
            fig_title = 'Cumulative Carbon Emissions:\nProjected Long-Term Investment Planning and Actual Resulting Emissions due to Short Term Operation\nUnconstrained Technology Deployment; No Lead Times'
        elif model_name == 'PC_1701_1to6':
            fig_title = 'Cumulative Carbon Emissions:\nProjected Long-Term Investment Planning and Actual Resulting Emissions due to Short Term Operation\nConstrained Technology Deployment; No Lead Times'
        elif model_name == 'euler_no_lead':
            fig_title = 'Cumulative Carbon Emissions:\nProjected Long-Term Investment Planning and Actual Resulting Emissions due to Short Term Operation\nConstrained Technology Deployment; No Lead Times'
        else:
            print("dataset not defined correctly")
            raise ValueError("Invalid dataset configuration: {}".format(model_name))
    elif plot_name == "capacity_addition":
        if model_name == 'PI_1701_1to6':
            fig_title = formatted_tech + ' Capacity Addition' + '\nUnconstrained Technology Deployment' + '\nNo Lead Times'
        elif model_name == 'PC_1701_1to6':
            fig_title = formatted_tech + ' Capacity Addition' + '\nConstrained Technology Deployment' + '\nNo Lead Times'
        elif model_name == 'euler_no_lead':
            fig_title = formatted_tech + ' Capacity Addition' + '\nConstrained Technology Deployment' + '\nNo Lead Times'
        else:
            print("dataset not defined correctly")
            raise ValueError("Invalid dataset configuration: {}".format(model_name))
    elif plot_name == "cumulative_emissions":
        if model_name == 'PI_1701_1to6':
            fig_title = 'Cumulative Carbon Emissions: \nUnconstrained Technology Deployment' + '\nNo Lead Times'
        elif model_name == 'PC_1701_1to6':
            fig_title = 'Cumulative Carbon Emissions: \nConstrained Technology Deployment' + '\nNo Lead Times'
        elif model_name == 'euler_no_lead':
            fig_title = 'Cumulative Carbon Emissions: \nConstrained Technology Deployment' + '\nNo Lead Times'
        elif model_name == 'cons_lead_15':
            fig_title = 'Cumulative Carbon Emissions: \nConstrained Technology Deployment' + '\nLead Times'
        else:
            print("dataset not defined correctly")
            raise ValueError("Invalid dataset configuration: {}".format(model_name))
    elif plot_name == "costs_over_time":
        # Initialize fig_title with a base string
        fig_title_start = ""
        fig_title_end = " over Time"
        # Check the presence of each cost type and create a list of descriptions for those present
        costs_descriptions = []  # "cost_opex_total", "cost_capex_total" , "cost_carrier_total"
        if "cost_opex_total" in costs and "cost_capex_total" in costs and "cost_carrier_total" in costs:
            costs_descriptions = ["Total System Costs (CAPEX, OPEX and Fuel)"]
        else:
            fig_title_start = "Sum of "
            if "cost_opex_total" in costs:
                costs_descriptions.append("Total OPEX")
            if "cost_capex_total" in costs:
                costs_descriptions.append("Total CAPEX")
            if "cost_carrier_total" in costs:
                costs_descriptions.append("Total Fuel Costs")
        # Join the descriptions with commas and add to the base fig_title
        if costs_descriptions:  # Only proceed if there are any descriptions to add
            fig_title = fig_title_start + " and ".join(costs_descriptions) + fig_title_end
        else:
            assert False, "no costs specified"
    elif plot_name == "full_ts_conv_tech":
        if model_name == 'PI_1701_1to6':
            if granularity == "y":
                fig_title = 'Yearly ' + formatted_tech + ' Operation \nUnconstrained Technology Deployment \nNo Lead Times'
            else:
                fig_title = formatted_tech + ' Operation Year ' + year_str + '\nUnconstrained Technology Deployment \nNo Lead Times'
        elif model_name == 'PC_1701_1to6':
            if granularity == "y":
                fig_title = 'Yearly ' + formatted_tech + ' Operation \nConstrained Technology Deployment \nNo Lead Times'
            else:
                fig_title = formatted_tech + ' Operation Year ' + year_str + '\nConstrained Technology Deployment \nNo Lead Times'
        elif model_name == 'euler_no_lead':
            if granularity == "y":
                fig_title = 'Yearly ' + formatted_tech + ' Operation \nConstrained Technology Deployment \nNo Lead Times'
            else:
                fig_title = formatted_tech + ' Operation Year ' + year_str + '\nConstrained Technology Deployment \nNo Lead Times'
        else:
            print("dataset not defined correctly")
            raise ValueError("Invalid dataset configuration: {}".format(model_name))
    fig_title = "Title not yet defined (Total system costs, mic between lead and no lead, constr. tech. deployment)"
    assert fig_title is not None, "title was not defined"
    return fig_title

# plotting functions
def full_ts_conv_tech(r, technology, granularity, year):
    plot_name = "full_ts_conv_tech"
    print("full_ts_conv_tech initiated")
    completereset = r.get_full_ts("flow_conversion_input")
    print("load complete")
    # completereset = complete.copy(deep=True)

    # Reset the index to work with it as columns
    completereset.reset_index(inplace=True)
    print("reset complete")
    # The 'level_0' column contains the scenario information, so we will rename it
    completereset.rename(columns={'level_0': 'scenario'}, inplace=True)
    print("rename complete")
    # Now we set the multi-index again, this time including the 'scenario'
    completereset.set_index(['scenario', 'technology', 'carrier', 'node'], inplace=True)
    print("re-index complete")
    # Perform the aggregation by summing over the 'node' level of the index
    aggregated_data = completereset.groupby(level=['scenario', 'technology', 'carrier']).sum()
    print("aggregation complete")
    # Filter the DataFrame to only keep rows where the technology is 'oil_boiler'
    tech_data = aggregated_data.xs(technology, level='technology')
    print("technology separated out")
    # Drop the 'carrier' level from the multi-index
    tech_data = tech_data.reset_index('carrier', drop=True)

    df = tech_data
    # indexing
    df = indexmapping(df)
    # Select only the desired rows
    if desired_scenarios is not None:
        df = df.loc[desired_scenarios]

    # Transpose DataFrame to plot each row as a series
    df = df.T

    if granularity == "y":
        # Calculate group for each set of 8760 rows
        df['group'] = df.index // 8760
        # Sum values within each group
        df = df.groupby('group').sum().reset_index()
        df = df.drop(columns=['group'])
    else:
        # Second year
        df = df.iloc[(8760*year):(8760*(year+1))]

    formatted_tech = reformat_prompt(technology)
    year_str = str(year*2)

    # generate figure title:
    fig_title = get_fig_title(plot_name=plot_name,
                              formatted_tech=formatted_tech,
                              granularity=granularity, year_str=year_str)

    # Plot the selected series
    ax = df.plot(kind='line', marker='', figsize=(10, 6), linewidth=1)
    plt.xlabel('Year')
    plt.ylabel("Conversion GWh?? ")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # Add faint grid
    plt.legend(loc='best')
    plt.title(fig_title)

    # Modify x-axis tick labels
    new_labels = [2022 + 2 * (int(label)) for label in ax.get_xticks()]
    ax.set_xticklabels(new_labels)

    # Show the plot
    plt.show(block=True)

def plot_costs_over_time(costs):
    plot_name = "costs_over_time"
    print("Plotting costs over time")
    df = {}
    for c in costs:
        df[c] = r.get_total(c)
    df = pd.concat(df,keys=df.keys())
    df = df.groupby(level=1).sum()

    # indexing
    df = indexmapping(df)
    # Select only the desired rows
    if desired_scenarios is not None:
        df = df.loc[desired_scenarios]

    # Transpose DataFrame to plot each row as a series
    df = df.T

    # generate figure title:
    # fig_title = get_fig_title(plot_name=plot_name, costs=costs)

    # Plot the selected series
    ax = df.plot(kind='line', marker='o', figsize=(10, 6))

    # plt.title(fig_title)
    plt.xlabel('Year')
    # plt.ylabel(formatted_costs)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # Add faint grid
    plt.legend(loc='best')

    # Modify x-axis tick labels
    new_labels = [2022 + 2 * (int(label)) for label in ax.get_xticks()]
    ax.set_xticklabels(new_labels)

    # Show the plot
    plt.show(block=True)

    return

def cumulative_emissions_plot():
    plot_name = "cumulative_emissions"
    print("Plotting Cumulative Emissions")
    # Set the color palette from Seaborn
    sns.set_palette(
        'deep')  # 'deep' is the default palette in Seaborn, but you can choose others like 'muted', 'bright', etc.
    df = r.get_total("carbon_emissions_cumulative")

    # indexing
    df = indexmapping(df)
    # Select only the desired rows
    if desired_scenarios is not None:
        df = df.loc[desired_scenarios]
    df_T = df.T

    fig, ax = plt.subplots(figsize=(12, 8))
    carbon_budget = 10.076
    # Number of scenarios and years
    num_scenarios = len(df_T.columns)
    num_years = len(df_T)
    # Creating an array for the x-axis positions
    x = np.arange(num_years)
    for i, scenario in enumerate(df_T.columns):
        plt.plot(x, df_T[scenario] / 1000, label=df_T.columns[i])
    # Adding labels and title
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Carbon Emissions [Gt CO₂]', fontsize=14)

    # generate figure title:
    #fig_title = get_fig_title(plot_name)
    # plt.title(fig_title)

    # Adjusting x-ticks
    plt.xticks(x, (df_T.index) * 2 + 2022)
    # Adding a legend
    plt.legend(title="Operation Foresight Horizon:")
    # Adding a thin red horizontal line at y = carbon_emissions_budget
    plt.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0.25, zorder=1, alpha=0.5)
    # Labeling the red line
    plt.text(14, carbon_budget - 0.04 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='right', color='black',
             fontsize=12)
    # Show the plot
    plt.show(block=True)
    return

def capacity_addition_plot(technology=None, capacity_type=None):
    plot_name = "capacity_addition"
    print("plotting yearly capacity addition")
    df = r.get_total("capacity_addition")

    # Assuming df is your DataFrame
    # Group by the first two levels of the multi-index ('scenario' and 'technology'), then sum

    # Reset the index to work with it as columns
    df.reset_index(inplace=True)
    print("reset complete")
    # The 'level_0' column contains the scenario information, so we will rename it
    df.rename(columns={'level_0': 'scenario'}, inplace=True)

    # Now we set the multi-index again, this time including the 'scenario'
    df.set_index(['scenario', 'technology', 'capacity_type', 'location'], inplace=True)
    print("re-index complete")
    # Perform the aggregation by summing over the 'node' level of the index
    df = df.groupby(level=['scenario', 'technology', 'capacity_type']).sum()
    print("aggregation complete")

    #indexing
    df = indexmapping(df)
    # Select only the desired rows
    if desired_scenarios is not None:
        df = df.loc[desired_scenarios]

    df = df.loc[(slice(None), technology, capacity_type), :]

    # Transpose DataFrame to plot each row as a series
    df = df.T

    # Plot the selected series
    ax = df.plot(kind='line', marker='o', figsize=(10, 6))

    formatted_tech = reformat_prompt(technology)

    # generate figure title:
    # fig_title = get_fig_title(plot_name,formatted_tech=formatted_tech)

    # plt.title(fig_title)

    plt.xlabel('Year')
    plt.ylabel('Capacity Addition ' + formatted_tech + ' GW??')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # Add faint grid
    plt.legend(loc='best')

    # Modify x-axis tick labels
    new_labels = [2022 + 2 * (int(label)) for label in ax.get_xticks()]
    ax.set_xticklabels(new_labels)

    # Show the plot
    plt.show(block=True)
    return

def feedback_plot():
    plot_name = "feedback_plot"
    print("plotting projected and actual cumulative carbon emissions")

    df = r.get_total("carbon_emissions_cumulative", keep_raw=True)

    # indexing
    df = indexmapping(df)
    # Select only the desired rows
    if desired_scenarios is not None:
        df = df.loc[desired_scenarios]

    assert len(df.columns) == 15, "Number of timesteps not equal to 15"
    # Order for 'actual' and '_inv' rows separately as you requested
    order_op = [str(i) for i in range(15)]  # '0' to '14'
    order_inv = [f"{i}_inv" for i in range(15)]  # '0_inv' to '14_inv'

    # Filter rows where the 'mf' level of the index ends with '_inv'
    inv_rows = df[df.index.get_level_values('mf').str.endswith('_inv')]
    inv_rows = inv_rows.reindex(order_inv, level=1)
    # inv_rows = inv_rows.loc[(slice(None), ['1_inv', '2_inv', '3_inv', '4_inv', '5_inv']), :]

    # Filter rows where the 'mf' level of the index does not end with '_inv'
    op_rows = df[~df.index.get_level_values('mf').str.endswith('_inv')]
    op_rows = op_rows.reindex(order_op, level=1)

    # Identifying the first non-NaN value in each 'mf' row
    actual_rows = op_rows.apply(
        lambda row: row.dropna().iloc[0] if not row.dropna().empty else np.nan, axis=1)
    actual_rows = actual_rows.values.flatten()

    # generate figure title:
    # fig_title = get_fig_title(plot_name=plot_name)
    fig_title = "feedback plot cons_lead_15"
    # plot figure:
    plt.figure(figsize=(10, 6))

    # Iterate through each row of inv_rows
    for row in inv_rows.index:
        plt.plot(inv_rows.columns, (inv_rows.loc[row, :])/ 1000, label=row[1])

    # Assuming 'actual_rows' needs to be plotted in a specific way as per your structure
    # Plotting 'actual_rows' directly if it's structured for direct plotting
    plt.plot(actual_rows/ 1000, label='Actual Rows', color='black', marker='o', markersize=4, markerfacecolor='black')

    # Creating an axis object 'ax' for further customization
    ax = plt.gca()

    # Legend
    inv_horizon, op_horizon = desired_scenarios[0].split('_')
    # Extract the second level of the MultiIndex, which corresponds to level 1 (since it's zero-indexed)
    second_level_values = inv_rows.index.get_level_values(1)
    # Perform the desired operation on these values
    new_labels = [str(2022 + int(s.split('_')[0]) * 2) for s in second_level_values]
    new_labels.append("Actual Emissions")
    plt.legend(new_labels, title='Foresight Horizon\nInvestment: '+str(int(inv_horizon)*2)+'a'+'\nOperation: '+str(int(op_horizon)*2)+'a\n\nProj. Emissions by\nInvestment Plan in:')

    # Modify x-axis tick labels
    # This assumes you want the labels to reflect a calculation based on the original x-axis labels.
    # Ensure ax.get_xticks() returns the expected values before applying the calculation.
    new_labels = [2022 + 2 * int(label) for label in ax.get_xticks()]
    ax.set_xticklabels(new_labels)

    carbon_budget = 10.076
    # Adding a thin red horizontal line at y = carbon_emissions_budget
    plt.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0.25, zorder=1, alpha=0.5)
    # Labeling the red line
    plt.text(14, carbon_budget - 0.04 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='right', color='black',
             fontsize=12)

    plt.title(fig_title)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Carbon Emissions [Gt CO₂]', fontsize=14)
    plt.show(block=True)

    return

def stair_plot(feature, model_list, r, technology=None, capacity_type=None):
    plot_name = "stair_plot"
    print("setting up stairs plot")
    global n
    # Number of rows and columns of stair plot
    n = 15
    carbon_budget = 10076
    skip_load = False
    if type(model_list) is not list:
        model_list = [model_list]
        skip_load = True

    # Initialize an empty list to store DataFrames from each model
    dfs = []
    # Iterate over each model name, fetching and processing its results
    for name in model_list:
        if skip_load is not True:
            r, config = read_in_results(name)

        # plot setup
        if feature == "costs":
            temp_df = {}
            for c in costs:
                temp_df[c] = r.get_total(c)
            temp_df = pd.concat(temp_df,keys=temp_df.keys())
            temp_df = temp_df.groupby(level=1).sum()
            temp_df = temp_df.sum(axis=1)
            # indexing
            temp_df = indexmapping(temp_df, special_model_name=name)
            # Append the processed DataFrame to the list
            dfs.append(temp_df)
            if "cost_opex_total" in costs and "cost_capex_total" in costs and "cost_carrier_total" in costs:
                cbar_label = "Total System Costs (CAPEX, OPEX and Fuel) [% more expensive]"
            else:
                cbar_start = "Sum of "
                costs_descriptions = []
                if "cost_opex_total" in costs:
                    costs_descriptions.append("Total OPEX")
                if "cost_capex_total" in costs:
                    costs_descriptions.append("Total CAPEX")
                if "cost_carrier_total" in costs:
                    costs_descriptions.append("Total Fuel Costs")
                # Join the descriptions with commas and add to the base fig_title
                cbar_label = cbar_start + " and ".join(costs_descriptions) + ' [% more expensive]'

        elif feature == "cumulative_emissions":
            temp_df = r.get_total("carbon_emissions_cumulative")
            temp_df = temp_df.iloc[:,-1]
            # indexing
            temp_df = indexmapping(temp_df, special_model_name=name)
            dfs.append(temp_df)
            cbar_label = 'Total Cumulative Eimssions [% over carbon budget]'
        elif feature == "capacity_addition":
            temp_df = r.get_total("capacity_addition")

            # Reset the index to work with it as columns
            temp_df.reset_index(inplace=True)
            print("reset complete")

            # The 'level_0' column contains the scenario information, so we will rename it
            temp_df.rename(columns={'level_0': 'scenario'}, inplace=True)

            # Now we set the multi-index again, this time including the 'scenario'
            temp_df.set_index(['scenario', 'technology', 'capacity_type', 'location'], inplace=True)
            print("re-index complete")

            # Perform the aggregation by summing over the 'node' level of the index
            temp_df = temp_df.groupby(level=['scenario', 'technology', 'capacity_type']).sum()
            print("aggregation complete")

            # indexing
            temp_df = indexmapping(temp_df, special_model_name=name)

            # sum over the years
            temp_df = temp_df.sum(axis=1)
            #get rid of energy part of df
            if capacity_type != "energy":
                temp_df = temp_df.loc[temp_df.index.get_level_values('capacity_type') != 'energy']

            temp_df = temp_df.loc[temp_df.index.get_level_values('technology').isin(technology)]

            # asserting that all capacity types are the same
            unique_capacity_types = temp_df.index.get_level_values('capacity_type').unique()
            assert len(unique_capacity_types) == 1, "Not all entries in 'capacity_type' are the same."

            # Group by 'scenario', then sum
            temp_df = temp_df.groupby(level=['scenario']).sum()

            # Append the processed DataFrame to the list
            dfs.append(temp_df)
            cbar_label = '[% of additional capacity installed]'
        else:
            assert False, "feature not available"

    # Concatenate all DataFrames in the list to a single DataFrame
    df = pd.concat(dfs, axis=0, join='outer')

    # get the threshold matrix (combinations of foresight horizons that don't stay within the carbon budget)
    threshold_matrix = emissions_threshold_matrix_creation(model_list)

    # min and max values
    min_value = df.min()
    max_value = df.max()
    if feature == "cumulative_emissions":
        min_value = carbon_budget
        pass

    # Normalize the dataset values to a range of 0 to 1 for color mapping
    normalized_df = (df - min_value) / (max_value - min_value)

    # get fig title
    fig_title = get_fig_title(plot_name=plot_name, model_list=model_list, feature=feature, technology=technology)

    # plot the data:
    plot_stairs(normalized_df, threshold_matrix, max_value, min_value, fig_title, cbar_label)

    return


if __name__ == "__main__":
    desired_scenarios = None


    # Select the model:
    # model_name = "cons_lead_5to7"  # 'PI_1701_1to6', 'PC_1701_1to6', "euler_no_lead", "cons_lead_1to7", "cons_lead_15"

    # Select only the desired scenarios:
    # desired_scenarios = ['15_15', '15_14', '15_13', '15_11', '15_10', '15_9', '15_8', '15_7', '15_6', '15_5', '15_4', '15_3', '15_2', '15_1']
    # desired_scenarios = ['15_15', '15_14', '15_13', '15_11', '15_10', '15_9', '15_8', '15_7']
    # desired_scenarios = ['15_15', '15_14', '15_13', '15_2', '15_1']
    # desired_scenarios = ['15_15', '15_14', '15_13', '15_11', '15_10', '15_9','15_8', '15_7', '15_6']
    # desired_scenarios = ['7_7','7_6', '7_2', '7_1', '6_6', '6_2', '6_1', '3_3', '3_2', '3_1', '1_1']
    desired_scenarios = ['7_7', '7_6', '7_5', '7_4', '7_3', '7_2', '7_1']
    # desired_scenarios = ['6_6', '6_5', '6_2', '6_1']
    # desired_scenarios = ['6_6', '6_2', '6_1', '3_3', '3_2', '3_1']
    # desired_scenarios = ['6_6', '6_5', '6_4', '6_3', '6_2', '6_1']
    # desired_scenarios = ['6_2', '5_2', '4_2', '3_2', '2_2']
    # desired_scenarios = ['6_1', '5_1', '4_1', '3_1', '2_1', '1_1']
    # desired_scenarios = ['6_3', '5_3', '4_3', '3_3']
    # desired_scenarios = ['6_4', '5_4', '4_4']
    # desired_scenarios = ['6_4', '6_3', '6_2', '6_1']
    # desired_scenarios = ['5_5', '5_4', '5_3', '5_2', '5_1']
    # desired_scenarios = ['4_4', '4_3', '4_2', '4_1']
    # desired_scenarios = ['2_2', '2_1']
    # desired_scenarios = ['2_2', '2_1']
    # desired_scenarios = ['1_1']
    # desired_scenarios = ['2_2', '2_1', '2_2', '2_1']
    # desired_scenarios = ['7_5']

    model_name = "lesscons_lead_init"
    r, config = read_in_results(model_name)
    print("Results read in")

    if True:
        order_op = [str(i) for i in range(15)]

        net_present_costs = r.get_total("net_present_cost", keep_raw=True)
        net_present_costs = indexmapping(net_present_costs)
        net_present_costs = net_present_costs.loc[desired_scenarios]
        net_present_costs = net_present_costs.reindex(order_op, level=1)

        net_present_costs = net_present_costs.reset_index()
        net_present_costs_filled = net_present_costs.T.ffill(axis=1).T

        net_present_costs = net_present_costs_filled.drop(columns=["mf", "level_0"])
        cumulative_sum_net_present_costs = net_present_costs.cumsum(axis=1, skipna=True)
        net_present_costs = net_present_costs.astype('float64')
        cumulative_sum_net_present_costs = cumulative_sum_net_present_costs.astype('float64')

        emissions_costs = r.get_total("cost_carbon_emissions_total", keep_raw=True)
        emissions_costs = indexmapping(emissions_costs)
        emissions_costs = emissions_costs.loc[desired_scenarios]
        emissions_costs = emissions_costs.reindex(order_op, level=1)
        emissions_costs = emissions_costs.reset_index()
        emissions_costs = emissions_costs.T.ffill(axis=1).T
        emissions_costs = emissions_costs.drop(columns=["mf", "level_0"])
        cumulative_sum_emissions_costs = emissions_costs.cumsum(axis=1, skipna=True)
        emissions_costs = emissions_costs.astype('float64')
        cumulative_sum_emissions_costs = cumulative_sum_emissions_costs.astype('float64')

        capex = r.get_total("cost_capex_total", keep_raw=True)
        capex = indexmapping(capex)
        capex = capex.loc[desired_scenarios]
        capex = capex.reindex(order_op, level=1)
        capex = capex.reset_index()
        capex_filled = capex.T.ffill(axis=1).T
        capex = capex_filled.drop(columns=["mf", "level_0"])
        cumulative_sum_capex = capex.cumsum(axis=1, skipna=True)
        capex = capex.astype('float64')
        cumulative_sum_capex = cumulative_sum_capex.astype('float64')

        opex = r.get_total("cost_opex_total", keep_raw=True) + r.get_total("cost_carrier_total", keep_raw=True)
        opex = indexmapping(opex)
        opex = opex.loc[desired_scenarios]
        opex = opex.reindex(order_op, level=1)
        opex = opex.reset_index()
        opex_filled = opex.T.ffill(axis=1).T
        opex = opex_filled.drop(columns=["mf", "level_0"])
        cumulative_sum_opex = opex.cumsum(axis=1, skipna=True)
        opex = opex.astype('float64')
        cumulative_sum_opex = cumulative_sum_opex.astype('float64')

        cost_total = r.get_total("cost_total", keep_raw=True)
        cost_total = indexmapping(cost_total)
        cost_total = cost_total.loc[desired_scenarios]
        cost_total = cost_total.reindex(order_op, level=1)
        cost_total = cost_total.reset_index()
        cost_total = cost_total.T.ffill(axis=1).T
        cost_total = cost_total.drop(columns=["mf", "level_0"])
        cumulative_cost_total = cost_total.cumsum(axis=1, skipna=True)
        cost_total = cost_total.astype('float64')
        cumulative_cost_total = cumulative_cost_total.astype('float64')

        cumulative_emissions = r.get_total("carbon_emissions_cumulative", keep_raw=True)
        cumulative_emissions = indexmapping(cumulative_emissions)
        cumulative_emissions = cumulative_emissions.loc[desired_scenarios]
        cumulative_emissions = cumulative_emissions.reindex(order_op, level=1)
        cumulative_emissions = cumulative_emissions.reset_index()
        cumulative_emissions = cumulative_emissions.T.ffill(axis=1).T
        cumulative_emissions.drop(columns=['mf', "level_0"], inplace=True)
        cumulative_emissions = cumulative_emissions.astype('float64')

        annual_emissions = r.get_total("carbon_emissions_annual", keep_raw=True)
        annual_emissions = indexmapping(annual_emissions)
        annual_emissions = annual_emissions.loc[desired_scenarios]
        annual_emissions = annual_emissions.reindex(order_op, level=1)
        annual_emissions = annual_emissions.reset_index()
        annual_emissions = annual_emissions.T.ffill(axis=1).T
        annual_emissions.drop(columns=['mf', "level_0"], inplace=True)
        annual_emissions = annual_emissions.astype('float64')

        emissions_carrier = r.get_total("carbon_emissions_carrier", keep_raw=True)
        emissions_carrier = indexmapping(emissions_carrier)
        emissions_carrier = emissions_carrier.loc[desired_scenarios]
        emissions_carrier = emissions_carrier.reset_index()
        emissions_carrier['mf'] = emissions_carrier['mf'].astype(int)
        # Sort within each 'Technology' and 'Node' group by '#
        emissions_carrier = emissions_carrier.sort_values(by=['carrier', 'node','mf'])
        emissions_carrier = emissions_carrier.groupby(['carrier', 'mf']).sum().reset_index()
        emissions_carrier = emissions_carrier.drop(columns='node')

        natural_gas_emissions = emissions_carrier[emissions_carrier['carrier'] == 'natural_gas']
        natural_gas_emissions.drop(columns=['mf', 'carrier', "level_0"], inplace=True)
        transposed_df = natural_gas_emissions.T
        transposed_df = transposed_df.mask(transposed_df == 0.0)
        transposed_df = transposed_df.ffill(axis=1)
        cumulative_natural_gas_emissions = transposed_df.T
        cumulative_natural_gas_emissions = cumulative_natural_gas_emissions.cumsum(axis=1)
        cumulative_natural_gas_emissions = cumulative_natural_gas_emissions.astype('float64')

        optimiser_perspective = net_present_costs.copy(deep=True)
        # Performing the transformation
        for i in range(len(optimiser_perspective)):
            # Sum all terms to the right of the diagonal, including the diagonal term
            optimiser_perspective.iloc[i, i:] = np.flip(np.cumsum(np.flip(optimiser_perspective.iloc[i, i:])))

        df = r.get_total("capacity_addition", keep_raw=True)
        df = indexmapping(df)
        df = df.loc[desired_scenarios]
        df.droplevel(level=0)
        df = df.set_index('mf', append=True)
        capacity_addition_df = df.groupby(level=['technology', 'capacity_type', 'mf']).sum()

        renewable_el_generation = ["nuclear", "wind_offshore", "wind_onshore",
                                   "photovoltaics", "run-of-river_hydro", "reservoir_hydro",
                                   "biomass_plant", "biomass_plant_CCS"]
        technologies = renewable_el_generation
        df_filtered = capacity_addition_df.loc[capacity_addition_df.index.get_level_values('technology').isin(renewable_el_generation)]
        df_filtered = df_filtered.groupby(level='mf').sum()
        df_filtered = df_filtered.reindex(order_op, level=1)

        df = r.get_total("capacity_existing", keep_raw=True)
        df = indexmapping(df)
        df = df.loc[desired_scenarios]
        df = df.reset_index(level='scenario', drop=True)
        df = df.set_index('mf', append=True)
        capacity_addition_df = df.groupby(level=['technology', 'capacity_type', 'mf']).sum()

        renewable_el_generation = ["nuclear", "wind_offshore", "wind_onshore",
                                   "photovoltaics", "run-of-river_hydro", "reservoir_hydro",
                                   "biomass_plant", "biomass_plant_CCS"]
        technologies = renewable_el_generation
        df_filtered = capacity_addition_df.loc[
            capacity_addition_df.index.get_level_values('technology').isin(renewable_el_generation)]
        df_filtered = df_filtered.groupby(level='mf').sum()
        df_filtered = df_filtered.reindex(order_op, level=1)

    print("done")
    # plot full time series of a technology
    if False:
        granularity = "y"  # "y",
        year = 10  # (0-14) (info only relevant for hourly resolution plots)
        technology = "natural_gas_boiler"  # "heat_pump", "natural_gas_boiler"
        full_ts_conv_tech(r, technology, granularity, year)

    # plot opex and capex
    if False:
        costs = ["cost_opex_total", "cost_capex_total" , "cost_carrier_total"]  # or "cost_opex_total", "cost_capex_total" , "cost_carrier_total"
        plot_costs_over_time(costs)

    # plot cumulative emissions
    if False:
        cumulative_emissions_plot()

    # plot capacity addition over time
    if False:
        technology = "heat_pump"  # or None
        capacity_type = 'power'  # or energy
        capacity_addition_plot(technology, capacity_type)

    # investment projections plot:
    if False:
        assert len(desired_scenarios) == 1, "too many scenarios for this plot"
        feedback_plot()

    # stairs plot:
    if True:
        # define parameters
        features = ["costs"]  # "costs", "cumulative_emissions", "capacity_addition"
        costs = ["cost_opex_total", "cost_capex_total", "cost_carrier_total"]  # or "cost_opex_total", "cost_capex_total" , "cost_carrier_total"

        # tech combinations
        technologies = ["natural_gas_boiler", "oil_boiler"] # "nuclear", "onshore_wind",            specify technologies
        renewable_el_generation =  ["nuclear", "wind_offshore", "wind_onshore",
                                    "photovoltaics", "run-of-river_hydro", "reservoir_hydro",
                                    "biomass_plant", "biomass_plant_CCS"]

        technologies = renewable_el_generation
        capacity_type = "power"  # "power", "energy"

        # multi model setup
        multi_model = True
        if multi_model is not True:
            pass
        else:
            model_name = ["cons_lead_1to4", "cons_lead_5to7"]  # ,  "some other name", "cons_lead_15"

        # generate the stairs plot
        for feature in features:
            stair_plot(feature, model_name, r, technology=technologies, capacity_type=capacity_type)
