import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors  # this is what I recently added
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import importlib
import os
import json
import pandas as pd
import math
from pandas import IndexSlice as idx
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from postresults import PostResult

from zen_garden.postprocess.results.results import Results
from zen_garden.model.default_config import Config,Analysis,Solver,System

# read in results
def read_in_results(model_name):
    # data_path = "/Users/lukashegner/PycharmProjects/ZEN-garden/data/"
    # outputs_path = data_path + "outputs"
    data_path = "/Volumes/Elements10T/RRE_Myopic_Operation_Project/Model_Outputs/"
    outputs_path = data_path + "/outputs"
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

# read in results
def read_in_special_results(model_name):
    data_path = "/Users/lukashegner/PycharmProjects/ZEN-garden_fork/data/"
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
        r = PostResult(out_folder)
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
    elif model_name == 'PC_ct_testing':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '15_15',
            'scenario_1': '15_1'
        }
    elif model_name == 'lesscons_lead_init' \
            or model_name == 'cons_lead_init'\
            or model_name == 'varcons_lead_init'\
            or model_name == 'cons_nolead_init'\
            or model_name == 'post_changes_to_model/lesscons_lead_init'\
            or model_name == 'post_changes_to_model/varcons_lead_init'\
            or model_name == 'post_changes_to_model/cons_lead_init':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '15_1',
            'scenario_1': '15_15',
            'scenario_2': '6_1',
            'scenario_3': '6_6'
        }
    elif model_name == 'lesscons_lead_init_2' \
            or model_name == 'cons_lead_init_2'\
            or model_name == 'varcons_lead_init_2'\
            or model_name == 'cons_nolead_init_2' \
            or model_name == 'post_changes_to_model/lesscons_lead_init_2' \
            or model_name == 'post_changes_to_model/varcons_lead_init_2' \
            or model_name == 'post_changes_to_model/cons_lead_init_2':
        # Define a dictionary mapping the current index values to the desired ones
        index_mapping = {
            'scenario_': '7_1',
            'scenario_1': '7_7'
        }
    elif model_name == 'cons_lead_1to7' or model_name == 'cons_nolead_1to7':
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
    elif model_name == 'cons_lead_1to4' or model_name == 'cons_nolead_1to4' \
            or model_name == 'lesscons_lead_1to4' or model_name == 'varcons_lead_1to4'\
            or model_name == 'post_changes_to_model/cons_lead_1to4'\
            or model_name == 'post_changes_to_model/lesscons_lead_1to4'\
            or model_name == 'post_changes_to_model/varcons_lead_1to4':
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
    elif model_name == 'cons_lead_5to7' or model_name == 'cons_nolead_5to7' \
            or model_name == 'lesscons_lead_5to7' or model_name == 'varcons_lead_5to7'\
            or model_name == 'post_changes_to_model/cons_lead_5to7':
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
            'scenario_17': '5_5'
        }
    elif model_name == 'cons_lead_8to10' or model_name == 'cons_nolead_8to10' \
            or model_name == 'lesscons_lead_8to10' or model_name == 'varcons_lead_8to10'\
            or model_name == 'post_changes_to_model/cons_lead_8to10':
        index_mapping = {
            'scenario_': '10_1',
            'scenario_1': '10_2',
            'scenario_2': '10_3',
            'scenario_3': '10_4',
            'scenario_4': '10_5',
            'scenario_5': '10_6',
            'scenario_6': '10_7',
            'scenario_7': '10_8',
            'scenario_8': '10_9',
            'scenario_9': '10_10',
            'scenario_10': '9_1',
            'scenario_11': '9_2',
            'scenario_12': '9_3',
            'scenario_13': '9_4',
            'scenario_14': '9_5',
            'scenario_15': '9_6',
            'scenario_16': '9_7',
            'scenario_17': '9_8',
            'scenario_18': '9_9',
            'scenario_19': '8_1',
            'scenario_20': '8_2',
            'scenario_21': '8_3',
            'scenario_22': '8_4',
            'scenario_23': '8_5',
            'scenario_24': '8_6',
            'scenario_25': '8_7',
            'scenario_26': '8_8'
        }
    elif model_name == 'cons_lead_11' or model_name == 'cons_nolead_11' \
            or model_name == 'lesscons_lead_11' or model_name == 'varcons_lead_11'\
            or model_name == 'post_changes_to_model/cons_lead_11':
        index_mapping = {
            'scenario_': '11_1',
            'scenario_1': '11_2',
            'scenario_2': '11_3',
            'scenario_3': '11_4',
            'scenario_4': '11_5',
            'scenario_5': '11_6',
            'scenario_6': '11_7',
            'scenario_7': '11_8',
            'scenario_8': '11_9',
            'scenario_9': '11_10',
            'scenario_10': '11_11'
        }
    elif model_name == 'cons_lead_12' or model_name == 'cons_nolead_12' \
            or model_name == 'lesscons_lead_12' or model_name == 'varcons_lead_12'\
            or model_name == 'post_changes_to_model/cons_lead_12':
        index_mapping = {
            'scenario_': '12_1',
            'scenario_1': '12_2',
            'scenario_2': '12_3',
            'scenario_3': '12_4',
            'scenario_4': '12_5',
            'scenario_5': '12_6',
            'scenario_6': '12_7',
            'scenario_7': '12_8',
            'scenario_8': '12_9',
            'scenario_9': '12_10',
            'scenario_10': '12_11',
            'scenario_11': '12_12'
        }
    elif model_name == 'cons_lead_13' or model_name == 'cons_nolead_13' \
            or model_name == 'lesscons_lead_13' or model_name == 'varcons_lead_13'\
            or model_name == 'post_changes_to_model/cons_lead_13':
        index_mapping = {
            'scenario_': '13_1',
            'scenario_1': '13_2',
            'scenario_2': '13_3',
            'scenario_3': '13_4',
            'scenario_4': '13_5',
            'scenario_5': '13_6',
            'scenario_6': '13_7',
            'scenario_7': '13_8',
            'scenario_8': '13_9',
            'scenario_9': '13_10',
            'scenario_10': '13_11',
            'scenario_11': '13_12',
            'scenario_12': '13_13',
        }
    elif model_name == 'cons_lead_14' or model_name == 'cons_nolead_14' \
            or model_name == 'lesscons_lead_14' or model_name == 'varcons_lead_14'\
            or model_name == 'post_changes_to_model/cons_lead_14':
        index_mapping = {
            'scenario_': '14_1',
            'scenario_1': '14_2',
            'scenario_2': '14_3',
            'scenario_3': '14_4',
            'scenario_4': '14_5',
            'scenario_5': '14_6',
            'scenario_6': '14_7',
            'scenario_7': '14_8',
            'scenario_8': '14_9',
            'scenario_9': '14_10',
            'scenario_10': '14_11',
            'scenario_11': '14_12',
            'scenario_12': '14_13',
            'scenario_13': '14_14'
        }
    elif model_name == 'cons_lead_15' or model_name == 'cons_nolead_15' \
            or model_name == 'lesscons_lead_15' or model_name == 'varcons_lead_15'\
            or model_name == 'post_changes_to_model/cons_lead_15':
        # Define a dictionary mapping the current index values to the desired ones,
        index_mapping = {
            'scenario_': '15_1',
            'scenario_1': '15_2',
            'scenario_2': '15_3',
            'scenario_3': '15_4',
            'scenario_4': '15_5',
            'scenario_5': '15_6',
            'scenario_7': '15_8',
            'scenario_6': '15_7',
            'scenario_8': '15_9',
            'scenario_9': '15_10',
            'scenario_10': '15_11',
            'scenario_11': '15_12',
            'scenario_12': '15_13',
            'scenario_13': '15_14',
            'scenario_14': '15_15'
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
                            color='black', linewidth=2.5)
                # Check right if not on the last column
                if j < n - 1 and threshold_matrix[i][j + 1] == 0:
                    ax.plot([j + 1 + half_step, j + 1 + half_step], [n - i - 1 + half_step, n - i + half_step],
                            color='black', linewidth=2.5)
                # Check above if not in the first row and to the right of the diagonal
                if i > 0 and threshold_matrix[i - 1][j] == 0 and j >= n - i:
                    ax.plot([j + half_step, j + 1 + half_step], [n - i + half_step, n - i + half_step], color='black',
                            linewidth=2.5)
                # Check left if not in the first column and strictly below the diagonal
                if j > 0 and threshold_matrix[i][j - 1] == 0 and j >= n - i:
                    ax.plot([j + half_step, j + half_step], [n - i - 1 + half_step, n - i + half_step], color='black',
                            linewidth=2.5)

def run_unused_plots():
    # plot capacity composition over time
    if False:
        desired_scenarios = ['6_6', '15_15', '6_1', '15_1']
        # model_name = get_model_names(desired_scenarios)
        el_gen_capacity_over_time()

    # plot el generation costs over time
    if False:
        desired_scenarios = ['6_6', '15_15', '6_1', '15_1']
        # model_name = get_model_names(desired_scenarios)
        el_gen_costs_over_time()

    # plot full time series of a technology
    if False:
        model_name = "cons_nolead_15"
        desired_scenarios = ['15_1']
        r, config = read_in_results(model_name)
        granularity = "y"  # "y", "h"
        year = 10  # (0-14) (info only relevant for hourly resolution plots)
        technology = "heat_pump"  # "heat_pump" natural_gas_boiler
        full_ts_conv_tech(r, technology, granularity, year)

    # plot opex and capex
    if False:
        costs = ["cost_opex_total", "cost_capex_total" , "cost_carrier_total"]  # or "cost_opex_total", "cost_capex_total" , "cost_carrier_total"
        plot_costs_over_time(costs)

    # plot cumulative emissions
    if False:
        model_name = "post_changes_to_model/cons_lead_15"
        desired_scenarios = ['15_15', '15_14', '15_13', '15_12', '15_11', '15_10', '15_9', '15_8', '15_7', '15_6', '15_5', '15_4', '15_3', '15_2', '15_1']
        r, config = read_in_special_results(model_name)
        cumulative_emissions_plot()

    # plot capacity addition over time
    if False:
        technology = "carbon_storage"  # or None heat_pump, carbon_storage, nuclear, hard_coal_plant, run-of-river_hydro, wind_offshore, biomass_plant
        capacity_type = 'power'  # or energy
        capacity_addition_plot(technology, capacity_type)

    # investment projections plot:
    if False:
        assert len(desired_scenarios) == 1, "too many scenarios for this plot"
        r, config = read_in_results(model_name)
        projected_vs_actual_emissions_plot()

    # investment projections plot:
    if False:
        desired_scenarios = ['15_1']
        assert len(desired_scenarios) == 1, "too many scenarios for this plot"
        feedback_plot_progression()

# Function to add data to the matrix, modified to ignore squares above the diagonal
def add_data_from_bottom(matrix, df, n):
    for key, value in df.items():
        col, row = map(int, key.split('_'))
        # Adjust indices for 0-based indexing and start from the bottom row
        adjusted_row = n - row
        adjusted_col = col - 1
        matrix[adjusted_row][adjusted_col] = value

def generate_positions_corrected(matrix):
    n = len(matrix)  # Assuming square matrix, so only one dimension is needed
    positions = []

    for i in range(n):
        for j in range(n):
            if matrix[i][j] == 1:
                # Calculate positions based on user's definition (correcting the calculation)
                # Inverting the row numbering according to the user's instructions
                row = n - i
                col = j + 1
                positions.append(f'{col}_{row}')

    return positions

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

def get_threshold_cost_matrix(threshold_matrix):
    matrix_of_ones = [[1 for _ in range(n)] for _ in range(n)]
    not_feasible_matrix = []
    # Extending the pattern to the rest of the matrix
    for i in range(n):
        row = [1] * (n - i - 1) + [0] * (i + 1)
        not_feasible_matrix.append(row)
    cost_matrix = \
        [[matrix_of_ones[i][j] - not_feasible_matrix[i][j] - threshold_matrix[i][j] for j in range(n)] for i in
         range(n)]
    return cost_matrix

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
    carbon_budget = 12232.4

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

def plot_stairs(normalized_df_emissions, threshold_matrix, max_value_emissions, min_value_emissions, fig_title, cbar_label,
                normalized_df_costs, max_value_costs, min_value_costs, cbar_label_costs, empty_plot):
    print("plotting stairs")
    # Sample matrix to color the squares
    emissions_matrix = []
    # Extending the pattern to the rest of the matrix
    for i in range(n):
        row = [1] * (n - i - 1) + [0] * (i + 1)
        emissions_matrix.append(row)

    # Add the dataset to the matrix, filling from the bottom
    add_data_from_bottom(emissions_matrix, normalized_df_emissions, n)
    # Threshold for rounding
    threshold = 10 ** -5
    # Rounding values in the matrix
    emissions_matrix = [[0 if abs(value) < threshold else value for value in row] for row in emissions_matrix]
    emissions_array = np.array(emissions_matrix)
    threshold_array = np.array(threshold_matrix)
    # element-wise multiplication
    emissions_matrix = emissions_array * threshold_array
    emissions_matrix = emissions_matrix.tolist()

    costs_matrix = [[0 for _ in range(n)] for _ in range(n)]
    # Add the dataset to the matrix, filling from the bottom
    add_data_from_bottom(costs_matrix, normalized_df_costs, n)

    # Creating a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Setting the ticks at the edges of each grid square
    ax.set_xticks(np.arange(0.5, n + 2, 1))
    ax.set_yticks(np.arange(0.5, n + 2, 1))

    # Setting the minor ticks at the center of each grid square for labeling
    ax.set_xticks(np.arange(1, n + 1, 1), minor=True, )
    ax.set_yticks(np.arange(1, n + 1, 1), minor=True)

    # Labeling the axes
    ax.set_xlabel("Investment Foresight [yr]", fontsize=14)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylabel("Operation Foresight [yr]", fontsize=14)

    # Adding a grid
    ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.05)

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
    cmap_costs = sns.light_palette("green", as_cmap=True)

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
                elif emissions_matrix[i][j] != 0 and empty_plot != True:
                    # Get color from color map using the normalized value
                    color_emissions = cmap(emissions_matrix[i][j])
                    # Adding a colored square at the specified location
                    square = patches.Rectangle((j + 0.5, n - i - .5), 1, 1, facecolor=color_emissions)
                    ax.add_patch(square)
                elif costs_matrix[i][j] != 0 and empty_plot != True:
                    color_costs = cmap_costs(costs_matrix[i][j])
                    square = patches.Rectangle((j + 0.5, n - i - .5), 1, 1, facecolor=color_costs)
                    ax.add_patch(square)
                    print("nothing to plot here")

    # Call the function to draw lines around 1s adjacent to 0s
    draw_lines_around_threshold(ax, threshold_matrix)

    if empty_plot != True:
        # Create a ScalarMappable with our red color map and the normalization
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=((min_value_emissions / min_value_emissions) - 1) * 100,
                                                      vmax=((max_value_emissions / min_value_emissions) - 1) * 100))

        # Create a ScalarMappable with our green color map and the normalization
        sm_costs = ScalarMappable(cmap=cmap_costs, norm=Normalize(vmin=((min_value_costs / min_value_costs) - 1) * 100,
                                                      vmax=((max_value_costs / min_value_costs) - 1) * 100))
    # plt.title(fig_title)

        # Create the colorbar
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
        cbar.set_label(cbar_label, fontsize=12)
        # Reverse the direction of the colorbar
        cbar.ax.invert_xaxis()
        cbar_pos = cbar.ax.get_position()
        new_pos = [cbar_pos.x0 - 0.05, cbar_pos.y0, cbar_pos.width, cbar_pos.height]
        cbar.ax.set_position(new_pos)

        # Create the colorbar
        cbar_costs = plt.colorbar(sm_costs, ax=ax, orientation='vertical', fraction=0.04, pad=0.1)
        cbar_costs.set_label(cbar_label_costs, fontsize=12)
        # Reverse the direction of the colorbar
        cbar_costs.ax.invert_yaxis()

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

    # # generate figure title:
    # fig_title = get_fig_title(plot_name=plot_name,
    #                           formatted_tech=formatted_tech,
    #                           granularity=granularity, year_str=year_str)

    # Plot the selected series
    ax = df.plot(kind='line', marker='', figsize=(10, 6), linewidth=1)
    plt.xlabel('Year')
    plt.ylabel("Conversion GWh?? ")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # Add faint grid
    plt.legend(loc='best')
    plt.title("Title")

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

    # Initialize an empty list to store DataFrames from each model
    dfs = []
    if type(model_name) is not str:
        for name in model_name:
            r, config = read_in_results(name)
            temp_df = r.get_total("carbon_emissions_cumulative")
            # indexing
            temp_df = indexmapping(temp_df, special_model_name=name)
            dfs.append(temp_df)
    else:
        r, config = read_in_results(model_name)
        temp_df = r.get_total("carbon_emissions_cumulative")
        # indexing
        temp_df = indexmapping(temp_df, special_model_name=model_name)
        dfs.append(temp_df)

    # Concatenate all DataFrames in the list to a single DataFrame
    df = pd.concat(dfs, axis=0, join='outer')

    # Select only the desired rows
    if desired_scenarios is not None:
        df = df.loc[desired_scenarios]
    df_T = df.T

    fig, ax = plt.subplots(figsize=(12, 8))
    carbon_budget = 12.2324
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
    global model_name
    with_projections = True
    print("plotting yearly capacity addition")
    if type(model_name) == str:
        model_name = [model_name]
    dfs =[]
    for name in model_name:
        r, config = read_in_results(name)
        if with_projections == True:
            temp_df = r.get_total("capacity_addition", keep_raw=True)
            mf_values = temp_df.index.get_level_values('mf')
            mask = mf_values.str.endswith('_inv')
            temp_df = temp_df.loc[mask]
        else:
            temp_df = r.get_total("capacity_addition")
        temp_df = indexmapping(temp_df, special_model_name=name)
        # Reset the index to work with it as columns
        temp_df.reset_index(inplace=True)
        print("reset complete")
        # The 'level_0' column contains the scenario information, so we will rename it
        temp_df.rename(columns={'level_0': 'scenario'}, inplace=True)
        # Now we set the multi-index again, this time including the 'scenario'
        print("re-index complete")
        if with_projections == True:
            temp_df.set_index(['scenario', 'technology', 'capacity_type', 'location', 'mf'], inplace=True)
            # Perform the aggregation by summing over the 'node' level of the index
            temp_df = temp_df.groupby(level=['scenario', 'technology', 'capacity_type', 'mf']).sum()
        else:
            temp_df.set_index(['scenario', 'technology', 'capacity_type', 'location'], inplace=True)
            # Perform the aggregation by summing over the 'node' level of the index
            temp_df = temp_df.groupby(level=['scenario', 'technology', 'capacity_type']).sum()
        print("aggregation complete")

        if desired_scenarios is not None:
            temp_df = temp_df.loc[temp_df.index.get_level_values('scenario').isin(desired_scenarios)]

        # Append the processed DataFrame to the list
        dfs.append(temp_df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')

    # Assuming df is your DataFrame
    # Group by the first two levels of the multi-index ('scenario' and 'technology'), then sum
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

def projected_vs_actual_emissions_plot():
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

    carbon_budget = 12.2324
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

def stair_plot(feature, model_list, r, technology=None, capacity_type=None, empty_plot=None):
    plot_name = "stair_plot"
    print("setting up stairs plot")
    global n
    # Number of rows and columns of stair plot
    n = 15
    carbon_budget = 12232.4
    skip_load = False
    if type(model_list) is not list:
        model_list = [model_list]
        skip_load = True

    # Initialize an empty list to store DataFrames from each model
    dfs_emissions = []
    dfs_costs = []
    # Iterate over each model name, fetching and processing its results
    for name in model_list:
        r, config = read_in_results(name)
        print("model successfully read_in: ")
        print(str(name))

        temp_df_emissions = {}
        temp_df_emissions = r.get_total("carbon_emissions_cumulative")
        temp_df_emissions = temp_df_emissions.iloc[:, -1]
        # indexing
        temp_df_emissions = indexmapping(temp_df_emissions, special_model_name=name)
        dfs_emissions.append(temp_df_emissions)
        cbar_label_emissions = 'Total Cumulative Eimssions [% over carbon budget]'

        costs = ['cost_opex_total', 'cost_capex_total', 'cost_carrier_total']
        temp_df_costs = {}
        for c in costs:
            temp_df_costs[c] = r.get_total(c)
        temp_df_costs = pd.concat(temp_df_costs, keys=temp_df_costs.keys())
        temp_df_costs = temp_df_costs.groupby(level=1).sum()
        temp_df_costs = temp_df_costs.sum(axis=1)
        # indexing
        temp_df_costs = indexmapping(temp_df_costs, special_model_name=name)
        # Append the processed DataFrame to the list
        dfs_costs.append(temp_df_costs)
        cbar_label_costs = "Total System Costs (CAPEX, OPEX and Fuel) [% more expensive]"

    # Concatenate all DataFrames in the list to a single DataFrame
    df_emissions = pd.concat(dfs_emissions, axis=0, join='outer')
    df_costs = pd.concat(dfs_costs, axis=0, join='outer')

    # get the threshold matrix (combinations of foresight horizons that don't stay within the carbon budget)
    threshold_matrix = emissions_threshold_matrix_creation(model_list)

    # get the cost matrix of entries that need to be filled with cost_data
    threshold_cost_matrix = get_threshold_cost_matrix(threshold_matrix)
    costs_list = generate_positions_corrected(threshold_cost_matrix)
    df_costs = df_costs.loc[df_costs.index.intersection(costs_list)]

    # min and max values
    min_value_emissions = carbon_budget
    max_value_emissions = df_emissions.max()

    min_value_costs = df_costs.min()
    max_value_costs = df_costs.max()

    # Normalize the dataset values to a range of 0 to 1 for color mapping
    normalized_df_emissions = (df_emissions - min_value_emissions) / (max_value_emissions - min_value_emissions)
    normalized_df_costs = (df_costs - min_value_costs) / (max_value_costs - min_value_costs)

    # TODO this is not clean: ask Jacob and make cahnges
    if len(normalized_df_costs) != 0:
        second_to_lowest_value = normalized_df_costs.nsmallest(2).iloc[-1]
        index_of_lowest = normalized_df_costs.idxmin()
        normalized_df_costs.loc[index_of_lowest] = second_to_lowest_value/10

    # get fig title
    fig_title = get_fig_title(plot_name=plot_name, model_list=model_list, feature=feature, technology=technology)

    # plot the data:
    plot_stairs(normalized_df_emissions, threshold_matrix, max_value_emissions, min_value_emissions, fig_title,
                cbar_label_emissions, normalized_df_costs, max_value_costs, min_value_costs, cbar_label_costs, empty_plot)

    return

def get_model_names(desired_scenarios):
    # get investment horizons
    inv_horizons = [int(s.split('_')[0]) for s in desired_scenarios]

    # Conditions and corresponding strings
    conditions_strings = {
        (1, 4): "cons_lead_1to4",
        (5, 7): "cons_lead_5to7",
        (8, 10): "cons_lead_8to10",
        11: "cons_lead_11",
        12: "cons_lead_12",
        13: "cons_lead_13",
        14: "cons_lead_14",
        15: "cons_lead_15",
    }

    # New list to add strings to, based on conditions met
    names = []

    # Check each condition
    for condition, string in conditions_strings.items():
        if isinstance(condition, tuple):  # Range condition
            # If any number in the range is in numbers, add the string
            if any(n in inv_horizons for n in range(condition[0], condition[1] + 1)):
                names.append(string)
        else:  # Single value condition
            # If the specific value is in numbers, add the string
            if condition in inv_horizons:
                names.append(string)
    return names


def comparative_costs_over_time():
    global model_name
    only2 = False
    discounted = False
    blank = False
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

        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df = df /1000
    df = df.sort_index()
    # Set the color palette
    colors = sns.color_palette('pastel')[:2]  # Choosing two colors from the 'pastel' palette
    saved_lines = {}
    saved_lines_capex = {}

    if len(desired_scenarios) == 4 and only2 is not True:
        # Set up a 2x2 grid of plots with no space between them
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        axs = axs.flatten()

        for i, scenario in enumerate(desired_scenarios):
            # For each subplot, filter df for the current scenario across both cost types
            scenario_data = {cost_type: df.loc[cost_type, scenario] for cost_type in
                             ['cost_opex_total', 'cost_capex_total']}

            # Convert the scenario data into a DataFrame for plotting
            scenario_df = pd.DataFrame(scenario_data)

            # Calculating the stacked values
            scenario_df['stacked_opex'] = scenario_df['cost_opex_total'] + scenario_df['cost_capex_total']

            # If scenario is "15_15" or "7_7", save the 'stacked_opex' line for later use
            if scenario in ["15_15", "7_7"]:
                saved_lines[scenario] = scenario_df['stacked_opex']
                saved_lines_capex[scenario] = scenario_df['cost_capex_total']

            # Plotting areas
            axs[i].fill_between(scenario_df.index, 0, scenario_df['cost_capex_total'], label='CAPEX', color=colors[0],
                                alpha=0.5)
            axs[i].fill_between(scenario_df.index, scenario_df['cost_capex_total'], scenario_df['stacked_opex'],
                                label='OPEX', color=colors[1], alpha=0.5)

            # Setting titles and labels
            if scenario in ["7_1", "7_4","7_7"]:
                axs[i].set_ylabel('Cost [billion €]')

            # Set y-axis limits
            if discounted == True:
                axs[i].set_ylim(0, 650)
            else:
                axs[i].set_ylim(0, 650)

        for i, scenario in enumerate(desired_scenarios):
            # Additional plotting of saved lines in specified scenarios
            if scenario == "7_7":
                # Plotting the line for "15_15" in the subplot of "7_7"
                if "15_15" in saved_lines:
                    axs[i].plot(saved_lines["15_15"].index, saved_lines["15_15"], 'k--', label='15_15 line')
                    axs[i].plot(saved_lines["15_15"].index, saved_lines_capex["15_15"], color='k', linestyle=':', label='15_15 line')
                    inv_horizon, op_horizon = scenario.split('_')
            #         # Adding a custom legend to the subplot
                    legend_elements = [plt.Line2D([0], [0], color='k', linestyle='--', label='Perfect Foresight  Total')]
                    legend_elements.append(Line2D([0], [0], color='k', linestyle=':', label='Perfect Foresight  CAPEX'))
                    axs[i].legend(handles=legend_elements, loc='upper right',
                                  title=f'Investment Foresight: {int(inv_horizon) * 2}yr\nOperation Foresight: {int(op_horizon) * 2}yr',
                                  framealpha=1)
            #         axs[i].legend(handles=legend_elements, loc='upper right', framealpha=1)
            if scenario == "7_4":
                # Plotting the line for "7_7" in the subplot of "7_1"
                if "7_7" in saved_lines:
                    axs[i].plot(saved_lines["7_7"].index, saved_lines["7_7"], 'k--', label='7_7 line')
                    axs[i].plot(saved_lines["7_7"].index, saved_lines_capex["7_7"], color='k', linestyle=':', label='7_7 line')
                    inv_horizon, op_horizon = scenario.split('_')
                    # Adding a custom legend to the subplot
                    legend_elements = [plt.Line2D([0], [0], color='k', linestyle='--', label='Total with OF: 14yr')]
                    legend_elements.append(Line2D([0], [0], color='k', linestyle=':', label='CAPEX with OF: 14yr'))
                    # legend_elements = [plt.Line2D([0], [0], color='k', linestyle='--', label='Standard Myopic Foresight')]
                    # legend_elements.append(Line2D([0], [0], color='k', linestyle=':', label='Standard Myopic Foresight, CAPEX'))
                    axs[i].legend(handles=legend_elements, loc='upper right',
                                  title=f'Investment Foresight: {int(inv_horizon) * 2}yr\nOperation Foresight: {int(op_horizon) * 2}yr',
                                  framealpha=1)
                    # axs[i].legend(handles=legend_elements, loc='upper right', framealpha=1)
            elif scenario == "15_4":
                # Plotting the line for "15_15" in the subplot of "15_1"
                if "15_15" in saved_lines:
                    axs[i].plot(saved_lines["15_15"].index, saved_lines["15_15"], 'k--', label='15_15 line')
                    axs[i].plot(saved_lines["15_15"].index, saved_lines_capex["15_15"], color='k', linestyle=':', label='15_15 line')
                    inv_horizon, op_horizon = scenario.split('_')
                    # Adding a custom legend to the subplot
                    legend_elements = [plt.Line2D([0], [0], color='k', linestyle='--', label='Perfect Foresight  Total')]
                    legend_elements.append(Line2D([0], [0], color='k', linestyle=':', label='Perfect Foresight  CAPEX'))
                    axs[i].legend(handles=legend_elements, loc='upper right',
                              title=f'Investment Foresight: {int(inv_horizon) * 2}yr\nOperation Foresight: {int(op_horizon) * 2}yr',
                              framealpha=1)
                    # axs[i].legend(handles=legend_elements, loc='upper right', framealpha=1)
            elif scenario == "15_15":
                legend_elements = [patches.Patch(facecolor=colors[1], edgecolor='k', alpha=0.5, label='OPEX'),
                                   patches.Patch(facecolor=colors[0], edgecolor='k', alpha=0.5, label='CAPEX')]
                axs[i].legend(handles=legend_elements, loc='upper right',
                              title=f'Perfect Foresight', framealpha=1) # , bbox_to_anchor=(0, 0)

        # Make the subplots touch by removing space between them
        plt.subplots_adjust(hspace=0, wspace=0)

        # Adjusting the x-tick labels
        axs[0].figure.canvas.draw()  # This ensures the ticks are updated
        original_ticks = axs[0].get_xticks()
        new_labels = [2022 + 2 * int(label) for label in original_ticks]
        for ax in axs:
            ax.set_xticklabels(new_labels, rotation=30)

    elif only2 == True:
        # Set up a 2x1 grid of plots with no space between them
        fig, axs = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
        axs = axs.flatten()

        for i, scenario in enumerate(['15_15', '15_4']):
            if blank is not True:
                # For each subplot, filter df for the current scenario across both cost types
                scenario_data = {cost_type: df.loc[cost_type, scenario] for cost_type in
                                 ['cost_opex_total', 'cost_capex_total']}

                # Convert the scenario data into a DataFrame for plotting
                scenario_df = pd.DataFrame(scenario_data)

                # Calculating the stacked values
                scenario_df['stacked_opex'] = scenario_df['cost_opex_total'] + scenario_df['cost_capex_total']

                # If scenario is "15_15", save the 'stacked_opex' line for later use
                if scenario in ["15_15"]:
                    saved_lines[scenario] = scenario_df['stacked_opex']
                    saved_lines_capex[scenario] = scenario_df['cost_capex_total']

                # Plotting areas
                axs[i].fill_between(scenario_df.index, 0, scenario_df['cost_capex_total'], label='CAPEX', color=colors[0],
                                    alpha=0.5)
                axs[i].fill_between(scenario_df.index, scenario_df['cost_capex_total'], scenario_df['stacked_opex'],
                                    label='OPEX', color=colors[1], alpha=0.5)

            axs[i].set_ylabel('Cost [billion €]')
            # Set y-axis limits
            if discounted == True:
                axs[i].set_ylim(0, 650)
            else:
                axs[i].set_ylim(0, 650)

        for i, scenario in enumerate(['15_15', '15_4']):
            if blank is not True:  # assuming there's a variable `blank` determining some condition
                legend_elements = []
                if scenario == "15_4":
                    if "15_15" in saved_lines:
                        axs[i].plot(saved_lines["15_15"].index, saved_lines["15_15"], 'k--', label='15_15 line')
                        axs[i].plot(saved_lines["15_15"].index, saved_lines_capex["15_15"], color='k', linestyle=':',
                                    label='15_15 line')
                        inv_horizon, op_horizon = scenario.split('_')
                        legend_elements = [plt.Line2D([0], [0], color='k', linestyle='--', label='P.F. Costs'),
                                           plt.Line2D([0], [0], color='k', linestyle=':', label='P.F. CAPEX')]

                elif scenario == "15_15":
                    legend_elements = [patches.Patch(facecolor=colors[1], edgecolor='k', alpha=0.5, label='OPEX'),
                                       patches.Patch(facecolor=colors[0], edgecolor='k', alpha=0.5, label='CAPEX')]

                if legend_elements:
                    # Create the legend
                    legend = axs[i].legend(handles=legend_elements, loc='upper right',
                                           title=f'Investment Foresight: 30yr\nOperation Foresight: 8yr' if scenario == "15_4" else 'Perfect Foresight',
                                           framealpha=1)
                    # Set the title of the legend to be bold
                    plt.setp(legend.get_title(), fontweight='bold')

        # Make the subplots touch by removing space between them
        plt.subplots_adjust(hspace=0, wspace=0)

        # Adjusting the x-tick labels
        axs[0].figure.canvas.draw()  # This ensures the ticks are updated
        if blank is True:
            # Set x-ticks to range from 0 to 14
            axs.set_xticks(np.arange(0, 15, 1))
            original_ticks = axs[0].get_xticks()
        else:
            original_ticks = axs[0].get_xticks()
        new_labels = [2022 + 2 * int(label) for label in original_ticks]
        for ax in axs:
            ax.set_xticklabels(new_labels, rotation=30)

    # Show the plot
    plt.show(block=True)

def el_gen_costs_over_time():
    global model_name
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]

    dfs = []
    for name in model_name:
        r, config = read_in_results(name)

        costs = ["cost_opex"] # , "cost_carrier"
        df = {}
        for c in costs:
            df[c] = r.get_total(c)
            # Reset the index to work with it as columns
            df[c].reset_index(inplace=True)
            print("reset complete")
            # The 'level_0' column contains the scenario information, so we will rename it
            df[c].rename(columns={'level_0': 'scenario'}, inplace=True)
            # Now we set the multi-index again, this time including the 'scenario'
            if c == "cost_opex":
                df[c].set_index(['scenario', 'technology', 'location'], inplace=True)
                df[c] = df[c].groupby(['scenario', 'technology']).sum()
            elif c == "cost_carrier":
                df[c].set_index(['scenario', 'carrier', 'node'], inplace=True)
                df[c] = df[c].groupby(['scenario', 'carrier']).sum()
            else: raise ValueError("this type of df does not fit here")
            df[c] = indexmapping(df[c], special_model_name=name)
            # reducing to the desired scenarios
            df[c] = df[c].loc[df[c].index.get_level_values('scenario').isin(desired_scenarios)]

            # # List of technologies to remove
            technologies_to_remove = [
                 "biomass_boiler", "biomass_boiler_DH", "carbon_pipeline", "carbon_storage",
                 "district_heating_grid", "electrode_boiler", "electrode_boiler_DH", "hard_coal_boiler",
                 "hard_coal_boiler_DH", "heat_pump", "heat_pump_DH", "industrial_gas_consumer", "lng_terminal",
                 "natural_gas_boiler", "natural_gas_boiler_DH", "natural_gas_pipeline", "natural_gas_storage",
                 "hydrogen_storage", "oil_boiler", "oil_boiler_DH", "waste_boiler_DH"]
            # list_of_not_others = ['hard_coal_plant', 'hard_coal_plant_CCS', 'lignite_coal_plant',
            #                       'natural_gas_turbine', 'natural_gas_turbine_CCS', 'nuclear', 'biomass_plant',
            #                       'biomass_plant_CCS', 'photovoltaics', 'wind_offshore', 'wind_onshore', 'power_line']
            # list_of_others = ['oil_plant', 'waste_plant']
            # list_of_storage = ['battery']
            # list_of_hydro = ['pumped_hydro', 'reservoir_hydro', 'run-of-river_hydro']
            # Remove rows where technology matches any in the removal list
            df[c].drop(index=technologies_to_remove, level='technology', inplace=True)
            # # list of technologies in storage
            df[c].rename(index={'oil_plant': 'Others', 'waste_plant': 'Others'}, level='technology', inplace=True)
            df[c].rename(index={'battery': 'storage'}, level='technology', inplace=True)
            df[c].rename(index={'pumped_hydro': 'Hydro', 'reservoir_hydro': 'Hydro', 'run-of-river_hydro': 'Hydro'}, level='technology', inplace=True)
            df[c] = df[c].groupby(level=['scenario', 'technology']).sum()

        df = pd.concat(df, keys=df.keys())



        # Append the processed DataFrame to the list
        dfs.append(df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')
    df_totals = df
    df_totals = df_totals.T.sum()

    # Check if the number of desired scenarios is exactly four
    if len(desired_scenarios) == 4:
        # Set up a 2x2 grid of plots
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()

        legend_labels = {}  # Dictionary to hold unique legend labels without scenario tags and their handles

        for i, scenario in enumerate(desired_scenarios):
            scenario_df = df.xs(scenario, level='scenario', drop_level=False)
            scenario_df = scenario_df.T
            scenario_df.plot(kind='bar', stacked=True, ax=axs[i])

            axs[i].set_title(f'Scenario {scenario}')
            axs[i].set_xlabel('Technology')
            axs[i].set_ylabel('OPEX')

            # Retrieve handles and labels for the current subplot
            handles, labels = axs[i].get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                # Extract technology name from the label
                # Assuming label format is like '(cost_opex, 15_1, technology)'
                technology = label.split(', ')[-1].rstrip(')')
                if technology not in legend_labels:  # Add to dictionary if not already present
                    legend_labels[technology] = handle

        # Hide the individual legends for each subplot
        for ax in axs:
            ax.get_legend().remove()

        plt.subplots_adjust(hspace=0.5)

        global_min = min(ax.get_ylim()[0] for ax in axs)
        global_max = max(ax.get_ylim()[1] for ax in axs)
        for ax in axs:
            ax.set_ylim(global_min, global_max)

        # Now create a single legend with the collected unique handles and labels, without scenario tags
        fig.legend(legend_labels.values(), legend_labels.keys(), loc='upper center', bbox_to_anchor=(0.5, 1),
                   ncol=math.ceil(len(legend_labels) / 4))

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(top=0.85)
        plt.show(block=True)

    print("done")


def el_gen_capacity_over_time():
    dfs = []
    global model_name
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]
    for name in model_name:
        r, config = read_in_results(name)
        temp_df = r.get_total("capacity")

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
        # reducing to the desired scenarios
        temp_df = temp_df.loc[temp_df.index.get_level_values('scenario').isin(desired_scenarios)]
        # get rid of capacit_type "energy"
        temp_df = temp_df[temp_df.index.get_level_values('capacity_type') != 'energy']
        temp_df.index = temp_df.index.droplevel('capacity_type')
        # List of technologies to remove
        technologies_to_remove = [
            "biomass_boiler", "biomass_boiler_DH", "carbon_pipeline", "carbon_storage",
            "district_heating_grid", "electrode_boiler", "electrode_boiler_DH", "hard_coal_boiler",
            "hard_coal_boiler_DH", "heat_pump", "heat_pump_DH", "industrial_gas_consumer", "lng_terminal",
            "natural_gas_boiler", "natural_gas_boiler_DH", "natural_gas_pipeline", "natural_gas_storage",
            "hydrogen_storage", "oil_boiler", "oil_boiler_DH", "waste_boiler_DH"]
        # Remove rows where technology matches any in the removal list
        temp_df.drop(index=technologies_to_remove, level='technology', inplace=True)
        # list of technologies in storage
        temp_df.rename(index={'battery': 'storage', 'pumped_hydro': 'storage'},
                  level='technology', inplace=True)
        temp_df = temp_df.groupby(level=['scenario', 'technology']).sum()


        # Append the processed DataFrame to the list
        dfs.append(temp_df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')

    color_mapping = {
        'hard_coal_plant': '#4c7ebf',
        'hard_coal_plant_CCS': '#af4e96',
        'lignite_coal_plant': '#8c9acc',
        'natural_gas_turbine': '#c6a178',
        'natural_gas_turbine_CCS': '#007894',
        'nuclear': '#05417e',
        'biomass_plant': '#a1ab71',
        'biomass_plant CCS': '#627213',
        'others': '#d6d6d6',
        'hydro': '#78642b',
        'solar_PV': '#a9a9a9',
        'wind_offshore': '#d48681',
        'wind_onshore': '#575757',
        'power_line': '#000000',
        'Storage': '#ffffff'
    }

    # Check if the number of desired scenarios is exactly four
    if len(desired_scenarios) == 4:
        # Set up a 2x2 grid of plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs = axs.flatten()  # Flatten the array to make iteration easier

        # Iterate through each desired scenario and its corresponding subplot
        for i, scenario in enumerate(desired_scenarios):
            # Assuming `df` is properly structured and filtered for the desired scenarios
            # Filter the DataFrame for the current scenario
            # This line is just for demonstration; replace `df` with your actual DataFrame variable
            scenario_df = df.loc[scenario] if scenario in df.index.levels[0] else pd.DataFrame()
            scenario_df = scenario_df.T

            # Plot the capacities over time for the scenario
            scenario_df.plot(kind='bar', stacked=True, ax=axs[i])

            axs[i].set_title(f'Scenario {scenario}')
            axs[i].set_xlabel('Technology')
            axs[i].set_ylabel('Capacity')
            axs[i].legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Hide the individual legends for each subplot
        for ax in axs:
            ax.get_legend().remove()

        plt.subplots_adjust(hspace=0.5)  # Adjust 'hspace' as needed for more space

        global_min = min(ax.get_ylim()[0] for ax in axs)
        global_max = max(ax.get_ylim()[1] for ax in axs)
        # Set the same Y-axis limits for all subplots
        for ax in axs:
            ax.set_ylim(global_min, global_max)

        # Create a single legend for the entire figure using handles and labels from the last subplot
        handles, labels = ax.get_legend_handles_labels()

        # Determine the number of columns for the legend to span three lines
        num_technologies = len(labels)
        ncols_legend = math.ceil(num_technologies / 3)

        # Create the legend with the determined number of columns
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=ncols_legend)

        # Adjust the layout to prevent overlapping
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to leave space for the legend at the top

        plt.subplots_adjust(top=0.85)

        # Show the plot
        plt.show(block=True)

    print("done")


def feedback_plot_progression():
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
    renewable_el_generation = ['biomass_plant', 'biomass_plant_CCS', 'nuclear',
                              'photovoltaics', 'reservoir_hydro', 'run-of-river_hydro',
                               'waste_plant', 'wind_offshore', 'wind_onshore']
    renewable_heating = ['biomass_boiler', 'biomass_boiler_DH', 'district_heating_grid',
                         'heat_pump', 'heat_pump_DH', 'waste_boiler_DH',]
    # renewable_storage = ['battery', 'pumped_hydro', 'hydrogen_storage']
    technologies_to_keep = renewable_el_generation + renewable_heating# + renewable_storage
    df_cap = df_cap.loc[(slice(None), technologies_to_keep), :]

    mapping_dict = {}
    for tech in renewable_el_generation:
        mapping_dict[tech] = 'renewable_el_generation'
    for tech in renewable_heating:
        mapping_dict[tech] = 'renewable_heating'
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
    def get_color_key(row_identifier):
        number_part = row_identifier.split('_')[0]
        return "df_" + number_part

    # Create figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # Choose a palette
    # palette = sns.color_palette("Set1", 7)  # 'Set2' for distinct, muted colors; change n_colors if more categories
    palette = sns.dark_palette("seagreen", reverse=True, n_colors=256)
    # Create equally spaced indices to select colors
    indices = np.linspace(0, 255, num=15, dtype=int)
    selected_colors = np.array(palette)[indices]

    # Convert the palette to a list of RGB strings colors can be directly used in plotting functions
    # colors = [mcolors.to_hex(c) for c in palette]
    iterable_colors = list(map(tuple, selected_colors))

    colors = {
        'df_0': iterable_colors[0],
        'df_1': iterable_colors[3],
        'df_2': iterable_colors[4],
        'df_3': iterable_colors[5],
        'df_4': iterable_colors[4],
        'df_5': iterable_colors[8],
        'df_6': iterable_colors[9],
        'df_14': iterable_colors[14]
    }

    # Plotting emissions on the primary y-axis
    legend_labels = [f"{2022 + 2 * int(row[1].split('_')[0])}" for row in inv_rows_emissions.index]
    legend_labels = legend_labels[:-1]
    i = 0
    for row in inv_rows_emissions.index:
        color_key = get_color_key(row[1])
        if i < len(legend_labels):
            label = legend_labels[i]
        else:
            label = None
        ax1.plot(((inv_rows_emissions.loc[row, :]) / 1000).dropna().index, ((inv_rows_emissions.loc[row, :]) / 1000).dropna(), label=label, color=colors[color_key])
        i += 1
    ax1.plot(op_rows_emissions / 1000, label='Actual', color='black', marker='', markersize=4,
             markerfacecolor='black', lw=3)

    first_y_value = op_rows_emissions[0] / 1000
    ax1.plot([-0.5, 0], [0, first_y_value], color='black', lw=3)

    # Setting the labels for the primary y-axis
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Cumulative Carbon Emissions [Gt CO₂]', fontsize=12)

    # Carbon budget line on the primary y-axis
    carbon_budget = 12.2324
    ax1.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0, xmax=0.95, zorder=1, alpha=0.5)
    ax1.text(2.25, carbon_budget + 0.005 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='left', color='black',
             fontsize=10)


    # Create secondary y-axis for the capacities
    ax2 = ax1.twinx()
    # Number of DataFrames (scenarios) and an example of time steps (assuming they're consistent across DataFrames)
    num_dfs = len(inv_cap_dataframes)
    time_steps = list(
        inv_cap_dataframes[next(iter(inv_cap_dataframes))].columns)  # Grab time steps from the first DataFrame
    bar_width = 0.8 / num_dfs  # Width of each bar, divided by the number of scenarios
    opacity = 0.5  # Adjust as needed for visibility

    # Optional: Data scaling factor (e.g., 0.5 to halve the bar heights)
    scaling_factor = 0.1

    # Define hatch patterns and edge color
    hatch_patterns = ['////////', '', '**', '.', '']
    edge_color = 'black'  # Color for the edges of each bar segment

    for i, time_step in enumerate(time_steps):
        for j, (df_name, df) in enumerate(inv_cap_dataframes.items()):
            df_number = int(df_name.split('_')[1])
            bar_position = np.arange(len(time_steps)) + (j - len(inv_cap_dataframes) / 2) * bar_width + (bar_width / 2)
            data = df[time_step]

            bottom = np.zeros(len(data))
            bar_color = colors.get(df_name, 'black')  # Fallback color
            edge_color = 'black'

            for k, category in enumerate(reversed(data.index)):
                if time_step < df_number and df_number != 14:
                    bar_color = 'none'
                    edge_color = 'lightgrey' # 'dimgrey'
                # Cycle through hatch patterns based on technology's position
                hatch = hatch_patterns[k % len(hatch_patterns)]
                ax2.bar(bar_position[i], data.loc[category]/1000, bottom=bottom, width=bar_width, alpha=opacity,
                        color=bar_color, edgecolor=edge_color, label=f'{df_name} - {category}', hatch=hatch)
                bottom += data.loc[category]/1000

    # Setting the label for the secondary y-axis
    ax2.set_ylabel('Projected Renewable Generation Capacity [TW]', fontsize=12, labelpad=15)
    ax2.set_ylim(0, 6.5)
    ax1.set_ylim(0, 13)

    # Legend and title
    # generate figure title:
    fig_title = "Progression of Capacity Expansion Planning over Time \n (varcons_lead_tsa50 15_1)"
    plt.title(fig_title)
    legend_techs = ["Heating", "Electricity Generation"] #  , "renewable_storage"
    # Create custom legend handles based on technologies and their corresponding hatch patterns
    legend_handles = [patches.Patch(facecolor='white', edgecolor='black', hatch=hatch_patterns[i], label=legend_techs[i])
                      for i, technology in enumerate(legend_techs)]

    # For better control, manually specify the legend to include labels from both plots
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    inv_horizon, op_horizon = desired_scenarios[0].split('_')
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0, 1),
               title='Foresight Horizon\nInvestment: ' + str(int(inv_horizon) * 2) + 'a' + '\nOperation: ' + str(
                   int(op_horizon) * 2) + 'a\n\nExpected Emissions\nand Capacities, by\nDecision Maker in:', framealpha=1)

    # When plotting your data, skip the automatic legend generation
    # After all plotting commands, manually add the legend with custom handles
    plt.legend(handles=legend_handles, title="Renewable Capacities", loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)


    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    ax1.spines['left'].set_position(('data', -0.5))
    ax1.set_xlim([-0.5, 14.5])

    # Modify x-axis tick labels if needed
    # This part may require adjustment depending on your data's time format
    new_labels = [2022 + 2 * int(label) for label in ax1.get_xticks()]
    ax1.set_xticklabels(new_labels)

    plt.show(block=True)
    return

def feedback_plot_progression_with_techs():
    global model_name
    print("plotting feedback_plot_progression")
    if type(model_name) is not str:
        model_name = str(model_name)
    r, config = read_in_results(model_name)

    projections =  ['0_inv', '3_inv', '4_inv', '14_inv'] # '2_inv','6_inv',

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
    renewable_heating = ['biomass_boiler', 'biomass_boiler_DH', 'district_heating_grid',
                         'heat_pump', 'heat_pump_DH', 'waste_boiler_DH',]
    # wind = ['wind_offshore', 'wind_onshore']
    biomass = ['biomass_plant', 'biomass_plant_CCS']
    hydro = ['reservoir_hydro', 'run-of-river_hydro']
    pv = ['photovoltaics']
    wind_on = ['wind_onshore']
    wind_off = ['wind_offshore']
    # renewable_storage = ['battery', 'pumped_hydro', 'hydrogen_storage']
    technologies_to_keep = renewable_el_generation # + renewable_heating# + renewable_storage
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
    carbon_budget = 12.2324
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
    ax1.set_ylim(0, 14.5)

    # Legend and title
    # generate figure title:
    fig_title = ""
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

def sens_analysis_plot(model_name, four_models, carb_stor, gas_turb, BECCS, nuclear):
    dfs = []
    emissions = True
    # Iterate over each model name in the list
    if type(model_name) == str:
        model_name = [model_name]
    for name in model_name:
        r, config = read_in_results(name)
        print(str(name) + " successfully read in")
        modified_name = name
        if "/" in modified_name:
            # Split by "/" and take the part after the last "/"
            modified_name = modified_name.split('/')[-1]
        modified_name = '_'.join(modified_name.split('_')[:-1])
        if modified_name.endswith("_init"):
            # Split the modified_name by "_" and remove the last part, then join back together
            modified_name = '_'.join(modified_name.split('_')[:-1])

        if emissions == True:
            temp_df = r.get_total("carbon_emissions_cumulative")
            if temp_df.index.name is None:
                temp_df.index.name = 'scenario'
            temp_df['model'] = modified_name
            temp_df = temp_df.set_index('model', append=True)
            temp_df = indexmapping(temp_df, special_model_name=name)
            temp_df = temp_df.loc[temp_df.index.get_level_values('scenario').isin(desired_scenarios)]
            dfs.append(temp_df)
            carbon_budget = 12.2324
        else:
            temp_df = r.get_total("capacity")

            # Reset the index to work with it as columns
            temp_df.reset_index(inplace=True)
            print("reset complete")
            # The 'level_0' column contains the scenario information, so we will rename it
            temp_df.rename(columns={'level_0': 'scenario'}, inplace=True)

            # Add 'model' column with the current model name
            temp_df['model'] = modified_name

            temp_df.set_index(['scenario', 'technology', 'capacity_type', 'location', 'model'], inplace=True)
            # Perform the aggregation by summing over the 'node' level of the index
            temp_df = temp_df.groupby(level=['scenario', 'technology', 'capacity_type', 'model']).sum()
            print("aggregation complete")
            # indexing
            temp_df = indexmapping(temp_df, special_model_name=name)
            # reducing to the desired scenarios
            temp_df = temp_df.loc[temp_df.index.get_level_values('scenario').isin(desired_scenarios)]
            # get rid of capacit_type "energy"
            temp_df = temp_df[temp_df.index.get_level_values('capacity_type') != 'energy']
            temp_df.index = temp_df.index.droplevel('capacity_type')
            if carb_stor == True:
                technologies_to_remove = ['battery', 'biomass_boiler', 'biomass_boiler_DH', 'biomass_plant',
                                           'biomass_plant_CCS', 'carbon_pipeline',
                                           'district_heating_grid', 'electrode_boiler', 'electrode_boiler_DH',
                                           'hard_coal_boiler', 'hard_coal_boiler_DH', 'hard_coal_plant',
                                           'hard_coal_plant_CCS', 'heat_pump', 'heat_pump_DH', 'hydrogen_storage',
                                           'industrial_gas_consumer', 'lignite_coal_plant', 'lng_terminal',
                                           'natural_gas_boiler', 'natural_gas_boiler_DH', 'natural_gas_pipeline',
                                           'natural_gas_storage', 'natural_gas_turbine', 'natural_gas_turbine_CCS',
                                           'nuclear', 'oil_boiler', 'oil_boiler_DH', 'oil_plant', 'photovoltaics',
                                           'power_line', 'pumped_hydro', 'reservoir_hydro', 'run-of-river_hydro',
                                           'waste_boiler_DH', 'waste_plant', 'wind_offshore', 'wind_onshore']
                temp_df.drop(index=technologies_to_remove, level='technology', inplace=True)
            if gas_turb == True:
                technologies_to_remove = ['battery', 'biomass_boiler', 'biomass_boiler_DH', 'biomass_plant',
                                           'biomass_plant_CCS', 'carbon_pipeline', 'carbon_storage',
                                           'district_heating_grid', 'electrode_boiler', 'electrode_boiler_DH',
                                           'hard_coal_boiler', 'hard_coal_boiler_DH', 'hard_coal_plant',
                                           'hard_coal_plant_CCS', 'heat_pump', 'heat_pump_DH', 'hydrogen_storage',
                                           'industrial_gas_consumer', 'lignite_coal_plant', 'lng_terminal',
                                           'natural_gas_boiler', 'natural_gas_boiler_DH', 'natural_gas_pipeline',
                                           'natural_gas_storage', 'natural_gas_turbine_CCS',
                                           'nuclear', 'oil_boiler', 'oil_boiler_DH', 'oil_plant', 'photovoltaics',
                                           'power_line', 'pumped_hydro', 'reservoir_hydro', 'run-of-river_hydro',
                                           'waste_boiler_DH', 'waste_plant', 'wind_offshore', 'wind_onshore']
                temp_df.drop(index=technologies_to_remove, level='technology', inplace=True)
            if BECCS == True:
                technologies_to_remove = ['battery', 'biomass_boiler', 'biomass_boiler_DH', 'biomass_plant',
                                            'carbon_pipeline', 'carbon_storage',
                                           'district_heating_grid', 'electrode_boiler', 'electrode_boiler_DH',
                                           'hard_coal_boiler', 'hard_coal_boiler_DH', 'hard_coal_plant',
                                           'hard_coal_plant_CCS', 'heat_pump', 'heat_pump_DH', 'hydrogen_storage',
                                           'industrial_gas_consumer', 'lignite_coal_plant', 'lng_terminal',
                                           'natural_gas_boiler', 'natural_gas_boiler_DH', 'natural_gas_pipeline',
                                           'natural_gas_storage', 'natural_gas_turbine', 'natural_gas_turbine_CCS',
                                           'nuclear', 'oil_boiler', 'oil_boiler_DH', 'oil_plant', 'photovoltaics',
                                           'power_line', 'pumped_hydro', 'reservoir_hydro', 'run-of-river_hydro',
                                           'waste_boiler_DH', 'waste_plant', 'wind_offshore', 'wind_onshore']
                temp_df.drop(index=technologies_to_remove, level='technology', inplace=True)
            if nuclear == True:
                technologies_to_remove = ['battery', 'biomass_boiler', 'biomass_boiler_DH', 'biomass_plant',
                                            'carbon_pipeline', 'carbon_storage', 'biomass_plant_CCS',
                                           'district_heating_grid', 'electrode_boiler', 'electrode_boiler_DH',
                                           'hard_coal_boiler', 'hard_coal_boiler_DH', 'hard_coal_plant',
                                           'hard_coal_plant_CCS', 'heat_pump', 'heat_pump_DH', 'hydrogen_storage',
                                           'industrial_gas_consumer', 'lignite_coal_plant', 'lng_terminal',
                                           'natural_gas_boiler', 'natural_gas_boiler_DH', 'natural_gas_pipeline',
                                           'natural_gas_storage', 'natural_gas_turbine', 'natural_gas_turbine_CCS',
                                           'oil_boiler', 'oil_boiler_DH', 'oil_plant', 'photovoltaics',
                                           'power_line', 'pumped_hydro', 'reservoir_hydro', 'run-of-river_hydro',
                                           'waste_boiler_DH', 'waste_plant', 'wind_offshore', 'wind_onshore']
                temp_df.drop(index=technologies_to_remove, level='technology', inplace=True)
            # List of technologies to remove
            if carb_stor != True and BECCS != True and gas_turb != True and nuclear != True:
                technologies_to_remove = [
                    "biomass_boiler", "biomass_boiler_DH", "carbon_pipeline", "carbon_storage",
                    "district_heating_grid", "electrode_boiler", "electrode_boiler_DH", "hard_coal_boiler",
                    "hard_coal_boiler_DH", "heat_pump", "heat_pump_DH", "industrial_gas_consumer", "lng_terminal",
                    "natural_gas_boiler", "natural_gas_boiler_DH", "natural_gas_pipeline", "natural_gas_storage",
                    "hydrogen_storage", "oil_boiler", "oil_boiler_DH", "waste_boiler_DH"]
                # Remove rows where technology matches any in the removal list
                temp_df.drop(index=technologies_to_remove, level='technology', inplace=True)
                # list of technologies in storage
                temp_df.rename(index={'battery': 'storage', 'pumped_hydro': 'storage'},
                               level='technology', inplace=True)
                nonrenewables_to_remove = [
                    "hard_coal_plant", "hard_coal_plant_CCS", "lignite_coal_plant", "natural_gas_turbine",
                    "natural_gas_turbine_CCS", "oil_plant", "power_line", "storage", "waste_plant"]
                # Remove rows where technology matches any in the removal list
                temp_df.drop(index=nonrenewables_to_remove, level='technology', inplace=True)
            temp_df = temp_df.groupby(level=['scenario', 'model']).sum()

            # Append the processed DataFrame to the list
            dfs.append(temp_df)

    # Concatenate all DataFrames in the list to a single DataFrame
    # Adjust 'axis' and 'join' arguments as per your data structure and needs
    df = pd.concat(dfs, axis=0, join='outer')

    # Set up a 3x3 grid of plots with touching subplots
    if four_models == True:
        fig, axs = plt.subplots(4, 3, figsize=(8, 6.2), sharex=True, sharey=True)
        model_descriptors = ['Cons No Lead', 'Cons Lead', 'Less Cons Lead', 'Var Cons Lead']
        models = ['cons_nolead', 'cons_lead', 'lesscons_lead', 'varcons_lead']
    else:
        fig, axs = plt.subplots(3, 3, figsize=(8, 5.3), sharex=True, sharey=True)
        models = ['cons_nolead', 'cons_lead', 'lesscons_lead']
        model_descriptors = ['Cons No Lead', 'Cons Lead', 'Less Cons Lead']

    scenario_descriptors = ['Investment Foresight:\n6 Years', '14 Years', '30 Years']
    scenarios = [('3_1', '3_3'), ('7_1', '7_7'), ('15_1', '15_15')] # ('3_1', '3_3')

    # Eliminate margins between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Label the rows and columns instead of individual plots
    for ax, row in zip(axs[:, 0], model_descriptors):
        ax.set_ylabel(row, rotation=90, size='large')

    for ax, col in zip(axs[0], scenario_descriptors):
        ax.set_title(col, size='large')

    for i, model in enumerate(models):
        for j, (scenario1, scenario2) in enumerate(scenarios):
            ax = axs[i, j]
            # Use MaxNLocator to automatically determine the number of ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=2))

            # Use FuncFormatter to format the tick labels
            def format_year(x, pos):
                return str(int(2022 + 2 * x))

            ticks = range(0, 15)
            # Set only every 4th tick to be shown
            ax.set_xticks(ticks[::7])  # This selects every 4th element from the ticks

            # You would still use your custom formatter to format these ticks
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_year))

            # Filter the DataFrame for the current model and scenarios
            scenario_df1 = df.loc[(scenario1, model)] if (scenario1, model) in df.index else pd.DataFrame()
            scenario_df2 = df.loc[(scenario2, model)] if (scenario2, model) in df.index else pd.DataFrame()

            if carb_stor == True or gas_turb == True or BECCS == True or nuclear ==True:
                if not scenario_df1.empty:
                    ax.plot(scenario_df1.index, scenario_df1.values, label=f'2 Years',
                            color='black', linestyle='--' if scenario1.endswith('_1') else '-')
                if not scenario_df2.empty:
                    ax.plot(scenario_df2.index, scenario_df2.values, label=f'Standard Myopic Foresight',
                            color='black', linestyle='--' if scenario2.endswith('_1') else '-')
            else:
                if not scenario_df1.empty:
                    ax.plot(scenario_df1.index, scenario_df1.values/1000, label=f'2 Years',
                            color = 'black', linestyle='--' if scenario1.endswith('_1') else '-')
                if not scenario_df2.empty:
                    ax.plot(scenario_df2.index, scenario_df2.values/1000, label=f'Standard Myopic Foresight',
                            color = 'black', linestyle='--' if scenario2.endswith('_1') else '-')
                ax.axhline(y=carbon_budget, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Adjust tick parameters
            if i < len(models) - 1:  # If not the last row, hide x-axis tick labels
                ax.tick_params(labelbottom=False)
            else:  # Last row, show x-axis tick labels and rotate
                ax.tick_params(axis='x', labelrotation=45)
                ax.tick_params(axis='y', labelrotation=0)

            if j > 0:  # If not the first column, hide y-axis tick labels
                ax.tick_params(labelleft=False)
            else:  # First column, show y-axis tick labels, ensure no rotation
                ax.tick_params(labelleft=True)
                ax.set_ylabel('', rotation=90)  # Explicitly mention no rotation

            # Optionally, adjust legend visibility
            if i == 0 and j == 2:
                if four_models == True:
                    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.07, 1.17), borderaxespad=1.5,
                                    title='Operation Foresight:')
                else:
                    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.07, 1.15), borderaxespad=1.5,
                                    title='Operation Foresight:')
                title_text = leg.get_title()
                title_text.set_ha("left")

    if emissions == True:
        fig.text(0.05, 0.5, 'Cumulative Carbon Emissions [Gt CO₂]', va='center', ha='center', rotation='vertical',
                 fontsize='large')
    elif carb_stor == True:
        fig.text(0.05, 0.5, 'Carbon Storage Capacity [kt CO₂/hour]', va='center', ha='center', rotation='vertical',
                 fontsize='large')
    elif gas_turb == True:
        fig.text(0.05, 0.5, 'Natural Gas Turbine Capacity [units]', va='center', ha='center', rotation='vertical',
                 fontsize='large')
    elif BECCS == True:
        fig.text(0.05, 0.5, 'BECCS Capacity [units]', va='center', ha='center', rotation='vertical',
                 fontsize='large')
    elif nuclear ==True:
        fig.text(0.05, 0.5, 'Nuclear Capacity [GW]', va='center', ha='center', rotation='vertical',
                 fontsize='large')
    else:
        fig.text(0.05, 0.5, 'Renewable\nEl. Gen. Capacity [TW]', va='center', ha='center', rotation='vertical', fontsize='large')

    plt.tight_layout()

    plt.subplots_adjust(left=0.1, hspace=0, wspace=0, right=0.65)  # Adjust 'right' as needed

    # Add labels to the right
    for i, ax in enumerate(axs[:, -1]):  # Iterate over the rightmost column of axes
        fig.text(1, ax.get_position().y0 + ax.get_position().height / 2, model_descriptors[i],
                 va='center', ha='left', rotation=0, fontsize='large')


    # Modify the x position slightly left from the edge to ensure visibility within the figure
    for i, ax in enumerate(axs[:, -1]):  # Iterate over the rightmost column of axes
        fig.text(0.66, ax.get_position().y0 + ax.get_position().height / 2, model_descriptors[i],
                 va='center', ha='left', rotation=90, fontsize='large', clip_on=False)

    # Show the plot
    plt.show(block=True)

def calculate_renewable_percentage(df):
    # Calculate the total and renewable sums across each column for each scenario
    total_sum = df.groupby('scenario').sum()
    renewable_sum = df[df['category'] == 'renewable'].groupby('scenario').sum()

    renewable_sum = renewable_sum.drop(columns='category')
    total_sum = total_sum.drop(columns='category')
    # Calculate the percentage of renewable
    percentage_renewable = (renewable_sum / total_sum) * 100
    return percentage_renewable

def rename_technology(tech):
    suffixes = ["_boiler", "_boiler_DH", "_plant", "_turbine"]
    for suffix in suffixes:
        if tech.endswith(suffix):
            return tech[:-len(suffix)]
    return tech  # Return the original string if no suffixes match

def sum_series_in_nested_dict(d, accumulator=None):
    """
    Recursively find all pandas Series in a nested dictionary and sum them up.

    Parameters:
    - d: The nested dictionary to process.
    - accumulator: Accumulates the sum of all pandas Series found. If None, initializes a new Series.

    Returns:
    - A pandas Series representing the sum of all pandas Series found in the dictionary.
    """
    if accumulator is None:
        accumulator = pd.Series(dtype='float64')

    for key, value in d.items():
        if isinstance(value, pd.Series):
            accumulator = accumulator.add(value, fill_value=0)
        elif isinstance(value, dict):
            accumulator = sum_series_in_nested_dict(value, accumulator)

    return accumulator


def normalize_by_scenario(df):
    normalized_dfs = []
    # Loop through each unique scenario
    for scenario, group_df in df.groupby(level='scenario'):
        # Calculate the total sum of each column for the current scenario
        total_sum = group_df.sum()
        normalized_df = group_df / total_sum
        normalized_dfs.append(normalized_df)

    normalized_df_final = pd.concat(normalized_dfs)

    return normalized_df_final


def format_string(input_string):
    # Step 1: Replace underscores with spaces
    formatted_string = input_string.replace("_", " ")

    # Step 2 & 3: Split into words and capitalize the first letter of each
    words = formatted_string.split()
    capitalized_words = [word.capitalize() for word in words]

    # Step 4: Join the words back into a string
    result = " ".join(capitalized_words)

    return result

def flow_converstion_output_percent_plot():
    print("getting all sector emissions")
    if model_name.endswith("_init"):
        e_15_1, em_15_1 = r.get_all_sector_emissions("scenario_")
        e_15_15, em_15_15 = r.get_all_sector_emissions("scenario_1")
    elif model_name.endswith("_15"):
        e_15_1, em_15_1 = r.get_all_sector_emissions("scenario_")
        e_15_15, em_15_15 = r.get_all_sector_emissions("scenario_14")
    else:
        raise NotImplementedError("Check the scenarios and model chosen")
    el_emissions_15_15 = sum_series_in_nested_dict(e_15_15['electricity'])
    heat_emissions_15_15 = sum_series_in_nested_dict(e_15_15['heat'])
    el_emissions_15_1 = sum_series_in_nested_dict(e_15_1['electricity'])
    heat_emissions_15_1 = sum_series_in_nested_dict(e_15_1['heat'])

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
    df_output = indexmapping(df_output)
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
    carbon_budget = 12232.4
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
        if scenario == '15_1':
            position_15_1 = position
            y_value_15_1 = value
            break

    if True:
        fig, axs = plt.subplots(3, 2, figsize=(9, 13))  # Creating a 2x2 grid of subplots

        # Top-left plot (Emissions delta for electricity)    (scenario_emissions_delta_el.columns
        axs[1, 0].plot(el_emissions_15_15.index, el_emissions_15_15,
                       label='Perfect Foresight', color='black', linestyle='-', zorder=2)
        axs[1, 0].plot(el_emissions_15_1.index, el_emissions_15_1,
                       label='IF:30yr,  OF:2yr', color='black', linestyle=':', zorder=2) # 'IF:30a,  OF:2a'
        axs[1, 0].scatter(position_15_1, el_emissions_15_1[position_15_1],
                          color='red', marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[1, 0].set_title('Electricity Generation Emissions')
        axs[1, 0].set_ylabel('Annual Emissions [Mt CO$_{2}$]', fontsize=12)
        axs[1, 0].grid(True, zorder=1)

        # Top-right plot (Emissions delta for heat)  scenario_emissions_delta_heat.sum()
        axs[1, 1].plot(heat_emissions_15_15.index, heat_emissions_15_15,
                       label='Perfect Foresight', color='black', linestyle='-', zorder=2)
        axs[1, 1].plot(heat_emissions_15_1.index, heat_emissions_15_1,
                       label='IF:30yr,  OF:2yr', color='black', linestyle=':', zorder=2) # 'IF:30a,  OF:2a'
        axs[1, 1].scatter(position_15_1, heat_emissions_15_1[position_15_1], color='red',
                          marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[1, 1].set_title('Heat Generation Emissions')
        axs[1, 1].set_ylabel('Annual Emissions [Mt CO$_{2}$]', fontsize=12)
        axs[1, 1].grid(True, zorder=1)

        # Bottom-left plot (Percentage renewable electricity)
        axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['15_1'],
                       label='IF:30yr,  OF:2yr', linestyle=':', color='black', zorder=2) # 'IF:30a,  OF:2a'
        axs[0, 0].plot(percentage_renewable_electricity.columns, percentage_renewable_electricity.loc['15_15'],
                       label='Perfect Foresight', linestyle='-', color='black', zorder=2)
        axs[0, 0].scatter(position_15_1, percentage_renewable_electricity.loc['15_1'][position_15_1], color='red',
                          marker='x',
                          label='Carbon Budget in OF', zorder=3)
        axs[0, 0].set_title('Electricity from Renewable Generation')
        axs[0, 0].set_ylabel('Renewable Electricity [%]', fontsize=12)
        axs[0, 0].grid(True, zorder=1)

        # Bottom-right plot (Percentage renewable heat)
        axs[0, 1].plot(percentage_renewable_electricity.columns, percentage_renewable_heat.loc['15_1'],
                       label='IF:30yr,  OF:2yr', linestyle=':', color='black', zorder=2) # 'IF:30a,  OF:2a'
        axs[0, 1].plot(percentage_renewable_electricity.columns, percentage_renewable_heat.loc['15_15'], label='15_15',
                       linestyle='-', color='black', zorder=2)
        axs[0, 1].scatter(position_15_1, percentage_renewable_heat.loc['15_1'][position_15_1], color='red', marker='x',
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
                    positions = np.arange(0, 15) - width / 2 if scenario == '15_1' else np.arange(0, 15) + width / 2
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
                            axs[2, 0].text(label_x_position+2.5, label_y_position, 'Perfect Foresight', ha='center')
                        else:
                            # Calculate the x and y positions for the label and arrow
                            x = positions[2]
                            y = bottom[2]
                            label_y_position = y + (0.16 * y)
                            label_x_position = x + 0.3
                            # Draw an arrow pointing to the bar
                            axs[2, 0].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                            # Add text label slightly above and to the right/left depending on position
                            axs[2, 0].text(label_x_position + 2.4, label_y_position + (0.01 * y), 'IF:30yr,  OF:2yr', ha='center') #  'IF:30a,  OF:2a'
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
                    positions = np.arange(0, 15) - width / 2 if scenario == '15_1' else np.arange(0, 15) + width / 2
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
                            if model_name.endswith("_15"):
                                # Draw an arrow pointing to the bar
                                axs[2, 1].plot([x, label_x_position], [y, label_y_position + (0.05 * y)], color="black", linewidth=1)
                                # Add text label slightly above and to the right/left depending on position
                                axs[2, 1].text(label_x_position + 2.5, label_y_position + (0.05 * y), 'Perfect Foresight',
                                               ha='center')
                            else:
                                # Draw an arrow pointing to the bar
                                axs[2, 1].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                                # Add text label slightly above and to the right/left depending on position
                                axs[2, 1].text(label_x_position+2.5, label_y_position, 'Perfect Foresight', ha='center')
                        else:
                            # Calculate the x and y positions for the label and arrow
                            x = positions[2]
                            y = bottom[2]
                            label_y_position = y + (0.06 * y)
                            label_x_position = x + 0.3
                            # Draw an arrow pointing to the bar
                            axs[2, 1].plot([x, label_x_position], [y, label_y_position], color="black", linewidth=1)
                            # Add text label slightly above and to the right/left depending on position
                            axs[2, 1].text(label_x_position + 2.4, label_y_position + (0.01 * y), 'IF:30yr,  OF:2yr', ha='center') # 'IF:30a,  OF:2a'
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
                        if model_name.endswith("_init"):
                            ax.set_ylim(0, 3000)
                            yticks_major = np.arange(0, 3000, 250)
                        elif model_name.endswith("_15"):
                            ax.set_ylim(0, 3100)
                            yticks_major = np.arange(0, 3000, 250)
                        else:
                            raise NotImplementedError("Check the scenarios and model chosen")
                    else:
                        if model_name.endswith("_init"):
                            ax.set_ylim(0, 1700)
                            yticks_major = np.arange(0, 1700, 250)
                        elif model_name.endswith("_15"):
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




def emissions_techs_plot():
    print("getting all sector emissions")
    if model_name.endswith("_init"):
        e_15_1, em_15_1 = r.get_all_sector_emissions("scenario_")
        e_15_15, em_15_15 = r.get_all_sector_emissions("scenario_1")
    elif model_name.endswith("_15"):
        e_15_1, em_15_1 = r.get_all_sector_emissions("scenario_")
        e_15_15, em_15_15 = r.get_all_sector_emissions("scenario_14")
    else:
        raise NotImplementedError("Check the scenarios and model chosen")
    el_emissions_15_15 = sum_series_in_nested_dict(e_15_15['electricity'])
    heat_emissions_15_15 = sum_series_in_nested_dict(e_15_15['heat'])
    el_emissions_15_1 = sum_series_in_nested_dict(e_15_1['electricity'])
    heat_emissions_15_1 = sum_series_in_nested_dict(e_15_1['heat'])

    tech_em_el_15_15 = em_15_15.loc["electricity"]
    tech_em_el_15_15.loc['natural_gas'] += tech_em_el_15_15.loc['lng']
    tech_em_el_15_15 = tech_em_el_15_15.drop('lng')
    tech_em_el_15_15 = tech_em_el_15_15.rename(index={'tech': 'CCS'})

    tech_em_el_15_1 = em_15_1.loc["electricity"]
    tech_em_el_15_1.loc['natural_gas'] += tech_em_el_15_1.loc['lng']
    tech_em_el_15_1 = tech_em_el_15_1.drop('lng')
    tech_em_el_15_1 = tech_em_el_15_1.rename(index={'tech': 'CCS'})

    order = [ "CCS", "hard_coal", "lignite", "oil", "natural_gas", "waste"]

    # Reorder tech_em_el_15_1
    complete_order = order + [tech for tech in tech_em_el_15_1.index if tech not in order]
    tech_em_el_15_1['order'] = pd.Categorical(tech_em_el_15_1.index, categories=complete_order, ordered=True)
    tech_em_el_15_1 = tech_em_el_15_1.sort_values('order')

    # Reorder tech_em_el_15_15
    tech_em_el_15_15['order'] = pd.Categorical(tech_em_el_15_15.index, categories=complete_order, ordered=True)
    tech_em_el_15_15 = tech_em_el_15_15.sort_values('order')

    # Remove the temporary 'order' column
    tech_em_el_15_1.drop(columns='order', inplace=True)
    tech_em_el_15_15.drop(columns='order', inplace=True)

    def colors(index):
        if index == "hard_coal":
            color = "black"
        elif index == "nuclear":
            color = "#eed202"
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

    colors_1 = [colors(idx) for idx in tech_em_el_15_1.index]
    colors_15 = [colors(idx) for idx in tech_em_el_15_15.index]

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    # Width of the bars in the bar plot
    width = 0.3

    # Locations for the groups on the x-axis
    x = range(len(tech_em_el_15_1.columns))

    # Plotting data from the first dataframe
    tech_em_el_15_1.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=colors_1, ax=ax, position=0.5, width=width)

    # Plotting data from the second dataframe
    tech_em_el_15_15.T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, color=colors_15, ax=ax, position=-0.5, width=width)

    # Setting x-axis labels
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(tech_em_el_15_1.columns)

    plt.title('Carrier Emissions over Time')
    plt.ylabel('Anual Emissions [Gt CO₂]')

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

    plt.show()


def utilization_rate_plot():
    df_output = r.get_total("flow_conversion_output")
    df_maxload = r.get_total("max_load")
    df_capacity = r.get_total("capacity")

    df_output.reset_index(inplace=True)
    df_maxload.reset_index(inplace=True)
    df_capacity.reset_index(inplace=True)

    df_output.rename(columns={'level_0': 'scenario'}, inplace=True)
    df_output.set_index(['scenario', 'technology', 'carrier', 'node'], inplace=True)
    df_maxload.rename(columns={'level_0': 'scenario'}, inplace=True)
    df_maxload.rename(columns={'location': 'node'}, inplace=True)
    df_maxload.set_index(['scenario', 'technology', 'capacity_type', 'node'], inplace=True)
    df_capacity.rename(columns={'level_0': 'scenario'}, inplace=True)
    df_capacity.rename(columns={'location': 'node'}, inplace=True)
    df_capacity.set_index(['scenario', 'technology', 'capacity_type', 'node'], inplace=True)

    df_output = indexmapping(df_output)
    df_maxload = indexmapping(df_maxload)
    df_capacity = indexmapping(df_capacity)

    df_output = df_output.loc[df_output.index.get_level_values('scenario').isin(desired_scenarios)]
    df_maxload = df_maxload.loc[df_maxload.index.get_level_values('scenario').isin(desired_scenarios)]
    df_capacity = df_capacity.loc[df_capacity.index.get_level_values('scenario').isin(desired_scenarios)]

    df_output = df_output[~df_output.index.get_level_values('carrier').isin(['carbon', 'natural_gas', 'natural_gas_industry'])]
    df_output = df_output.rename(index={'district_heat': 'heat'}, level='carrier')
    df_maxload = df_maxload[df_maxload.index.get_level_values('capacity_type') != 'energy']
    df_capacity = df_capacity[df_capacity.index.get_level_values('capacity_type') != 'energy']

    df_output = df_output.droplevel('carrier')
    df_maxload = df_maxload.droplevel('capacity_type')
    df_capacity = df_capacity.droplevel('capacity_type')

    df_output = df_output.groupby(['scenario', 'technology']).sum()
    df_product = df_capacity * df_maxload
    df_grouped_sum = df_product.groupby(['scenario', 'technology']).sum()

    new_index = df_output.index.to_frame()
    new_index['technology'] = new_index['technology'].map(
        lambda x: x[:-3] if x.endswith('_DH') else (x[:-4] if x.endswith('_CCS') else x))
    # Convert the DataFrame back to a MultiIndex
    new_multi_index = pd.MultiIndex.from_frame(new_index)
    df_output.index = new_multi_index
    df_output = df_output.groupby(level=df_output.index.names).sum()

    new_index = df_grouped_sum.index.to_frame()
    new_index['technology'] = new_index['technology'].map(
        lambda x: x[:-3] if x.endswith('_DH') else (x[:-4] if x.endswith('_CCS') else x))
    # Convert the DataFrame back to a MultiIndex
    new_multi_index = pd.MultiIndex.from_frame(new_index)
    df_grouped_sum.index = new_multi_index
    df_grouped_sum = df_grouped_sum.groupby(level=df_grouped_sum.index.names).sum()

    heat = ['biomass_boiler', 'electrode_boiler', 'heat_pump', 'hard_coal_boiler',
            'natural_gas_boiler', 'oil_boiler', 'waste_boiler']

    nuclear = ['nuclear']

    electricity = ['biomass_plant', 'photovoltaics', 'reservoir_hydro', 'run-of-river_hydro',
                  'wind_offshore', 'wind_onshore', 'hard_coal_plant', 'lignite_coal_plant',
                   'natural_gas_turbine', 'nuclear', 'oil_plant', 'waste_plant', 'pumped_hydro']

    renewables = ['biomass_boiler', 'electrode_boiler', 'heat_pump','biomass_plant', 'photovoltaics',
                  'reservoir_hydro', 'run-of-river_hydro', 'wind_offshore', 'wind_onshore','pumped_hydro']
    conventional = ['hard_coal_boiler', 'natural_gas_boiler', 'oil_boiler', 'waste_boiler',
                    'hard_coal_plant', 'lignite_coal_plant', 'natural_gas_turbine', 'nuclear',
                    'oil_plant', 'waste_plant']

    df_heat_out = df_output.copy()
    df_electricity_out = df_output.copy()
    df_nuclear_out = df_output.copy()
    df_heat_out = df_heat_out[df_heat_out.index.get_level_values('technology').isin(heat)]
    df_electricity_out = df_electricity_out[df_electricity_out.index.get_level_values('technology').isin(electricity)]
    df_nuclear_out = df_nuclear_out[df_nuclear_out.index.get_level_values('technology').isin(nuclear)]

    df_heat_max = df_grouped_sum.copy()
    df_electricity_max = df_grouped_sum.copy()
    df_nuclear_max = df_grouped_sum.copy()
    df_heat_max = df_heat_max[df_heat_max.index.get_level_values('technology').isin(heat)]
    df_electricity_max = df_electricity_max[df_electricity_max.index.get_level_values('technology').isin(electricity)]
    df_nuclear_max = df_nuclear_max[df_nuclear_max.index.get_level_values('technology').isin(nuclear)]

    # Create a new index by modifying the 'technology' level
    new_index = df_heat_out.index.to_frame()
    new_index['technology'] = new_index['technology'].map(
        lambda x: 'renewable' if x in renewables else ('conventional' if x in conventional else x))
    # Convert the DataFrame back to a MultiIndex
    new_multi_index = pd.MultiIndex.from_frame(new_index)
    df_heat_out.index = new_multi_index

    # Create a new index by modifying the 'technology' level
    new_index = df_electricity_out.index.to_frame()
    new_index['technology'] = new_index['technology'].map(
        lambda x: 'renewable' if x in renewables else ('conventional' if x in conventional else x))
    # Convert the DataFrame back to a MultiIndex
    new_multi_index = pd.MultiIndex.from_frame(new_index)
    df_electricity_out.index = new_multi_index

    # Create a new index by modifying the 'technology' level
    new_index = df_heat_max.index.to_frame()
    new_index['technology'] = new_index['technology'].map(
        lambda x: 'renewable' if x in renewables else ('conventional' if x in conventional else x))
    # Convert the DataFrame back to a MultiIndex
    new_multi_index = pd.MultiIndex.from_frame(new_index)
    df_heat_max.index = new_multi_index

    # Create a new index by modifying the 'technology' level
    new_index = df_electricity_max.index.to_frame()
    new_index['technology'] = new_index['technology'].map(
        lambda x: 'renewable' if x in renewables else ('conventional' if x in conventional else x))
    # Convert the DataFrame back to a MultiIndex
    new_multi_index = pd.MultiIndex.from_frame(new_index)
    df_electricity_max.index = new_multi_index

    df_heat_out = df_heat_out.groupby(['scenario', 'technology']).sum()
    df_electricity_out = df_electricity_out.groupby(['scenario', 'technology']).sum()

    df_heat_max = df_heat_max.groupby(['scenario', 'technology']).sum()
    df_electricity_max = df_electricity_max.groupby(['scenario', 'technology']).sum()

    util_rate_heat = (df_heat_out / df_heat_max) * 100
    util_rate_el = (df_electricity_out / df_electricity_max) * 100
    util_rate_nuclear = (df_nuclear_out / df_nuclear_max) * 100









    # Plotting things
    # Create a figure with two subplots (one above the other)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharey=True, sharex=True)
    plt.rcParams.update({'font.size': 12})

    # Define colors and line styles
    colors = {
        'renewable_el': 'gold',
        'renewable_heat': 'indianred',
        'conventional_el': 'goldenrod',
        'conventional_heat': 'firebrick'
    }

    line_styles = {
        '15_15': '-',
        '15_1': '--'
    }

    # Subplot 1: Renewable technologies
    for scenario in ['15_15', '15_1']:
        ax[0].plot(util_rate_el.loc[(scenario, 'renewable')].values.T,
                   label=f'Electricity {scenario} Renewable',
                   color=colors['renewable_el'], linestyle=line_styles[scenario])
        ax[0].plot(util_rate_heat.loc[(scenario, 'renewable')].values.T,
                   label=f'Heat {scenario} Renewable',
                   color=colors['renewable_heat'], linestyle=line_styles[scenario])
    ax[0].set_ylabel('Utilisation Rate [%]', fontsize=14)


    # Subplot 2: Conventional technologies
    for scenario in ['15_15', '15_1']:
        ax[1].plot(util_rate_el.loc[(scenario, 'conventional')].values.T,
                   label=f'Electricity {scenario} Conventional',
                   color=colors['conventional_el'], linestyle=line_styles[scenario])
        ax[1].plot(util_rate_heat.loc[(scenario, 'conventional')].values.T,
                   label=f'Heat {scenario} Conventional',
                   color=colors['conventional_heat'], linestyle=line_styles[scenario])
    ax[1].set_ylabel('Utilisation Rate [%]', fontsize=14)

    ax[0].set_ylim(0, 100)
    ax[1].set_ylim(0, 100)
    newticklabels = [0, 20, 40, 60, 80, 100]

    ax[0].set_yticklabels(newticklabels, fontsize=14)
    ax[1].set_yticklabels(newticklabels, fontsize=14)
    ax[0].tick_params(axis='x', direction='in')

    # Define legend elements
    legend_elements_ren = [
        patches.Patch(facecolor='gold', edgecolor='gold', label='Electricity'),
        patches.Patch(facecolor='indianred', edgecolor='indianred', label='Heat'),
        Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Perfect Foresight'),
        Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='IF: 30yr   OF: 2yr')
    ]

    legend_elements_con = [
        patches.Patch(facecolor='goldenrod', edgecolor='goldenrod', label='Electricity'),
        patches.Patch(facecolor='firebrick', edgecolor='firebrick', label='Heat'),
        Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Perfect Foresight'),
        Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='IF: 30yr   OF: 2yr')
    ]

    # Adding legends to subplots with custom handles and bold titles
    ax[0].legend(handles=legend_elements_ren, title='Renewable', title_fontsize=14, fontsize=12)
    ax[1].legend(handles=legend_elements_con, title='Conventional', title_fontsize=14, fontsize=12)

    current_ticks = ax[1].get_xticks()  # Assume both subplots should have the same x-axis ticks
    new_labels = [2022 + 2 * int(tick) for tick in current_ticks]

    ax[1].set_xticklabels(new_labels, fontsize=14)
    # Adjust layout and show plot
    plt.subplots_adjust(hspace=0)  # Adjust horizontal space (try values between 0.1 and 0.5)
    plt.show()

def feedback_plot_progression_with_techs_heat():
    global model_name
    print("plotting feedback_plot_progression")
    if type(model_name) is not str:
        model_name = str(model_name)
    r, config = read_in_results(model_name)

    projections =  ['0_inv', '3_inv', '4_inv', '14_inv'] # '2_inv','6_inv',

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
    carbon_budget = 12.2324
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
    ax1.set_ylim(0, 14.5)

    # Legend and title
    # generate figure title:
    fig_title = ""
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




if __name__ == "__main__":
    if False:
        run_unused_plots()

    if True:
        # model_name = "PI_small_drastic_coal_production_phaseout"
        # model_name = ("PI_small")
        # model_name = ("PI_small_drastic_coal_capacity_phaseout")
        model_name = ("PI_small_test_cap_constraint")

        r, config = read_in_special_results(model_name)
        print("read in")

    # costs plot: comparison of scenarios
    if False:
        # model_name = ["cons_nolead_init", "cons_nolead_init_2"]
        model_name = ["post_changes_to_model/cons_lead_15", "post_changes_to_model/cons_lead_5to7"]
        desired_scenarios = ['7_7', '15_15', '7_4', '15_4']
        comparative_costs_over_time()

    # difference in operation plot
    if False:
        # model_name = "cons_nolead_init"
        model_name = "post_changes_to_model/cons_lead_15"
        desired_scenarios = ['15_15', '15_1']
        r, config = read_in_special_results(model_name)
        flow_converstion_output_percent_plot()

    # difference in operation plot
    if False:
        model_name = "post_changes_to_model/cons_lead_15"
        desired_scenarios = ['15_15', '15_1']
        r, config = read_in_special_results(model_name)
        emissions_techs_plot()

    # util_rates
    if False:
        model_name = "post_changes_to_model/cons_lead_15"
        desired_scenarios = ['15_15', '15_1']
        r, config = read_in_results(model_name)
        utilization_rate_plot()

    # feedback to investent plot_
    if True:
        if True:
            model_name =  "post_changes_to_model/cons_lead_15"
        else:
            model_name = "cons_nolead_15"

        desired_scenarios = ['15_1']
        assert len(desired_scenarios) == 1, "too many scenarios for this plot"
        feedback_plot_progression_with_techs()

    # feedback to investent plot heat
    if True:
        if True:
            model_name =  "post_changes_to_model/cons_lead_15"
        else:
            model_name = "cons_nolead_15"

        desired_scenarios = ['15_1']
        assert len(desired_scenarios) == 1, "too many scenarios for this plot"
        feedback_plot_progression_with_techs_heat()

    # comparison of construction times/ no construction times



    # stranded assets plot



    # sensitivity analysis plot
    if False:
        four_models = True
        carb_stor = False
        gas_turb = False
        BECCS = False
        nuclear = False
        # desired_scenarios = ['3_1', '3_3', '7_7', '15_15', '7_1', '15_1']
        desired_scenarios = ['3_1', '3_3', '4_1', '4_4', '7_7', '15_15', '7_1', '15_1']
        if False:
            model_name = ["cons_nolead_1to4", "cons_nolead_init", "cons_lead_1to4",
                          "cons_lead_init", "lesscons_lead_1to4", "lesscons_lead_init",
                          "varcons_lead_1to4", "varcons_lead_init"]
        else:
            model_name = ["cons_nolead_1to4", "cons_nolead_init", "cons_nolead_init_2", "post_changes_to_model/cons_lead_1to4",
                          "post_changes_to_model/cons_lead_init", "post_changes_to_model/lesscons_lead_1to4",
                          "post_changes_to_model/lesscons_lead_init", "post_changes_to_model/varcons_lead_1to4",
                          "post_changes_to_model/varcons_lead_init_2", "post_changes_to_model/lesscons_lead_init_2",
                          "post_changes_to_model/cons_lead_init_2", "post_changes_to_model/varcons_lead_init", ]
        sens_analysis_plot(model_name, four_models, carb_stor, gas_turb, BECCS, nuclear)

    # stairs plot:
    if False:
        model_name = ["cons_nolead_1to4", "cons_nolead_5to7"]
        single_model = False
        if single_model  == True:
            r, config = read_in_results(model_name)
        else:
            r, config = read_in_results(model_name[0])
        # define parameters
        features = ["cumulative_emissions"]  # "costs", "cumulative_emissions", "capacity_addition"
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
        empty_plot = False
        if multi_model is not True:
            pass
        elif False:
            model_name = ["cons_nolead_1to4", "cons_nolead_5to7", "cons_nolead_8to10",
                          "cons_nolead_11", "cons_nolead_12", "cons_nolead_13", "cons_nolead_14",
                          "cons_nolead_15"]  # ,  "some other name", "cons_lead_15", "cons_lead_8to10", "cons_lead_11", "cons_lead_15"
        elif False:
            model_name = ["cons_lead_1to4", "cons_lead_5to7", "cons_lead_8to10", "cons_lead_11",
                          "cons_lead_12", "cons_lead_13", "cons_lead_14", "cons_lead_15"]
        elif True:
            model_name = ["post_changes_to_model/cons_lead_1to4", "post_changes_to_model/cons_lead_5to7", "post_changes_to_model/cons_lead_8to10", "post_changes_to_model/cons_lead_11",
                          "post_changes_to_model/cons_lead_12", "post_changes_to_model/cons_lead_13", "post_changes_to_model/cons_lead_14", "post_changes_to_model/cons_lead_15"]
        else:
            model_name = ["cons_lead_1to4"]  # ,  "some other name", "cons_lead_15", "cons_lead_8to10", "cons_lead_11", "cons_lead_15"

        # generate the stairs plot
        for feature in features:
            stair_plot(feature, model_name, r, technology=technologies, capacity_type=capacity_type, empty_plot=empty_plot)
