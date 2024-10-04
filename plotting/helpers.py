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
from matplotlib.patches import Rectangle


from zen_garden.postprocess.results.results import Results
from zen_garden.model.default_config import Config,Analysis,Solver,System


# read in results
def read_in_results(model_name):
    # data_path = "/Users/lukashegner/PycharmProjects/ZEN-garden_fork/data/"
    # outputs_path = data_path + "outputs"
    data_path = "/Volumes/Elements10T/RRE_Myopic_Operation_Paper/Model_Outputs/"
    outputs_path = data_path + "outputs"
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
    # data_path = "/Users/lukashegner/PycharmProjects/ZEN-garden_fork/data/"
    # outputs_path = data_path + "outputs"
    data_path = "/Volumes/Elements10T/RRE_Myopic_Operation_Paper/Model_Outputs/"
    outputs_path = data_path + "outputs"
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

def indexmapping(df, special_model_name=None):
    global model_name
    if special_model_name is not None:
        model_name = special_model_name

    if model_name.startswith('PC_ct_') or model_name.endswith('_test'):
        # Define a dictionary mapping the current index values to the desired ones
        # PC_ct_vartdr_w_pass_tra_ETS1and2
        index_mapping = {
            'scenario_': '15_1',
            'scenario_1': '15_5',
            'scenario_2': '5_1',
            'scenario_3': '5_5'
        }
    else:
        print("dataset not defined correctly")
        raise ValueError("Invalid dataset configuration: {}".format(model_name))

    df = df.rename(index=index_mapping)
    return df

# Function to sort the DataFrame rows based on the index
def sort_dataframe(data):
    # Define a custom sort key function with a differentiator
    def sort_key(val):
        val_str = str(val)  # Convert the index value to a string
        if '_inv' in val_str:
            num_part = int(val_str.split('_')[0])  # Extract the numeric part
            inv_part = 0  # Sort "_inv" rows before non-inv rows
        else:
            num_part = int(val_str)  # Extract the numeric part
            inv_part = 1  # Non-inv rows get a different sort order
        return (num_part, inv_part)

    if isinstance(data, pd.Series):
        # If it's a Series, sort the index
        sorted_index = sorted(data.index, key=sort_key)
        return data.loc[sorted_index]
    elif isinstance(data, pd.DataFrame):
        # Determine if sorting needs to be applied to rows or columns
        if any('_inv' in str(i) for i in data.index) or data.index.dtype == 'object':
            # Sort rows if index contains "_inv" or is object type (may contain mixed types)
            return data.loc[sorted(data.index, key=sort_key)]
        elif any('_inv' in str(i) for i in data.columns) or data.columns.dtype == 'object':
            # Sort columns if columns contain "_inv" or are object type (may contain mixed types)
            return data.loc[:, sorted(data.columns, key=sort_key)]
        else:
            # Return the DataFrame as-is if no sorting is needed
            return data
    else:
        raise TypeError("Input should be a pandas DataFrame or Series.")


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

def update_indices(data):
    """
    Recursively traverse the nested dictionary and update the index of all pandas Series.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            update_indices(value)  # Recursive call for nested dictionaries
    elif isinstance(data, pd.Series):
        data.index = 2022 + (data.index * 2)  # Update the index of the Series

def rename_columns(columns):
    return [2022 + (col * 2) if isinstance(col, int) else col for col in columns]

def sector_proportion(df):
    renewable_technologies = [
        "photovoltaics", "wind_onshore", "run-of-river_hydro", "reservoir_hydro",
        "wind_offshore", "biomass_plant", "heat_pump",
        "biomass_boiler", "electrode_boiler", "heat_pump_DH", "biomass_boiler_DH",
        "electrode_boiler_DH", "BEV"
    ]

    renewable_technologies_el = [
        "photovoltaics", "wind_onshore", "run-of-river_hydro", "reservoir_hydro",
        "wind_offshore", "biomass_plant"
    ]

    renewable_technologies_heat = [
        "heat_pump", "biomass_boiler", "electrode_boiler", "heat_pump_DH", "biomass_boiler_DH",
        "electrode_boiler_DH"
    ]

    renewable_technologies_tra = [
        "BEV"
    ]


def rename_technology(tech):
    suffixes = ["_boiler", "_boiler_DH", "_plant", "_turbine"]
    for suffix in suffixes:
        if tech.endswith(suffix):
            return tech[:-len(suffix)]
    return tech  # Return the original string if no suffixes match


def calculate_renewable_percentage(df):
    # Calculate the total and renewable sums across each column for each scenario
    total_sum = df.groupby('scenario').sum()
    renewable_sum = df[df['category'] == 'renewable'].groupby('scenario').sum()

    renewable_sum = renewable_sum.drop(columns='category')
    total_sum = total_sum.drop(columns='category')
    # Calculate the percentage of renewable
    percentage_renewable = (renewable_sum / total_sum) * 100
    return percentage_renewable

def format_string(input_string):
    # Step 1: Replace underscores with spaces
    formatted_string = input_string.replace("_", " ")

    # Step 2 & 3: Split into words and capitalize the first letter of each
    words = formatted_string.split()
    capitalized_words = [word.capitalize() for word in words]

    # Step 4: Join the words back into a string
    result = " ".join(capitalized_words)

    return result

