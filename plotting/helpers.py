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
    # data_path = "/Users/lukashegner/PycharmProjects/ZEN-garden_fork/data/"
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
        r = PostResult(out_folder)
    return r, config

def indexmapping(df, special_model_name=None):
    global model_name
    if special_model_name is not None:
        model_name = special_model_name

    if model_name.startswith('PC_ct_vartdr_w_pass_tra_') or model_name.startswith('PC_ct_base'):
        # Define a dictionary mapping the current index values to the desired ones
        # PC_ct_vartdr_w_pass_tra_ETS1and2
        index_mapping = {
            'scenario_': '15_1',
            'scenario_1': '15_3',
            'scenario_2': '15_5',
            'scenario_3': '10_1',
            'scenario_4': '10_3',
            'scenario_5': '10_1',
            'scenario_6': '5_1',
            'scenario_7': '5_3',
            'scenario_8': '5_5'
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

