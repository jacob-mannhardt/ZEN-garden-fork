from dask.array import store
from networkx.algorithms.efficiency_measures import efficiency

from zen_garden.postprocess.results.results import Results
from zen_garden.postprocess.comparisons import compare_model_values, compare_configs, compare_component_values
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def post_compute_storage_level(result):
    """
    Computes the fully resolved storage_level variable from the flow_storage_charge and flow_storage_discharge variables
    (ZEN-garden's Result objects doesn't reconstruct the aggregated storage_level variables)
    """
    storage_level = result.get_full_ts("storage_level").transpose()
    flow_storage_charge = result.get_full_ts("flow_storage_charge").transpose()
    flow_storage_discharge = result.get_full_ts("flow_storage_discharge").transpose()
    self_discharge = result.get_total("self_discharge")
    efficiency_charge = result.get_total("efficiency_charge").squeeze()
    efficiency_discharge = result.get_total("efficiency_discharge").squeeze()
    for ts in range(storage_level.shape[0]-1):
        storage_level.loc[ts+1] = storage_level.loc[ts] * (1-self_discharge) + (efficiency_charge * flow_storage_charge.loc[ts+1] - flow_storage_discharge.loc[ts+1]/efficiency_discharge)
    return storage_level

def post_compute_storage_level_kotzur(result):
    """

    """
    storage_level_inter = result.get_full_ts("storage_level_inter").T
    self_discharge = pd.DataFrame([result.get_total("self_discharge")] * len(storage_level_inter.index.values), columns=result.get_total("self_discharge").index).reset_index(drop=True)
    exponents = storage_level_inter.index.values%24
    storage_level_inter_sd = storage_level_inter * (1-self_discharge)**exponents[:,None]
    storage_level_intra = result.get_full_ts("storage_level_intra").T
    return storage_level_inter_sd + storage_level_intra


rZEN = Results(path='../data/outputs/dummy_model_TSA_ZEN')
rKot = Results(path='../data/outputs/dummy_model_TSA_Kot')
rGab = Results(path='../data/outputs/dummy_model_TSA_Gab')
rOp = Results(path='../data/outputs/operation')

a = 1