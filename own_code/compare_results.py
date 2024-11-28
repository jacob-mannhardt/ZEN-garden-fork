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


rTest = Results(path='../data/outputs/dummy_model_TSA')
rKot = Results(path='../data/outputs/dummy_model_TSA_kot')
rZEN = Results(path='../data/outputs/dummy_model_TSA_ZEN')


#compare_parameters = compare_model_values([rZEN, rGab], component_type = 'parameter')
#compare_variables = compare_model_values([rZEN, rGab], component_type = 'variable')

plt.figure()
plt.plot(rTest.get_full_ts("flow_storage_charge").T["battery"])
plt.plot(rTest.get_full_ts("flow_storage_discharge").T["battery"])
plt.plot(post_compute_storage_level_kotzur(rTest)["battery"])
plt.plot(rTest.get_full_ts("storage_level_inter").T["battery"])
plt.plot(rTest.get_full_ts("storage_level_intra").T["battery"])
plt.legend(["charge", "discharge", "storage_level", "storage_level_inter", "storage_level_intra"])
plt.show()

rZENUnc.get_full_ts("flow_storage_charge")
rKotUnc.get_full_ts("flow_storage_charge")

storage_level_kot = post_compute_storage_level_kotzur(rKotUnc)
ex_post_ZEN = post_compute_storage_level(rZENUnc)


ex_post_Gab = post_compute_storage_level(rGab)
sl_diff = ex_post_ZEN-rGab.get_full_ts("storage_level").transpose()
sl_diff_Gab = ex_post_Gab-rGab.get_full_ts("storage_level").transpose()

charge_diff = (rZEN.get_full_ts("flow_storage_charge").transpose()-rGab.get_full_ts("flow_storage_charge").transpose())
charge_diff = charge_diff[abs(charge_diff)>1e-3]
charge_diff = charge_diff.stack()

storage_level_rZEN = rZEN.get_full_ts("storage_level").transpose()
storage_level_rGab = rGab.get_full_ts("storage_level").transpose()

plt.figure()
a = 1