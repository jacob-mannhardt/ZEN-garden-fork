from scipy.optimize import anderson

from zen_garden.postprocess.results.results import Results
from zen_garden.postprocess.comparisons import compare_model_values, compare_configs, compare_component_values
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def post_compute_storage_level(result, scenario_name=None):
    """
    Computes the fully resolved storage_level variable from the flow_storage_charge and flow_storage_discharge variables
    (ZEN-garden's Result objects doesn't reconstruct the aggregated storage_level variables)
    """
    # TODO doesn't work for multi year
    if not scenario_name:
        scenario_name = next(iter(result.solution_loader.scenarios))
    storage_level = result.get_full_ts("storage_level", scenario_name=scenario_name).transpose()
    flow_storage_charge = result.get_full_ts("flow_storage_charge", scenario_name=scenario_name).transpose()
    flow_storage_discharge = result.get_full_ts("flow_storage_discharge", scenario_name=scenario_name).transpose()
    self_discharge = result.get_total("self_discharge", scenario_name=scenario_name)
    efficiency_charge = result.get_total("efficiency_charge", scenario_name=scenario_name).squeeze()
    efficiency_discharge = result.get_total("efficiency_discharge", scenario_name=scenario_name).squeeze()
    for ts in range(storage_level.shape[0]-1):
        storage_level.loc[ts+1] = storage_level.loc[ts] * (1-self_discharge) + (efficiency_charge * flow_storage_charge.loc[ts+1] - flow_storage_discharge.loc[ts+1]/efficiency_discharge)
    return storage_level

def post_compute_storage_level_kotzur(result, scenario_name=None):
    """

    """
    if not scenario_name:
        scenario_name = next(iter(result.solution_loader.scenarios))
    storage_level_inter = result.get_full_ts("storage_level_inter", scenario_name=scenario_name).T
    self_discharge = pd.DataFrame([result.get_total("self_discharge", scenario_name=scenario_name)] * len(storage_level_inter.index.values), columns=result.get_total("self_discharge", scenario_name=scenario_name).index).reset_index(drop=True)
    exponents = storage_level_inter.index.values%24
    storage_level_inter_sd = storage_level_inter * (1-self_discharge)**exponents[:,None]
    storage_level_intra = result.get_full_ts("storage_level_intra", scenario_name=scenario_name).T
    return storage_level_inter_sd + storage_level_intra

def sort_scenarios(results):
    rep_settings = {}
    for result in results:
        scenarios = get_scenarios_with_unique_representation_settings(result)
        for scen_name, scen in scenarios.items():
            key = scen.analysis.time_series_aggregation.storageRepresentationMethod + "_" + str(scen.analysis.time_series_aggregation.hoursPerPeriod)
            if key not in rep_settings:
                rep_settings[key] = {scen.system.aggregated_time_steps_per_year: [s for s_n, s in result.solution_loader.scenarios.items() if s.analysis.time_series_aggregation.storageRepresentationMethod
                                          == scen.analysis.time_series_aggregation.storageRepresentationMethod
                                          and s.analysis.time_series_aggregation.hoursPerPeriod
                                          == scen.analysis.time_series_aggregation.hoursPerPeriod
                                     and s.system.aggregated_time_steps_per_year
                                     == scen.system.aggregated_time_steps_per_year]}
            elif scen.system.aggregated_time_steps_per_year not in rep_settings[key].keys():
                rep_settings[key][scen.system.aggregated_time_steps_per_year] = [s for s_n, s in result.solution_loader.scenarios.items() if s.analysis.time_series_aggregation.storageRepresentationMethod
                                          == scen.analysis.time_series_aggregation.storageRepresentationMethod
                                          and s.analysis.time_series_aggregation.hoursPerPeriod
                                          == scen.analysis.time_series_aggregation.hoursPerPeriod
                                     and s.system.aggregated_time_steps_per_year
                                     == scen.system.aggregated_time_steps_per_year]
            else:
                rep_settings[key][scen.system.aggregated_time_steps_per_year].extend(
                    [s for s_n, s in result.solution_loader.scenarios.items() if s.analysis.time_series_aggregation.storageRepresentationMethod
                    == scen.analysis.time_series_aggregation.storageRepresentationMethod and s.analysis.time_series_aggregation.hoursPerPeriod
                    == scen.analysis.time_series_aggregation.hoursPerPeriod and s.system.aggregated_time_steps_per_year
                    == scen.system.aggregated_time_steps_per_year])
    return rep_settings

def get_scenarios_with_unique_representation_settings(r):
    scenarios = {}
    for s_n, s in r.solution_loader.scenarios.items():
        if s_n == "scenario_":
            continue
        if all([True  if (val.analysis.time_series_aggregation.storageRepresentationMethod
                           != s.analysis.time_series_aggregation.storageRepresentationMethod
                           or val.analysis.time_series_aggregation.hoursPerPeriod
                           != s.analysis.time_series_aggregation.hoursPerPeriod
                           or val.system.aggregated_time_steps_per_year
                           != s.system.aggregated_time_steps_per_year)
                else False for val in scenarios.values()]):
            scenarios[s_n] = s
    return scenarios

def average_benchmarking_values(rep_settings):
    for rs_name, rss_list in rep_settings.items():
        avg_scens = []
        for ag_ts, rss_identical in rss_list.items():
            avg_scen = next(iter(rss_identical))
            for scen in rss_identical:
                if scen.name != avg_scen.name:
                    for key, value in scen.benchmarking.items():
                        avg_scen.benchmarking[key] += value
            for key, value in avg_scen.benchmarking.items():
                avg_scen.benchmarking[key] = value / len(rss_identical)
            avg_scens.append(avg_scen)
        rep_settings[rs_name] = avg_scens
    return rep_settings

def compare_KPI(results, KPI_name, reference="ZEN", average=False):
    """

    """
    if isinstance(results, Results):
        results = [results]
    rep_settings = sort_scenarios(results)
    if average:
        rep_settings = average_benchmarking_values(rep_settings)
    plt.figure()
    for rs_name, rss in rep_settings.items():
        KPI = []
        agg_ts = []
        for rs in rss:
            agg_ts.append(rs.system.aggregated_time_steps_per_year)
            KPI.append(rs.benchmarking[KPI_name])
        plt.scatter(agg_ts, KPI, label=rs_name)

    if KPI_name == "objective_value":
        if reference == "ZEN":
            rRef = Results(path="../data/outputs/operation_fully_resolved_ZEN")
        elif reference == "Kot":
            rRef = Results(path="../data/outputs/operation_fully_resolved_Kot")
        else:
            raise NotImplementedError
        benchmarking = next(iter(rRef.solution_loader.scenarios.values())).benchmarking
        plt.axhline(y=benchmarking[KPI_name], label="Reference")
    plt.xlabel("Hours per Period")
    plt.ylabel(KPI_name)
    plt.legend()
    plt.show()

def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n=[]
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1])
    axe.add_artist(l1)
    return axe

def plot_capacities(r, tech_type, capacity_type, carrier, average=False):
    """

    """
    rep_settings = sort_scenarios(r)
    if average:
        raise NotImplementedError
    capacities = []
    for rs_name, rss_list in rep_settings.items():
        for ag_ts, rss_identical in rss_list.items():
            scen_name = next(iter(rss_identical)).name
            capacity = r.get_total("capacity", scenario_name=scen_name)
            r.get_total("set_input_carriers")
            capacity = capacity.loc[pd.IndexSlice[1]]

def cycles_per_year(r):
    # TODO divide by max(storage_level)-min(storage_level) instead of capacity (better for existing capacities)
    capacity = r.get_total("capacity").loc[pd.IndexSlice[:,"energy",:], :]
    capacity = capacity.reset_index(level="capacity_type", drop=True)
    capacity = capacity.rename_axis(index={"location": "node"})
    cycles_per_year = (r.get_total("flow_storage_charge") + r.get_total("flow_storage_discharge")) / (2*capacity)
    return cycles_per_year

rRep2 = Results(path="../data/outputs/operation_multiRepTs_384to768")
rRep_r1 = Results(path="../data/outputs/operation_multiRepTs_24to192_r1")
compare_KPI([rRep_r1,rRep2], "objective_value")
sort_scenarios([rRep_r1,rRep2])
rRep_r2 = Results(path="../data/outputs/operation_multiRepTs_24to192_r2")
r_ZEN_f = Results(path="../data/outputs/operation_fully_resolved_ZEN")
r_Kot_f = Results(path="../data/outputs/operation_fully_resolved_Kot")
a = 1