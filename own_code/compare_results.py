from zen_garden.postprocess.results.multi_hdf_loader import file_names_maps
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
                rep_settings[key] = {scen.system.aggregated_time_steps_per_year: [(s, result.name) for s_n, s in result.solution_loader.scenarios.items() if s.analysis.time_series_aggregation.storageRepresentationMethod
                                          == scen.analysis.time_series_aggregation.storageRepresentationMethod
                                          and s.analysis.time_series_aggregation.hoursPerPeriod
                                          == scen.analysis.time_series_aggregation.hoursPerPeriod
                                     and s.system.aggregated_time_steps_per_year
                                     == scen.system.aggregated_time_steps_per_year]}
            elif scen.system.aggregated_time_steps_per_year not in rep_settings[key].keys():
                rep_settings[key][scen.system.aggregated_time_steps_per_year] = [(s, result.name) for s_n, s in result.solution_loader.scenarios.items() if s.analysis.time_series_aggregation.storageRepresentationMethod
                                          == scen.analysis.time_series_aggregation.storageRepresentationMethod
                                          and s.analysis.time_series_aggregation.hoursPerPeriod
                                          == scen.analysis.time_series_aggregation.hoursPerPeriod
                                     and s.system.aggregated_time_steps_per_year
                                     == scen.system.aggregated_time_steps_per_year]
            else:
                rep_settings[key][scen.system.aggregated_time_steps_per_year].extend(
                    [(s, result.name) for s_n, s in result.solution_loader.scenarios.items() if s.analysis.time_series_aggregation.storageRepresentationMethod
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
        avg_scens = {}
        for ag_ts, rss_identical in rss_list.items():
            avg_scen = next(iter(rss_identical))[0]
            for scen in rss_identical:
                if scen[0].name != avg_scen.name:
                    for key, value in scen[0].benchmarking.items():
                        avg_scen.benchmarking[key] += value
            for key, value in avg_scen.benchmarking.items():
                avg_scen.benchmarking[key] = value / len(rss_identical)
            avg_scens[ag_ts] = avg_scen
        rep_settings[rs_name] = avg_scens
    return rep_settings

def plot_obj_err_vs_solving_time(results):
    if isinstance(results, Results):
        results = [results]
    rep_settings = sort_scenarios(results)
    rep_settings = average_benchmarking_values(rep_settings)
    dataset_name = next(iter(results)).name
    ref_base = dataset_name.split("_")[0]
    ref_name = ref_base + "_fully_resolved_ZEN"
    r_ref = Results(path="../data/outputs/"+ref_name)
    ref_obj_val = next(iter(r_ref.solution_loader.scenarios.values())).benchmarking["objective_value"]
    plt.figure()
    for rs_name, rss in rep_settings.items():
        solving_time = []
        relative_objective_error = []
        for ag_ts, rs in rss.items():
            solving_time.append(rs.benchmarking["solving_time"])
            relative_objective_error.append(abs(rs.benchmarking["objective_value"]-ref_obj_val)/ref_obj_val)
        data = pd.DataFrame({"solving_time": solving_time, "relative_objective_error": relative_objective_error})
        data = data.sort_values(by=["solving_time"])
        plt.plot(data["solving_time"], data["relative_objective_error"], label=rs_name)
    plt.xlabel("Solving Time (s)")
    plt.ylabel("Relative Objective Error")
    plt.legend()
    plt.show()

def get_KPI_data(results, KPI_name, tech_type, capacity_type, carrier):
    """

    """
    if isinstance(results, Results):
        results = [results]
    rep_settings = sort_scenarios(results)
    color_mapping = {label: col for label, col in zip(rep_settings.keys(), plt.cm.Set1.colors)}
    data = []
    positions = []
    labels = []
    for rs_name, rss in rep_settings.items():
        for ag_ts, rss_identical in rss.items():
            positions.append(next(iter(rss_identical))[0].system.aggregated_time_steps_per_year)
            labels.append(rs_name)
            KPI = []
            for rs in rss_identical:
                if KPI_name == "capacity":
                    KPI.append(get_capacities(results, rs, tech_type, capacity_type, carrier))
                elif KPI_name == "storage_cycles":
                    KPI.append(get_cycles_per_year(results, rs))
                else:
                    KPI.append(rs[0].benchmarking[KPI_name])
            data.append(KPI)
    return data, positions, labels, color_mapping


def plot_KPI(results, KPI_name, plot_type="whiskers", reference="ZEN", tech_type="storage", capacity_type="energy",
             carrier="electricity", subplots=False):
    """
    Plot KPI data with optional subplots.

    Args:
        results: Input results data.
        KPI_name: Name of the KPI to plot.
        plot_type: Plot type ('whiskers' or 'violins').
        reference: Reference scenario for benchmarking (e.g., "ZEN").
        tech_type: Technology type to filter (e.g., "storage").
        capacity_type: Capacity type to filter (e.g., "energy").
        carrier: Energy carrier (e.g., "electricity").
        subplots: Whether to create subplots.
        num_subplots: Number of subplots to create if `subplots=True`.
    """
    data, positions, labels, color_mapping = get_KPI_data(
        results=results, KPI_name=KPI_name, tech_type=tech_type, capacity_type=capacity_type, carrier=carrier)

    if subplots:
        if isinstance(next(iter(next(iter(data)))), pd.Series):
            data_dict = {ind: [] for ind in next(iter(next(iter(data)))).index}
            for run in data:
                for ind in next(iter(run)).index:
                    data_dict[ind].append([pd.Series(series[ind]) for series in run])
            subplot_labels = [key for key in data_dict.keys()]
        else:
            raise NotImplementedError

        num_subplots = len(data_dict)
        # Determine grid size for subplots
        num_rows = int(np.ceil(np.sqrt(num_subplots)))
        num_cols = int(np.ceil(num_subplots / num_rows))
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8), sharex=True, sharey=False)

        # Flatten the axes array for easier handling
        if num_rows > 1 or num_cols > 1:
            axs = axs.flatten()
        else:
            axs = [axs]  # Single plot case

        # Ensure the number of axes matches `num_subplots`
        for idx, ax in enumerate(axs):
            if idx < num_subplots:
                # Reuse the single-plot logic for each subplot
                _plot_single_KPI(ax, data_dict[subplot_labels[idx]], positions, labels, color_mapping, KPI_name, plot_type, reference)
            else:
                ax.axis('off')  # Hide unused subplots

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_single_KPI(ax, data, positions, labels, color_mapping, KPI_name, plot_type, reference)

    plt.show()

def _plot_single_KPI(ax, data, positions, labels, color_mapping, KPI_name, plot_type, reference):
    """
    Helper function to plot a single KPI on a given axis.
    """
    unique_positions = sorted(set(positions))
    equidistant_positions = np.linspace(0, len(unique_positions) - 1, len(unique_positions))
    position_map = {pos: equidistant_positions[i] for i, pos in enumerate(unique_positions)}
    label_offsets = {label: i for i, label in enumerate(color_mapping.keys())}
    num_labels = len(color_mapping)

    for dat, pos, label in zip(data, positions, labels):
        base_pos = position_map[pos]
        offset = (label_offsets[label] - (num_labels - 1) / 2) * 0.2
        plot_pos = base_pos + offset

        if all(isinstance(d, pd.Series) for d in dat):
            combined = pd.concat(dat, axis=1)
            mean_vals = combined.mean(axis=1)
            std_devs = combined.std(axis=1)
            bottom = 0
            for idx, (mean, std) in enumerate(zip(mean_vals, std_devs)):
                ax.bar(plot_pos, mean, color=color_mapping[label], edgecolor="black", width=0.15,
                       bottom=bottom, label=None if idx > 0 else label)
                ax.errorbar(plot_pos, bottom + mean, yerr=std, fmt='none', ecolor='black', capsize=4)
                bottom += mean
        else:
            if plot_type == "violins":
                violin = ax.violinplot([dat], positions=[plot_pos], widths=0.15, showmeans=True, showextrema=False)
                for pc in violin['bodies']:
                    pc.set_facecolor(color_mapping[label])
                    pc.set_edgecolor(color_mapping[label])
                    pc.set_alpha(0.7)
            else:
                mean_val = np.mean(dat)
                lower_whisker = np.percentile(dat, 5)
                upper_whisker = np.percentile(dat, 95)
                ax.scatter([plot_pos], [mean_val], color=color_mapping[label], zorder=3)
                ax.vlines(plot_pos, lower_whisker, upper_whisker, color=color_mapping[label], linewidth=2)

    for i in range(1, len(unique_positions)):
        ax.axvline(x=(equidistant_positions[i - 1] + equidistant_positions[i]) / 2, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.7)

    ax.set_xticks(equidistant_positions)
    ax.set_xticklabels([str(pos) for pos in unique_positions])
    ax.set_xlim(min(equidistant_positions) - 0.5, max(equidistant_positions) + 0.5)

    legend_handles = [plt.Line2D([0], [0], color=col, lw=4, label=label) for label, col in color_mapping.items()]
    ax.legend(handles=legend_handles, title="Labels", loc='best')

    if KPI_name == "objective_value" and reference:
        rRef = Results(path=f"../data/outputs/operation_fully_resolved_{reference}")
        benchmarking = next(iter(rRef.solution_loader.scenarios.values())).benchmarking
        ax.axhline(y=benchmarking[KPI_name], label="Reference")

    ax.set_xlabel("Hours per Period")
    ax.set_ylabel(KPI_name)

def get_capacities(results, rs, tech_type, capacity_type, carrier):
    """

    """
    scen, result_name = rs[0], rs[1]
    r = [r for r in results if r.name == result_name][0]
    capacity = r.get_total("capacity", scenario_name=scen.name)
    r.get_total("set_input_carriers")
    capacity = capacity.loc[pd.IndexSlice[1]]
    return capacity

def get_cycles_per_year(results, rs, method="max_min"):
    scen, result_name = rs[0], rs[1]
    r = [r for r in results if r.name == result_name][0]
    if method == "capacity":
        capacity = r.get_total("capacity", scenario_name=scen.name).loc[pd.IndexSlice[:,"energy",:], :]
        capacity = capacity.reset_index(level="capacity_type", drop=True)
        capacity = capacity.rename_axis(index={"location": "node"})
    else:
        if "kotzur" in scen.name:
            sl = post_compute_storage_level_kotzur(r,scenario_name=scen.name)
        else:
            sl = r.get_full_ts("storage_level", scenario_name=scen.name).T
        capacity = sl.max()-sl.min()
    flow_charge = r.get_total("flow_storage_charge", scenario_name=scen.name).squeeze()
    flow_discharge = r.get_total("flow_storage_discharge", scenario_name=scen.name).squeeze()
    cycles_per_year = (flow_charge+flow_discharge) / (2*capacity)
    # drop entries with capacity approx. zero
    mean_capacity = capacity.groupby(level='technology').transform('mean')
    mask = abs(abs((capacity-mean_capacity) / mean_capacity) - 1) > 1e-3
    mask2 = capacity > 1e-5
    cycles_per_year = cycles_per_year.where(mask&mask2).dropna()
    cycles_per_year = cycles_per_year.groupby(level="technology").mean()
    return cycles_per_year

def load_results(ts="all", reps=False):
    """

    """
    results = []
    if ts == "all":
        results.append(Results(path="../data/outputs/operation_multiRepTs_24to192_r1"))
        results.append(Results(path="../data/outputs/operation_multiRepTs_384to768"))
        results.append(Results(path="../data/outputs/operation_multiRepTs_1536to1536"))
        results.append(Results(path="../data/outputs/operation_multiRepTs_3072to3072"))
        results.append(Results(path="../data/outputs/operation_multiRepTs_6144to6144_r1"))




r1 = Results(path="../data/outputs/operation_multiRepTs_6144to6144_r2")
plot_KPI(r1, "storage_cycles", subplots=True)
r2 = Results(path="../data/outputs/operation_multiRepTs_384to768")
plot_KPI([r1, r2],"storage_cycles", subplots=True)
r3 = Results(path="../data/outputs/operation_multiRepTs_6144to6144_r3")
a = 1