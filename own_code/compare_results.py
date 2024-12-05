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

def plot_KPI(results, KPI_name, plot_type="whiskers", reference="ZEN", tech_type="storage", capacity_type="energy", carrier="electricity"):
    """

    """
    data, positions, labels, color_mapping = get_KPI_data(results=results, KPI_name=KPI_name, tech_type=tech_type, capacity_type=capacity_type, carrier=carrier)
    fig, ax = plt.subplots()
    # Create an array of equidistant positions for the x-axis
    unique_positions = sorted(set(positions))  # Get unique positions
    equidistant_positions = np.linspace(0, len(unique_positions) - 1, len(unique_positions))  # Create equidistant positions
    # Map the original positions to the equidistant positions
    position_map = {pos: equidistant_positions[i] for i, pos in enumerate(unique_positions)}
    label_offsets = {label: i for i, label in enumerate(color_mapping.keys())}
    num_labels = len(color_mapping)
    # Plot with offsets
    for dat, pos, label in zip(data, positions, labels):
        base_pos = position_map[pos]  # Get base position for this time step
        offset = (label_offsets[label] - (num_labels - 1) / 2) * 0.2  # Calculate offset
        plot_pos = base_pos + offset  # Adjust position with offset

        if all(isinstance(d, pd.Series) for d in dat):
            # Handle list of Series for stacked bar plot
            # Combine all Series in `dat` into a DataFrame (each Series is a column)
            combined = pd.concat(dat, axis=1)

            # Calculate the mean and std for each index (across all Series for each index)
            mean_vals = combined.mean(axis=1)  # Mean for each index across all Series
            std_devs = combined.std(axis=1)  # Standard deviation for each index across all Series

            # Initialize the bottom of the stack
            bottom = 0

            # Plot each mean as a stacked bar segment, and add error bars
            for idx, (mean, std) in enumerate(zip(mean_vals, std_devs)):
                ax.bar(plot_pos, mean, color=color_mapping[label], edgecolor="black", width=0.15,
                       bottom=bottom, label=None if idx > 0 else label)

                # Add error bar for each segment
                ax.errorbar(plot_pos, bottom + mean, yerr=std, fmt='none', ecolor='black', capsize=4)

                bottom += mean  # Update bottom for stacking the next bar segment

        else:
            if plot_type == "violins":
                violin = ax.violinplot([dat], positions=[plot_pos], widths=0.15, showmeans=True, showextrema=False)
                # Set custom color for this violin
                for pc in violin['bodies']:
                    pc.set_facecolor(color_mapping[label])
                    pc.set_edgecolor(color_mapping[label])
                    pc.set_alpha(0.7)
            else:
                # Calculate statistics
                mean_val = np.mean(dat)
                lower_whisker = np.percentile(dat, 5)  # 5th percentile
                upper_whisker = np.percentile(dat, 95)  # 95th percentile
                # Plot the mean as a marker
                ax.scatter([plot_pos], [mean_val], color=color_mapping[label], zorder=3)
                # Plot the whiskers
                ax.vlines(plot_pos, lower_whisker, upper_whisker, color=color_mapping[label], linewidth=2)
    # Add vertical lines to separate groups
    for i in range(1, len(unique_positions)):
        ax.axvline(x=(equidistant_positions[i - 1] + equidistant_positions[i]) / 2, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    # Set x-axis ticks to show original positions
    ax.set_xticks(equidistant_positions)
    ax.set_xticklabels([str(pos) for pos in unique_positions])
    # Adjust x-axis limits to reflect the new spacing
    ax.set_xlim(min(equidistant_positions) - 0.5, max(equidistant_positions) + 0.5)
    legend_handles = [plt.Line2D([0], [0], color=col, lw=4, label=label) for label, col in color_mapping.items()]
    ax.legend(handles=legend_handles, title="Labels", loc='best')

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
    plt.show()

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

#snap_ZEN_f = Results(path="../data/outputs/snapshot_fully_resolved_ZEN")
#snap_Kot_f = Results(path="../data/outputs/snapshot_fully_resolved_Kot")
#rRep2 = Results(path="../data/outputs/operation_multiRepTs_384to768")
#plot_KPI(rRep2,"storage_cycles")
rRep_r1 = Results(path="../data/outputs/operation_multiRepTs_24to192_r1")
plot_KPI(rRep_r1, "storage_cycles")
rRep3 = Results(path="../data/outputs/operation_multiRepTs_1536to1536")
rRep_r2 = Results(path="../data/outputs/operation_multiRepTs_24to192_r2")
r_ZEN_f = Results(path="../data/outputs/operation_fully_resolved_ZEN")
r_Kot_f = Results(path="../data/outputs/operation_fully_resolved_Kot")
a = 1