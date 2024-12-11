from networkx.algorithms.bipartite.basic import color

from zen_garden.postprocess.results.results import Results
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os

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

def get_KPI_data(results, KPI_name, tech_type, capacity_type, carrier, drop_ag_ts, rep_methods, reference):
    """

    """
    if isinstance(results, Results):
        results = [results]
    rep_settings = sort_scenarios(results)
    color_mapping = {label: col for label, col in zip(rep_settings.keys(), plt.cm.tab20b.colors)}
    data = []
    positions = []
    labels = []
    dataset_name = next(iter([results] if isinstance(results, Results) else results)).name.split("_")[0]
    rRef = Results(path=f"../data/outputs/{dataset_name}_fully_resolved_{reference}")
    ref_bench = next(iter(rRef.solution_loader.scenarios.values())).benchmarking
    unit_mapping = {"capacity": next(iter(results)).get_unit("capacity")[0], "storage_cycles": "-",
                    "objective_value": next(iter(results)).get_unit("cost_shed_demand")[0],
                    "solving_time": "s", "construction_time": "s", "number_of_storage_constraints": "-",
                    "number_of_storage_variables": "-"}
    unit_factors = {key: 1 for key in unit_mapping.keys()}
    if unit_mapping["objective_value"] == "MEURO":
        unit_factors["objective_value"] = 1e6
        unit_mapping["objective_value"] = "€"
        ref_bench["objective_value"] = ref_bench["objective_value"] * 1e6
    if rep_methods == "minimal":
        rep_settings = {key: value for key, value in rep_settings.items() if "ZEN" in key or "kot" in key}
    for rs_name, rss in rep_settings.items():
        for ag_ts, rss_identical in rss.items():
            if ag_ts == drop_ag_ts:
                continue
            positions.append(next(iter(rss_identical))[0].system.aggregated_time_steps_per_year)
            labels.append(rs_name)
            KPI = []
            for rs in rss_identical:
                if KPI_name == "capacity":
                    KPI.append(get_capacities(results, rs, tech_type, capacity_type, carrier) * unit_factors[KPI_name])
                elif KPI_name == "storage_cycles":
                    KPI.append(get_cycles_per_year(results, rs))
                else:
                    KPI.append(rs[0].benchmarking[KPI_name] * unit_factors[KPI_name])
            data.append(KPI)
    return data, positions, labels, color_mapping, unit_mapping[KPI_name], ref_bench


def plot_KPI(results, KPI_name, plot_type="whiskers", reference="ZEN", tech_type="storage", capacity_type="energy",
             carrier="electricity", subplots=False, drop_ag_ts=None, relative_scale_flag=False, rep_methods=None, use_log=False, title="", file_format=None):
    """
    Plot KPI data with optional subplots and an optional secondary y-axis for relative scale.

    Args:
        results: Input results data.
        KPI_name: Name of the KPI to plot.
        plot_type: Plot type ('whiskers' or 'violins').
        reference: Reference scenario for benchmarking (e.g., "ZEN").
        tech_type: Technology type to filter (e.g., "storage").
        capacity_type: Capacity type to filter (e.g., "energy").
        carrier: Energy carrier (e.g., "electricity").
        subplots: Whether to create subplots.
        drop_ag_ts: Whether to drop aggregate timeseries data.
        relative_scale_flag: Whether to include a secondary y-axis for relative scale.
        reference_value: The value for normalizing data on the secondary y-axis.
    """
    data, positions, labels, color_mapping, unit, ref_bench = get_KPI_data(
        results=results, KPI_name=KPI_name, tech_type=tech_type, capacity_type=capacity_type, carrier=carrier, drop_ag_ts=drop_ag_ts, rep_methods=rep_methods, reference=reference)

    if subplots:
        if isinstance(next(iter(next(iter(data)))), pd.Series):
            data_dict = {ind: [] for ind in next(iter(next(iter(data)))).index}
            for idx, run in enumerate(data):
                for ind_name in next(iter(run)).index:
                    data_dict[ind_name].append({(labels[idx], positions[idx]): [pd.Series(series[ind_name]) for series in run]})
            subplot_labels = [key for key in data_dict.keys()]
        else:
            raise NotImplementedError

        num_subplots = len(data_dict)
        # Determine grid size for subplots
        num_rows = 1#int(np.ceil(np.sqrt(num_subplots)))
        num_cols = int(np.ceil(num_subplots / num_rows))
        fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=False, figsize=(num_cols * 9, num_rows * 8))

        # Flatten the axes array for easier handling
        if num_rows > 1 or num_cols > 1:
            axs = axs.flatten()
        else:
            axs = [axs]  # Single plot case

        # Ensure the number of axes matches `num_subplots`
        for idx, ax in enumerate(axs):
            if idx < num_subplots:
                # Reuse the single-plot logic for each subplot
                _plot_single_KPI(ax, data_dict[subplot_labels[idx]], positions, labels, color_mapping, unit, KPI_name,
                                 plot_type, reference, add_legend=False, ref_bench=ref_bench, use_log=use_log)
                ax.set_title(subplot_labels[idx])
            else:
                ax.axis('off')  # Hide unused subplots

        # Add a single legend for all subplots
        legend_handles = [plt.Line2D([0], [0], color=col, lw=4, label=label) for label, col in color_mapping.items()]
        fig.legend(handles=legend_handles)
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        secondary_ax = None

        # Create a secondary y-axis if relative scale is enabled
        if relative_scale_flag:
            secondary_ax = ax.twinx()

        _plot_single_KPI(ax, data, positions, labels, color_mapping, unit, KPI_name, plot_type, reference,
                         ref_bench=ref_bench, add_legend=True, secondary_ax=secondary_ax,
                         relative_scale_flag=relative_scale_flag, use_log=use_log)
    fig.suptitle(title)
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path,f"{title}.svg"), format=file_format)
    plt.show()


def _plot_single_KPI(ax, data, positions, labels, color_mapping, unit, KPI_name, plot_type, reference, ref_bench, use_log,
                     add_legend=True, secondary_ax=None, relative_scale_flag=False):
    """
    Helper function to plot a single KPI on a given axis, with an optional secondary axis.
    """
    unique_positions = sorted(set(positions))
    equidistant_positions = np.linspace(0, len(unique_positions) - 1, len(unique_positions))
    position_map = {pos: equidistant_positions[i] for i, pos in enumerate(unique_positions)}
    label_offsets = {label: i for i, label in enumerate(color_mapping.keys())}
    num_labels = len(color_mapping)

    if use_log:
        ax.set_yscale("log")

    for dat, pos, label in zip(data, positions, labels):
        if isinstance(dat, dict):
            label_pos = [key for key in dat.keys()][0]
            label = label_pos[0]
            pos = label_pos[1]
            dat = [val for val in dat.values()][0]
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

        # Plot on the secondary axis if enabled
        if relative_scale_flag and secondary_ax:
            reference_value = ref_bench[KPI_name]
            relative_dat = [d / reference_value for d in dat]
            mean_rel = np.mean(relative_dat)
            secondary_ax.scatter([plot_pos], [mean_rel], color=color_mapping[label], linestyle='--', zorder=3, alpha=0)

    # Ensure vertical lines are drawn correctly on the primary axis
    for i in range(1, len(unique_positions)):
        ax.axvline(x=(equidistant_positions[i - 1] + equidistant_positions[i]) / 2, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_xticks(equidistant_positions)
    ax.set_xticklabels([str(pos) for pos in unique_positions])
    ax.set_xlim(min(equidistant_positions) - 0.5, max(equidistant_positions) + 0.5)
    if KPI_name == "objective_value":
        ax.ticklabel_format(axis='y', style='sci', scilimits=(9, 9))

    if add_legend:
        legend_handles = [plt.Line2D([0], [0], color=col, lw=4, label=label) for label, col in color_mapping.items() if label in labels]
        ax.legend(handles=legend_handles, loc='best')

    # Plot reference line for "objective_value"
    if KPI_name == "objective_value" and reference:
        reference_value = ref_bench[KPI_name]  # Get reference value
        ax.axhline(y=reference_value, label=f"{reference} {KPI_name} Benchmark", color='grey', linestyle='-')

    name_dict = {"objective_value": "Objective Value", "storage_cycles": "Number of Storage Cycles",
                 "capacity": "Installed Capacity", "number_of_storage_variables": "Number of Storage Variables",
                 "number_of_storage_constraints": "Number of Storage Constraints", "solving_time": "Solving Time",
                 "construction_time": "Construction Time"}
    # Set secondary axis labels if enabled
    if relative_scale_flag and secondary_ax:
        secondary_ax.set_ylabel(f"Relative {name_dict[KPI_name]}")

    ax.set_xlabel("Aggregated Time Steps [h]")
    ax.set_ylabel(f"{name_dict[KPI_name]} [{unit}]")

def get_capacities(results, rs, tech_type, capacity_type, carrier):
    """

    """
    if tech_type in ["conversion", "transport"]:
        capacity_type = "power"
    scen, result_name = rs[0], rs[1]
    r = [r for r in results if r.name == result_name][0]
    capacity = r.get_total("capacity", scenario_name=scen.name)
    capacity = capacity.loc[pd.IndexSlice[:,capacity_type,:],:]
    capacity = capacity.reset_index(level="capacity_type", drop=True)
    if tech_type == "conversion":
        output_carriers = r.get_total("set_output_carriers", scenario_name=scen.name)
        conversion_techs = output_carriers[output_carriers == carrier].index
        capacity = capacity[capacity.index.get_level_values("technology").isin(conversion_techs)]
    capacity = capacity.groupby(level="technology").sum()
    capacity = capacity.squeeze()
    capacity = capacity[capacity>1e-3]
    r.get_total("set_input_carriers")
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
    mask2 = capacity > 1e-1
    cycles_per_year = cycles_per_year.where(mask&mask2).dropna()
    cycles_per_year = cycles_per_year.groupby(level="technology").mean()
    return cycles_per_year

def representation_comparison_plots(r, file_format=None):
    """

    """
    full_ts = r.get_full_ts("demand", scenario_name="scenario_ZEN-garden_rep_hours_8760_1").T["heat", "DE"]
    single_day = full_ts[0:24]
    weekly_rolling_average = full_ts.groupby(np.arange(len(full_ts)) // 168).mean()
    full_ts_kotzur = r.get_full_ts("demand", scenario_name="scenario_kotzur_24_1").T["heat", "DE"]
    single_day_kotzur = full_ts_kotzur[0:24]
    weekly_rolling_average_kotzur = full_ts_kotzur.groupby(np.arange(len(full_ts)) // 168).mean()
    full_ts_ZEN = r.get_full_ts("demand", scenario_name="scenario_ZEN-garden_rep_hours_24_1").T["heat", "DE"]
    single_day_ZEN = full_ts_ZEN[0:24]
    weekly_rolling_average_ZEN = full_ts_ZEN.groupby(np.arange(len(full_ts)) // 168).mean()

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    tab20b_colors = plt.cm.tab20b(np.linspace(0, 0.5, 3))
    # Plot the first row (single days)
    axs[0, 0].plot(single_day, label="Fully resolved", color=tab20b_colors[0])
    axs[0, 0].set_title('Daily Profile: Fully Resolved')
    axs[0, 0].set_ylabel("Heat Demand [GW]")
    axs[0, 0].set_xlabel("Time [hours]")
    axs[0, 1].plot(single_day_kotzur, label="Representative Days", color=tab20b_colors[1])
    axs[0, 1].set_title('Daily Profile: Single Representative Day')
    axs[0, 1].set_ylabel("Heat Demand [GW]")
    axs[0, 1].set_xlabel("Time [hours]")
    axs[0, 2].plot(single_day_ZEN, label="Scenario ZEN", color=tab20b_colors[2])
    axs[0, 2].set_title('Daily Profile: 24 Representative Hours')
    axs[0, 2].set_ylabel("Heat Demand [GW]")
    axs[0, 2].set_xlabel("Time [hours]")

    # Plot the second row (weekly averages)
    axs[1, 0].plot(weekly_rolling_average, label="Fully resolved", color=tab20b_colors[0])
    axs[1, 0].set_title('Yearly Profile: Fully Resolved')
    axs[1, 0].set_ylabel("Heat Demand [GW]")
    axs[1, 0].set_xlabel("Time [weeks]")
    axs[1, 1].plot(weekly_rolling_average_kotzur, label="Representative Days", color=tab20b_colors[1])
    axs[1, 1].set_title('Yearly Profile: Single Representative Day')
    axs[1, 1].set_ylabel("Heat Demand [GW]")
    axs[1, 1].set_xlabel("Time [weeks]")
    axs[1, 2].plot(weekly_rolling_average_ZEN, label="Scenario ZEN", color=tab20b_colors[2])
    axs[1, 2].set_title('Yearly Profile: 24 Representative Hours')
    axs[1, 2].set_ylabel("Heat Demand [GW]")
    axs[1, 2].set_xlabel("Time [weeks]")

    # Adjust layout for better spacing
    plt.tight_layout()
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"representation_comparison_plots.svg"), format=file_format)
    # Show the plot
    plt.show()


def kotzur_representation_plot(r, all=True, file_format=None):
    """

    """
    storage_level = post_compute_storage_level_kotzur(r, scenario_name="scenario_kotzur_3072_1")["battery", "BE"]
    storage_level_intra = r.get_full_ts("storage_level_intra", scenario_name="scenario_kotzur_3072_1").T["battery", "BE"]
    storage_level_inter_w_self_discharge = storage_level-storage_level_intra
    storage_level_inter_w_self_discharge_cut = storage_level_inter_w_self_discharge[24*38:24*41].reset_index(drop=True)[[0,23,24,47,48,71]]
    storage_level_intra_cut = storage_level_intra[24*38:24*41].reset_index(drop=True)
    storage_level_cut = storage_level[24*38:24*41].reset_index(drop=True)

    plt.figure(figsize=(9,4.8)) #, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Plot the first series in the left subplot
    storage_level_cut.plot(label='Fully Resolved Storage Level', color='black', marker="o")
    plt.xlabel('Time')
    plt.ylabel('Storage Level')

    x = range(72)
    plt.xticks(x, labels=[str(tick) if tick in [0, 12, 24, 36, 48, 60, 71] else "" for tick in x])

    if all:
        # Plot the second series in the right subplot (primary y-axis for intra values)
        storage_level_intra_cut[:24].plot(label='Intra Storage Level', color=plt.get_cmap('tab20b')(2/19), marker="o")
        storage_level_intra_cut[24:48].plot(label="", color=plt.get_cmap('tab20b')(2/19), marker="o")
        storage_level_intra_cut[48:].plot(label="", color=plt.get_cmap('tab20b')(2/19), marker="o")
        # Create a secondary y-axis for the inter values
        storage_level_inter_w_self_discharge_cut[:2].plot(label='Inter Storage Level', color=plt.get_cmap('tab20b')(10/19), marker="o")
        storage_level_inter_w_self_discharge_cut[2:4].plot(label="", color=plt.get_cmap('tab20b')(10/19), marker="o")
        storage_level_inter_w_self_discharge_cut[4:].plot(label="", color=plt.get_cmap('tab20b')(10/19), marker="o")

    # Adjust layout for better spacing
    #plt.tight_layout()
    plt.legend()
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"kotzur_representation_plot.svg"), format=file_format)

    # Show the plot
    plt.show()



if __name__ == "__main__":
    r = Results(path="../data/outputs/outputs_euler/snapshot_multiRepTs_24to8760_r1")
    r2 = Results(path="../data/outputs/outputs_euler/snapshot_multiRepTs_24to8760_r2")
    r3 = Results(path="../data/outputs/outputs_euler/snapshot_multiRepTs_24to8760_r3")
    r4 = Results(path="../data/outputs/outputs_euler/snapshot_multiRepTs_24to8760_r4")
    r5 = Results(path="../data/outputs/outputs_euler/snapshot_multiRepTs_24to8760_r5")
    a = 1