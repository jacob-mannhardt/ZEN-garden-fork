from networkx.algorithms.bipartite.basic import color

from zen_garden.postprocess.results.results import Results
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import os
from eth_colors import ETHColors

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
            #key = scen.analysis.time_series_aggregation.storageRepresentationMethod + "_" + str(scen.analysis.time_series_aggregation.hoursPerPeriod)
            key = scen.analysis.time_series_aggregation.storageRepresentationMethod
            if "ZEN" not in key:
                key = key.capitalize()
            if "Gabrielli" in key:
                if scen.analysis.time_series_aggregation.hoursPerPeriod == 24:
                    key += " RD"
                else:
                    key += " RH"

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
                if scen[0] is not avg_scen:
                    for key, value in scen[0].benchmarking.items():
                        avg_scen.benchmarking[key] += value
            for key, value in avg_scen.benchmarking.items():
                avg_scen.benchmarking[key] = value / len(rss_identical)
            avg_scens[ag_ts] = avg_scen
        rep_settings[rs_name] = avg_scens
    return rep_settings

def plot_obj_err_vs_solving_time(results, logx=False, logy=False, rep_methods=None, file_format=None):
    if isinstance(results, Results):
        results = [results]
    rep_settings = sort_scenarios(results)
    rep_settings = average_benchmarking_values(rep_settings)
    if rep_methods == "minimal":
        rep_settings = {key: value for key, value in rep_settings.items() if "ZEN" in key or "kot" in key}
    elif rep_methods == "reduced":
        rep_settings = {key: value for key, value in rep_settings.items() if "ZEN" in key or "kot" in key or "gabrielli_24" in key}
    dataset_name = next(iter(results)).name
    ref_base = dataset_name.split("_")[0]
    ref_name = ref_base + "_fully_resolved_ZEN"
    r_ref = Results(path="../data/outputs/"+ref_name)
    ref_obj_val = next(iter(r_ref.solution_loader.scenarios.values())).benchmarking["objective_value"]
    style_mapping = {"ZEN-garden": {"color": eth_c.getColor("petrol"), "marker": "o"},
                     "Gabrielli RD": {"color": eth_c.getColor("green"), "marker": "v"},
                     "Gabrielli RH": {"color": eth_c.getColor("purple"), "marker": "s"},
                     "Kotzur": {"color": eth_c.getColor("yellow"), "marker": "X"}}
    plt.figure()
    for rs_name, rss in rep_settings.items():
        solving_time = []
        relative_objective_error = []
        for ag_ts, rs in rss.items():
            solving_time.append(rs.benchmarking["solving_time"])
            relative_objective_error.append(abs(rs.benchmarking["objective_value"]-ref_obj_val)/ref_obj_val)
        data = pd.DataFrame({"solving_time": solving_time, "relative_objective_error": relative_objective_error})
        data = data.sort_values(by=["solving_time"])
        plt.scatter(data["relative_objective_error"],data["solving_time"], label=rs_name, color=style_mapping[rs_name]["color"], marker=style_mapping[rs_name]["marker"])
    plt.xlabel("Relative objective error [-]")
    plt.ylabel("Solving Time [s]")
    if logy and not logx:
        plt.semilogy()
    elif not logy and logx:
        plt.semilogx()
    elif logy and logx:
        plt.loglog()
    plt.legend()
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"obj_err_vs_solving_time.{file_format}"), format=file_format)
    plt.show()

def get_KPI_data(results, KPI_name, tech_type, capacity_type, carrier, drop_ag_ts, rep_methods, reference, storage_name, conv_names, ts_name=None):
    """

    """
    if isinstance(results, Results):
        results = [results]
    rep_settings = sort_scenarios(results)
    style_mapping = {"ZEN-garden":{"color": eth_c.getColor("petrol"), "marker": "o"}, "Gabrielli RD":{"color": eth_c.getColor("green"), "marker": "v"},
                     "Gabrielli RH":{"color": eth_c.getColor("purple"), "marker": "s"}, "Kotzur":{"color": eth_c.getColor("yellow"), "marker": "X"},
                     "Representative hours": {"color": eth_c.getColor("red"), "marker": "h"}, "Representative days": {"color": eth_c.getColor("bronze"), "marker": "d"}}
    stor_vars_min = next(iter(results)).solution_loader.scenarios["scenario_kotzur_24_1"].benchmarking["number_of_storage_variables"]
    bench_max = next(iter(results)).solution_loader.scenarios["scenario_kotzur_8760_1"].benchmarking
    identical_limits = {"min": stor_vars_min, "max": bench_max["number_of_storage_constraints"], "ref": max(bench_max["number_of_variables"], bench_max["number_of_constraints"]), "scaling": 0.05}
    if "time" in KPI_name:
        identical_limits = {"min": 1, "max": bench_max["solving_time"], "scaling": 0.2}
    data = []
    positions = []
    labels = []
    if "daily" in next(iter([results] if isinstance(results, Results) else results)).name:
        dataset_name = (next(iter([results] if isinstance(results, Results) else results)).name.split("_")[0] + "_"
        + next(iter([results] if isinstance(results, Results) else results)).name.split("_")[1])
    else:
        dataset_name = next(iter([results] if isinstance(results, Results) else results)).name.split("_")[0]
    rRef = Results(path=f"../data/outputs/{dataset_name}_fully_resolved_{reference}")
    ref_bench = next(iter(rRef.solution_loader.scenarios.values())).benchmarking
    unit_mapping = {"capacity": {"power": next(iter(results)).get_unit("capacity")[0],
                                 "energy": next(iter(results)).get_unit("capacity").loc[pd.IndexSlice[:,"energy"]][0].replace(" ","").replace("*","")},
                    "storage_cycles": "-", "objective_value": next(iter(results)).get_unit("cost_shed_demand")[0],
                    "solving_time": "s", "construction_time": "s", "number_of_storage_constraints": "-",
                    "number_of_storage_variables": "-", "RMSE": "%"}
    unit_factors = {key: 1 for key in unit_mapping.keys()}
    if unit_mapping["objective_value"] == "MEURO":
        if ref_bench["objective_value"] > 1e3:
            unit_factors["objective_value"] = 1e-3
            unit_mapping["objective_value"] = "billion euros"
            ref_bench["objective_value"] = ref_bench["objective_value"] * 1e-3
        else:
            raise NotImplementedError
    if unit_mapping["capacity"]["energy"] == "GWh":
        unit_factors["capacity"] = 1e-3
        unit_mapping["capacity"]["energy"] = "TWh"
        unit_mapping["capacity"]["power"] = "TW"
    if rep_methods == "minimal":
        rep_settings = {key: value for key, value in rep_settings.items() if "ZEN" in key or "Kot" in key}
    elif rep_methods == "reduced":
        rep_settings = {key: value for key, value in rep_settings.items() if "ZEN" in key or "Kot" in key or "RD" in key}
    elif rep_methods == "hours/days":
        rep_settings = {"Representative hours": value for key, value in rep_settings.items() if "ZEN" in key} | {"Representative days": value for key, value in rep_settings.items() if "Kot" in key}
    style_mapping = {key: value for key, value in style_mapping.items() if key in rep_settings}
    for rs_name, rss in rep_settings.items():
        for ag_ts, rss_identical in rss.items():
            if ag_ts == drop_ag_ts:
                continue
            positions.append(next(iter(rss_identical))[0].system.aggregated_time_steps_per_year)
            labels.append(rs_name)
            KPI = []
            for rs in rss_identical:
                if KPI_name == "capacity":
                    KPI.append(get_capacities(results, rs, tech_type, capacity_type, carrier, conv_names) * unit_factors[KPI_name])
                elif KPI_name == "storage_cycles":
                    KPI.append(get_cycles_per_year(results, rs, storage_name))
                elif KPI_name == "RMSE":
                    KPI.append(compute_RMSE(results, rs, rRef, ts_name))
                else:
                    KPI.append(rs[0].benchmarking[KPI_name] * unit_factors[KPI_name])
            data.append(KPI)
    unit = unit_mapping[KPI_name]
    if KPI_name == "capacity" and (storage_name or (tech_type=="storage" and capacity_type=="energy")):
        unit = unit["energy"]
    return data, positions, labels, style_mapping, unit, ref_bench, identical_limits


def plot_KPI(results, KPI_name, plot_type="whiskers", reference="ZEN", tech_type="storage", capacity_type="energy",
             carrier="electricity", subplots=False, drop_ag_ts=None, relative_scale_flag=False, rep_methods=None, use_log=False, title="", file_format=None, storage_name=None, conv_names=None, ts_name=None):
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
    data, positions, labels, style_mapping, unit, ref_bench, identical_limits = get_KPI_data(
        results=results, KPI_name=KPI_name, tech_type=tech_type, capacity_type=capacity_type, carrier=carrier,
        drop_ag_ts=drop_ag_ts, rep_methods=rep_methods, reference=reference, storage_name=storage_name, conv_names=conv_names, ts_name=ts_name)
    data_dict = {}
    stacked_colors = {}
    if isinstance(next(iter(next(iter(data)))), pd.Series):
        data_dict = {ind: [] for ind in max([x[0] for x in data], key=len).index}
        tab_cols = plt.cm.tab20b(np.linspace(0, 1, len(max([x[0] for x in data], key=len))))
        stacked_colors = {label: tab_cols[ind] for ind, label in  enumerate(max([x[0] for x in data], key=len).index)}
        for idx, run in enumerate(data):
            for ind_name in next(iter(run)).index:
                data_dict[ind_name].append(
                    {(labels[idx], positions[idx]): [pd.Series(series[ind_name]) for series in run]})
        subplot_labels = [key for key in data_dict.keys()]
    elif subplots == True:
        raise NotImplementedError

    if subplots:
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
                _plot_single_KPI(ax, data_dict[subplot_labels[idx]], positions, labels, style_mapping, unit, KPI_name,
                                 plot_type, reference, add_legend=False, ref_bench=ref_bench, use_log=use_log, identical_limits=identical_limits, stacked_colors=stacked_colors)
                ax.set_title(subplot_labels[idx])
            else:
                ax.axis('off')  # Hide unused subplots

        # Add a single legend for all subplots
        legend_handles = [plt.Line2D([0], [0], color=style["color"], lw=4, label=label) for label, style in style_mapping.items()]
        fig.legend(handles=legend_handles)
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        secondary_ax = None

        # Create a secondary y-axis if relative scale is enabled
        if relative_scale_flag:
            secondary_ax = ax.twinx()
        if data_dict and storage_name:
            data = data_dict[storage_name]
            split = storage_name.split('_')
            if len(split) == 2:
                storage_name = storage_name.split('_')[0] + " " + storage_name.split('_')[1]
            else:
                storage_name = storage_name
            title = f"Storage cycles {storage_name}" if KPI_name == "storage_cycles" else f"Installed {storage_name} capacity"
        _plot_single_KPI(ax, data, positions, labels, style_mapping, unit, KPI_name, plot_type, reference,
                         ref_bench=ref_bench, add_legend=True, secondary_ax=secondary_ax,
                         relative_scale_flag=relative_scale_flag, use_log=use_log, identical_limits=identical_limits, stacked_colors=stacked_colors)
    #fig.suptitle(title)
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        if KPI_name == "RMSE":
            title = ts_name
        if title:
            plt.savefig(os.path.join(path, f"{title}.{file_format}"), format=file_format)
        else:
            plt.savefig(os.path.join(path,f"{KPI_name}.{file_format}"), format=file_format)
    plt.show()


def _plot_single_KPI(ax, data, positions, labels, style_mapping, unit, KPI_name, plot_type, reference, ref_bench, use_log, identical_limits,
                     add_legend=True, secondary_ax=None, relative_scale_flag=False, stacked_colors=None):
    """
    Helper function to plot a single KPI on a given axis, with an optional secondary axis.
    """
    unique_positions = sorted(set(positions))
    equidistant_positions = np.linspace(0, len(unique_positions) - 1, len(unique_positions))
    position_map = {pos: equidistant_positions[i] for i, pos in enumerate(unique_positions)}
    label_offsets = {label: i for i, label in enumerate(style_mapping.keys())}
    num_labels = len(style_mapping)
    stacked = False
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
            combined = pd.concat(dat, axis=1)#.sort_values(by=[0], ascending=False)
            mean_vals = combined.mean(axis=1)
            std_devs = combined.std(axis=1)
            bottom = 0
            for idx, (mean, std) in enumerate(zip(mean_vals, std_devs)):
                if len(mean_vals) == 1:
                    ax.bar(plot_pos, mean, color=style_mapping[label]["color"], edgecolor="black", width=0.15,
                           bottom=bottom, label=None if idx > 0 else label)
                    ax.errorbar(plot_pos, bottom + mean, yerr=std, fmt='none', ecolor='black', capsize=4)
                    bottom += mean
                else:
                    stacked = True
                    ax.bar(plot_pos, mean, color=stacked_colors[mean_vals.index[idx]], edgecolor="black", width=0.15,
                           bottom=bottom, label=mean_vals.index[idx])
                    ax.errorbar(plot_pos, bottom + mean, yerr=std, fmt='none', ecolor='black', capsize=4)
                    bottom += mean
        else:
            if plot_type == "violins":
                violin = ax.violinplot([dat], positions=[plot_pos], widths=0.15, showmeans=True, showextrema=False)
                for pc in violin['bodies']:
                    pc.set_facecolor(style_mapping[label]["color"])
                    pc.set_edgecolor(style_mapping[label]["color"])
                    pc.set_alpha(0.7)
            else:
                mean_val = np.mean(dat)
                lower_whisker = np.percentile(dat, 5)
                upper_whisker = np.percentile(dat, 95)
                ax.scatter([plot_pos], [mean_val], color=style_mapping[label]["color"], zorder=3, marker=style_mapping[label]["marker"])
                ax.vlines(plot_pos, lower_whisker, upper_whisker, color=style_mapping[label]["color"], linewidth=2)
                if "number_of_storage" in KPI_name and 1 or "time" in KPI_name:
                    padding = (identical_limits["max"] - identical_limits["min"]) * identical_limits["scaling"]
                    ax.set_ylim(ymin=identical_limits["min"]-padding,ymax=identical_limits["max"]+padding)
                    #ax.axhline(y=0, color='grey', linestyle='-')
                    if "number_of_storage" in KPI_name:
                        ax.axhspan(identical_limits["min"]-padding, 0, color='grey', alpha=0.5)
                if KPI_name=="RMSE":
                    ax.axhline(y=0, color='grey', linestyle='-')
        # Plot on the secondary axis if enabled
        if relative_scale_flag and secondary_ax:
            if KPI_name == "number_of_storage_variables":
                reference_value = ref_bench["number_of_variables"] /100
            elif KPI_name == "number_of_storage_constraints":
                reference_value = ref_bench["number_of_constraints"]/100
            else:
                reference_value = ref_bench[KPI_name]
            relative_dat = [d / reference_value for d in dat]
            mean_rel = np.mean(relative_dat)
            secondary_ax.scatter([plot_pos], [mean_rel], color=style_mapping[label]["color"], linestyle='--', zorder=3, alpha=0)
            if "number_of_storage" in KPI_name and 0:
                padding = (identical_limits["max"]- identical_limits["min"]) * identical_limits["scaling"]
                secondary_ax.set_ylim(ymin=(identical_limits["min"] - padding)/identical_limits["ref"]*100, ymax=(identical_limits["max"] + padding)/identical_limits["ref"]*100)

    # Ensure vertical lines are drawn correctly on the primary axis
    for i in range(1, len(unique_positions)):
        ax.axvline(x=(equidistant_positions[i - 1] + equidistant_positions[i]) / 2, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.7, zorder=1)

    ax.set_xticks(equidistant_positions)
    ax.set_xticklabels([str(pos) for pos in unique_positions])
    ax.set_xlim(min(equidistant_positions) - 0.5, max(equidistant_positions) + 0.5)
    ax.yaxis.offsetText.set_x(-0.05)
    secondary_xax = ax.twiny()
    secondary_xax.set_xticks(equidistant_positions)
    secondary_xax.set_xticklabels([str(pos//24) for pos in unique_positions])
    secondary_xax.set_xlabel("Number of aggregated representative days [d]")
    secondary_xax.set_xlim(min(equidistant_positions) - 0.5, max(equidistant_positions) + 0.5)

    if add_legend:
        if KPI_name in ["capacity", "storage_cycles"]:
            if KPI_name == "capacity" and stacked:
                legend_handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in stacked_colors.items()]
            else:
                legend_handles = [plt.Line2D([0], [0], color=style["color"], lw=4, label=label) for label, style in style_mapping.items() if label in labels]
        else:
            legend_handles = [plt.Line2D([0], [0], color=style["color"],marker= style["marker"], linestyle="None", label=label) for label, style in style_mapping.items() if label in labels]
        ax.legend(handles=legend_handles, loc='best')

    # Plot reference line for "objective_value"
    if KPI_name == "objective_value" and reference:
        reference_value = ref_bench[KPI_name]  # Get reference value
        ax.axhline(y=reference_value, label=f"{reference} {KPI_name} Benchmark", color='grey', linestyle='-')

    name_dict = {"objective_value": "Objective value", "storage_cycles": "Number of storage cycles",
                 "capacity": "Installed capacity", "number_of_storage_variables": "Number of storage variables",
                 "number_of_storage_constraints": "Number of storage constraints", "solving_time": "Solving time",
                 "construction_time": "Construction time", "RMSE": "NRMSE"}
    # Set secondary axis labels if enabled
    if relative_scale_flag and secondary_ax:
        if KPI_name == "number_of_storage_variables":
            st = "share of all variables [%]"
        elif KPI_name == "number_of_storage_constraints":
            st = "share of all constraints [%]"
        else:
            st = name_dict[KPI_name].lower() + "[-]"
        secondary_ax.set_ylabel(f"Relative {st}")

    ax.set_xlabel("Number of aggregated time steps [h]")
    ax.set_ylabel(f"{name_dict[KPI_name]} [{unit}]")

def get_capacities(results, rs, tech_type, capacity_type, carrier, conv_names):
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
    if conv_names:
        capacity = capacity.loc[conv_names]
    capacity = capacity[capacity>1e-3]
    r.get_total("set_input_carriers")
    return capacity

def get_cycles_per_year(results, rs, storage_name, method="max_min"):
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
    if storage_name:
        if storage_name in cycles_per_year.index:
            cycles_per_year = cycles_per_year.loc[[storage_name]]
        else:
            cycles_per_year = pd.Series([])
    return cycles_per_year

def tsa_demonstration_plot(r, file_format=None, step=False, agg_ts=24, rounding=10):
    """

    """
    full_ts = r.get_full_ts("demand", scenario_name="scenario_gabrielli_rep_hours_8760_1").T["heat", "DE"]
    single_day = full_ts[0:24]
    full_ts_ZEN = r.get_full_ts("demand", scenario_name=f"scenario_ZEN-garden_rep_hours_{agg_ts}_1").T["heat", "DE"]
    single_day_ZEN = full_ts_ZEN[0:24].apply(lambda x: round(x / rounding) * rounding)
    plt.figure()
    if step:
        plt.step(np.arange(24), single_day_ZEN, color=eth_c.getColor("red"))
        plt.step(np.arange(24), single_day, color=eth_c.getColor("blue"))
    else:
        single_day.plot(color=eth_c.getColor("blue"))
        single_day_ZEN.plot(color=eth_c.getColor("red"))
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"tsa_demo.svg"), format=file_format)
    plt.show()

def representation_comparison_plots(r, file_format=None, show_full_ts=False):
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

    if show_full_ts:
        plt.figure(figsize=(20, 4))
        full_ts.plot(linewidth=0.5, color=eth_c.getColor("blue"))
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        if file_format:
            path = os.path.join(os.getcwd(), "plots")
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, f"full_ts.svg"), format=file_format)
        plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=False)

    # Plot the first row (single days)
    ymin, ymax = min(min(single_day), min(single_day_kotzur), min(single_day_ZEN)), max(max(single_day), max(single_day_kotzur), max(single_day_ZEN))
    padding = (ymax-ymin)*0.05
    ymin, ymax = ymin - padding, ymax + padding
    axs[0, 0].plot(single_day, label="Fully resolved", color=eth_c.getColor("blue"), linewidth=3)
    axs[0, 0].set_title('Daily Profile: Fully Resolved')
    axs[0, 0].set_xlabel("Time [hours]")
    axs[0, 1].plot(single_day_kotzur, label="Representative days", color=eth_c.getColor("bronze"), linewidth=3)
    axs[0, 1].set_title('Daily profile: Single representative day')
    axs[0, 1].set_xlabel("Time [hours]")
    axs[0, 1].set_ylim(ymin, ymax)
    axs[0, 2].plot(single_day_ZEN, label="Scenario ZEN", color=eth_c.getColor("red"), linewidth=3)
    axs[0, 2].set_title('Daily profile: 24 representative hours')
    axs[0, 2].set_xlabel("Time [hours]")
    axs[0, 2].set_ylim(ymin, ymax)

    # Plot the second row (weekly averages)
    ymin, ymax = min(min(weekly_rolling_average), min(weekly_rolling_average_kotzur), min(weekly_rolling_average_ZEN)), max(max(weekly_rolling_average), max(weekly_rolling_average_kotzur), max(weekly_rolling_average_ZEN))
    padding = (ymax-ymin)*0.05
    ymin, ymax = ymin - padding, ymax + padding
    axs[1, 0].plot(weekly_rolling_average, label="Fully resolved", color=eth_c.getColor("blue"), linewidth=3)
    axs[1, 0].set_title('Yearly profile: Fully resolved')
    axs[1, 0].set_xlabel("Time [weeks]")
    axs[1, 1].plot(weekly_rolling_average_kotzur, label="Representative days", color=eth_c.getColor("bronze"), linewidth=3)
    axs[1, 1].set_title('Yearly profile: Single representative day')
    axs[1, 1].set_xlabel("Time [weeks]")
    axs[1, 1].set_ylim(ymin, ymax)
    axs[1, 2].plot(weekly_rolling_average_ZEN, label="Scenario ZEN", color=eth_c.getColor("red"), linewidth=3)
    axs[1, 2].set_title('Yearly profile: 24 representative hours')
    axs[1, 2].set_xlabel("Time [weeks]")
    axs[1, 2].set_ylim(ymin, ymax)

    # Adjust layout for better spacing
    axs[0,0].yaxis.set_visible(False)
    axs[0,1].yaxis.set_visible(False)
    axs[0,2].yaxis.set_visible(False)
    axs[1,0].yaxis.set_visible(False)
    axs[1,1].yaxis.set_visible(False)
    axs[1,2].yaxis.set_visible(False)

    plt.tight_layout()
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"representation_comparison_plots.svg"), format=file_format)
    # Show the plot
    plt.show()


def kotzur_representation_plot(r, all=True, file_format=None, agg_ts="768", start=936, end=936+72, zoom=None, rh_title="ZEN-garden", gab="RD"):
    """
    zoom = 35
    """
    data_original = r.get_full_ts("demand", scenario_name="scenario_ZEN-garden_rep_hours_8760_1").T["electricity", "BE"][start:end].reset_index(drop=True)
    data_kot = r.get_full_ts("demand", scenario_name=f"scenario_kotzur_{agg_ts}_1").T["electricity", "BE"][start:end].reset_index(drop=True)
    data_zen = r.get_full_ts("demand", scenario_name=f"scenario_ZEN-garden_rep_hours_{agg_ts}_1").T["electricity", "BE"][start:end].reset_index(drop=True)
    ymin = min(data_zen.min(), data_original.min(), data_kot.min()) - 0.2
    ymax = max(data_zen.max(), data_original.max(), data_kot.max()) + 0.2

    #font_params = {'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12 }


    path = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(path):
        os.makedirs(path)

    # fully resolved ts
    plt.figure(figsize=(7,4))
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.step(np.arange(72), data_original, where="mid", color=eth_c.getColor(color="blue"), linewidth=2)
    plt.ylim(ymin,ymax)
    plt.title("Fully resolved time series")
    #plt.rcParams.update(font_params)
    if file_format:
        plt.savefig(os.path.join(path, f"tree_1.svg"), format=file_format)
    plt.show()

    # aggregated days
    plt.figure(figsize=(7,4))
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    if zoom:
        for i in [23,47]:
            plt.vlines(x=i+1/2, ymin=ymin, ymax=ymax, color='gray', linestyle='--', alpha=0.7)
        plt.step(np.arange(data_kot.size), data_kot, label="Aggregated time series", where="mid", color=eth_c.getColor(color="bronze"), linewidth=2)
        #plt.step(np.arange(data_kot.size), data_kot, label="Aggregated time series", where="mid", color=eth_c.getColor(color="bronze"), linewidth=2)

    else:
        plt.step(np.arange(72), data_kot, label="Aggregated time series", where="mid", color=eth_c.getColor(color="bronze"), linewidth=2)
        plt.step(np.arange(72), data_original, label="Fully resolved time series", where="mid", alpha=0.7, linewidth=2)
    plt.ylim(ymin,ymax)
    plt.title("Representative days")
    plt.legend()
    #plt.rcParams.update(font_params)
    if file_format:
        plt.savefig(os.path.join(path, f"tree2.svg"), format=file_format)
    plt.show()

    # aggregated days gabrielli zoom
    if zoom:
        plt.figure(figsize=(7, 4))
        plt.gca().xaxis.set_visible(False)
        plt.gca().yaxis.set_visible(False)
        linewidth = 2
        linewidth_or = 2
        if zoom:
            data_kot = data_kot[zoom - 10:zoom]
            data_original = data_original[zoom - 10:zoom]
            ymin = min(data_kot.min(), data_original.min()) - 0.2
            ymax = max(data_kot.max(), data_original.max()) + 0.2
            data_kot = data_kot.apply(lambda x: round(x / 0.2) * 0.2)
            for i in range(0, len(data_kot) - 1):
                plt.vlines(x=i + 1 / 2, ymin=ymin, ymax=ymax, color='gray', linestyle='--', alpha=0.7)
            linewidth = 3
            linewidth_or = 3

        plt.step(np.arange(data_kot.size), data_kot, label="Aggregated time series", where="mid",
                 color=eth_c.getColor(color="bronze"), linewidth=linewidth)
        plt.step(np.arange(data_original.size), data_original, label="Fully resolved time series", where="mid", alpha=0.7,
                 linewidth=linewidth_or, color=eth_c.getColor(color="blue"))
        plt.ylim(ymin, ymax)
        plt.title("Representative days")
        plt.legend()
        # plt.rcParams.update(font_params)
        if file_format:
            plt.savefig(os.path.join(path, f"tree4.svg"), format=file_format)
        plt.show()

    # aggregated hours
    plt.figure(figsize=(7,4))
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    linewidth = 2
    linewidth_or = 2
    if zoom:
        data_zen = data_zen[zoom-10:zoom]
        ymin = min(data_zen.min(), data_original.min()) - 0.2
        ymax = max(data_zen.max(), data_original.max()) + 0.2
        data_zen = data_zen.apply(lambda x: round(x / 0.2) * 0.2)
        for i in range(0, len(data_zen)-1):
            plt.vlines(x=i+1/2, ymin=ymin, ymax=ymax, color='gray', linestyle='--', alpha=0.7)
        linewidth = 3
        linewidth_or = 3

    plt.step(np.arange(data_zen.size),data_zen, label="Aggregated time series", where="mid", color=eth_c.getColor(color="red"), linewidth=linewidth)
    plt.step(np.arange(data_original.size), data_original, label="Fully resolved time series", where="mid", alpha=0.7, linewidth=linewidth_or, color=eth_c.getColor(color="blue"))
    plt.ylim(ymin,ymax)
    if zoom:
        plt.title(f"{rh_title}")
    else:
        plt.title("Representative hours")
    plt.legend()
    #plt.rcParams.update(font_params)
    if file_format:
        plt.savefig(os.path.join(path, f"tree3{rh_title}.svg"), format=file_format)
    plt.show()


    # storage level gabrielli rep days
    if gab == "RD":
        storage_level = post_compute_storage_level_kotzur(r, scenario_name=f"scenario_kotzur_{agg_ts}_1")["battery", "BE"][start:end].reset_index(drop=True)
    else:
        storage_level = r.get_full_ts("storage_level", scenario_name=f"scenario_gabrielli_rep_hours_{agg_ts}_1").T["battery", "BE"][start:end].reset_index(drop=True)
    plt.figure(figsize=(7, 4))
    ymin = min(r.get_full_ts("storage_level_intra", scenario_name=f"scenario_kotzur_{agg_ts}_1").T["battery", "BE"][start:end])
    ymax = max(storage_level)
    padding = (ymax-ymin) * 0.05
    plt.ylim(ymin-padding, ymax+padding)
    if zoom:
        storage_level[zoom - 10:zoom].plot(marker=".", color="black")
    else:
        for i in [23,47]:
            plt.vlines(x=i+1/2, ymin=ymin, ymax=ymax, color='gray', linestyle='--', alpha=0.7)
        storage_level.plot(marker=".", color="black")
    plt.title(f"Storage level representation Gabrielli {gab}")
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    if file_format:
        plt.savefig(os.path.join(path, f"tree_Gabrielli_{gab}.svg"), format=file_format)
    plt.show()

    # storage level kotzur
    storage_level = post_compute_storage_level_kotzur(r, scenario_name=f"scenario_kotzur_{agg_ts}_1")["battery", "BE"]
    storage_level_intra = r.get_full_ts("storage_level_intra", scenario_name=f"scenario_kotzur_{agg_ts}_1").T["battery", "BE"]
    storage_level_inter_w_self_discharge = storage_level-storage_level_intra
    storage_level_inter_w_self_discharge_cut = storage_level_inter_w_self_discharge[start:end].reset_index(drop=True)[[0,23,24,47,48,71]]
    storage_level_intra_cut = storage_level_intra[start:end].reset_index(drop=True)
    storage_level_cut = storage_level[start:end].reset_index(drop=True)


    plt.figure(figsize=(7,4))
    storage_level_cut.plot(label='Superposed storage level', color='black', marker=".")

    #x = range(72)
    #plt.xticks(x, labels=[str(tick) if tick in [0, 12, 24, 36, 48, 60, 71] else "" for tick in x])

    if all:
        # Plot the second series in the right subplot (primary y-axis for intra values)
        storage_level_intra_cut[:24].plot(label='Intra storage level', color=eth_c.getColor("bronze", shade=60), marker=".")
        storage_level_intra_cut[24:48].plot(label="", color=eth_c.getColor("bronze", shade=60), marker=".")
        storage_level_intra_cut[48:].plot(label="", color=eth_c.getColor("bronze", shade=60), marker=".")
        # Create a secondary y-axis for the inter values
        storage_level_inter_w_self_discharge_cut[:2].plot(label='Inter storage level', color=eth_c.getColor("grey"), marker=".")
        storage_level_inter_w_self_discharge_cut[2:4].plot(label="", color=eth_c.getColor("grey"), marker=".")
        storage_level_inter_w_self_discharge_cut[4:].plot(label="", color=eth_c.getColor("grey"), marker=".")
    for i in [23, 47]:
        plt.vlines(x=i + 1 / 2, ymin=ymin, ymax=ymax, color='gray', linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.legend()
    plt.title("Storage level representation Kotzur")
    if file_format:
        path = os.path.join(os.getcwd(), "plots")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f"kotzur_representation_plot.svg"), format=file_format)

    # Show the plot
    plt.show()

def get_results(dataset, cluster_method=""):
    """

    """
    results = []
    if dataset == "operation":
        dataset += "_new"
    if cluster_method:
        cluster_method += "_"
    for run in range(5):
        results.append(Results(path=f"../data/outputs/outputs_euler/{dataset}_multiRepTs_24to8760_{cluster_method}r{run+1}"))
    return results

def kotzur_proof(r, carrier):
    plt.figure()
    r.get_full_ts("demand", scenario_name="scenario_kotzur_8760_1").groupby("carrier").sum().T[carrier].groupby(
        np.arange(8760) // 168).mean().plot(label="8760")
    r.get_full_ts("demand", scenario_name="scenario_kotzur_6144_1").groupby("carrier").sum().T[carrier].groupby(
        np.arange(8760) // 168).mean().plot(label="6144")
    r.get_full_ts("demand", scenario_name="scenario_kotzur_3072_1").groupby("carrier").sum().T[carrier].groupby(
        np.arange(8760) // 168).mean().plot(label="3072")
    plt.legend()
    plt.show()


def compute_RMSE(results, rs, rRef, ts_name="all"):
    """

    """
    scen, result_name = rs[0], rs[1]
    r = [r for r in results if r.name == result_name][0]
    if ts_name == "all":
        agg_ts = {"demand": {"atts": ["electricity", "heat"], "ind_name": "carrier"},
              "max_load": {"atts": ["photovoltaics"], "ind_name": "technology"}}
    elif ts_name == "reduced":
        agg_ts = {"demand": {"atts": ["electricity"], "ind_name": "carrier"},
              "max_load": {"atts": ["photovoltaics"], "ind_name": "technology"}}
    elif ts_name == "electricity":
        agg_ts = {"demand": {"atts": ["electricity"], "ind_name": "carrier"}}
    elif ts_name == "heat":
        agg_ts = {"demand": {"atts": ["heat"], "ind_name": "carrier"}}
    elif ts_name == "max_load":
        agg_ts = {"max_load": {"atts": ["photovoltaics"], "ind_name": "technology"}}
    RMSE_sum = 0
    for param, specs in agg_ts.items():
            for att in specs["atts"]:
                ts_original = rRef.get_full_ts(param).groupby(specs["ind_name"]).sum().T[att]
                ts_agg = r.get_full_ts(param, scenario_name=scen.name).groupby(specs["ind_name"]).sum().T[att]
                RMSE_sum += (((ts_original - ts_agg)/ (ts_original.max()-ts_original.min())) ** 2).sum()
    num_atts = sum(len(val["atts"]) for val in agg_ts.values())
    num_ts = 8760
    RMSE = np.sqrt(RMSE_sum / (num_atts * num_ts))
    return RMSE

def agg_demo_plots(file_format=None):
    """

    """
    r = Results(path="../data/outputs/agg_demo_plots")

    agg_ts = {"demand": {"atts": ["electricity", "heat"], "ind_name": "carrier"},
              "max_load": {"atts": ["photovoltaics"], "ind_name": "technology"}}
    for param, specs in agg_ts.items():
            for att in specs["atts"]:
                ts_original = r.get_full_ts(param, scenario_name="scenario_original")
                if "capacity_type" in ts_original.index.names:
                    ts_original = ts_original.droplevel("capacity_type")
                ts_original = ts_original.T[att, "DE"].squeeze()
                ymin = ts_original.min() - 0.05 * ts_original.mean()
                ymax = ts_original.max() + 0.05 * ts_original.mean()
                plt.figure()
                plt.ylim(ymin, ymax)
                for i in range(0, len(ts_original) - 1):
                    plt.vlines(x=i + 1 / 2, ymin=ymin, ymax=ymax, color='gray', linestyle='-', alpha=0.6, linewidth=0.9)
                ts_original.plot(linewidth=3)
                plt.xticks([])
                plt.yticks([])
                if file_format:
                    path = os.path.join(os.getcwd(), "plots")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(os.path.join(path, f"agg_demo_or_{param}_{att}.svg"), format=file_format)
                plt.show()
                ts_agg = r.get_full_ts(param, scenario_name="scenario_")
                if "capacity_type" in ts_agg.index.names:
                    ts_agg = ts_agg.droplevel("capacity_type")
                plt.figure()
                ts_agg = ts_agg.T[att, "DE"].squeeze()
                ts_agg.plot(linewidth=3)
                plt.ylim(ymin, ymax)
                for i in range(0, len(ts_agg) - 1):
                    plt.vlines(x=i + 1 / 2, ymin=ymin, ymax=ymax, color='gray', linestyle='-', alpha=0.6, linewidth=0.9)
                plt.xticks([])
                plt.yticks([])
                if file_format:
                    path = os.path.join(os.getcwd(), "plots")
                    if not os.path.exists(path):
                        os.makedirs(path)
                    plt.savefig(os.path.join(path, f"agg_demo_ag_{param}_{att}.svg"), format=file_format)
                plt.show()



if __name__ == "__main__":
    eth_c = ETHColors()
    r1 = Results(path="../data/outputs/outputs_euler/operation_new_multiRepTs_24to8760_r1")
    r2 = Results(path="../data/outputs/outputs_euler/operation_new_multiRepTs_24to8760_r2")
    r3 = Results(path="../data/outputs/outputs_euler/operation_new_multiRepTs_24to8760_r3")
    r4 = Results(path="../data/outputs/outputs_euler/operation_new_multiRepTs_24to8760_r4")
    r5 = Results(path="../data/outputs/outputs_euler/operation_new_multiRepTs_24to8760_r5")
    a = 1