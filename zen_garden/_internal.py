"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch),
               Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Compilation  of the optimization problem.
"""
import cProfile
import importlib.util
import logging
import os
from collections import defaultdict
import importlib

from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils, OptimizationError
from .preprocess.unit_handling import Scaling

# we setup the logger here
setup_logger()


def main(config, dataset_path=None, job_index=None):
    """
    This function runs ZEN garden,
    it is executed in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    """

    # print the version
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    logging.info(f"Optimizing for dataset {config.analysis['dataset']}")
    # get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.analysis["folder_output"] = os.path.abspath(config.analysis['folder_output'])

    ### SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    input_data_checks.read_system_file(config)

    # from previous code: (defines "system" for later comparison --> changed to config.system)
    # system_path = os.path.join(config.analysis['dataset'], "system.py")
    # spec = importlib.util.spec_from_file_location("module", system_path)
    # module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(module)
    system_comp =config.system
    # config.system_comp.update(system_comp)

    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()
    # overwrite default system and scenario dictionaries
    scenarios,elements = ScenarioUtils.get_scenarios(config,job_index)
    # get the name of the dataset
    model_name, out_folder = StringUtils.setup_model_folder(config.analysis,config.system)
    # clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config,out_folder)
    ### ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements):
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(config, scenario_dict=scenario_dict, input_data_checks=input_data_checks)
        # define foresight type dependent horizon
        optimization_setup.set_fs_type()
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()

        # To check if the years_in_rolling_horizon_operation for the
        # current scenario is different from system's years_in_rolling_horizon
        comparison_horizon_op = system_comp["years_in_rolling_horizon_operation"]
        comparison_horizon_inv = system_comp["years_in_rolling_horizon"]
        if scenario != '':
            scenario_years_in_rolling_horizon = \
                scenario_dict.get("system", {}).get("years_in_rolling_horizon")
            scenario_years_in_rolling_horizon_operation = \
                scenario_dict.get("system", {}).get("years_in_rolling_horizon_operation")
            # if the current scenario alters the foresight horizon of operation or investment,
            # adjust the comparison_horizon accordingly
            if scenario_years_in_rolling_horizon != None:
                comparison_horizon_inv = scenario_years_in_rolling_horizon
            if scenario_years_in_rolling_horizon_operation != None:
                comparison_horizon_op = scenario_years_in_rolling_horizon_operation

        # are operational decisions to be separarted from investment decisions?
        if system_comp["use_rolling_horizon"] and system_comp["myopic_operation"] and comparison_horizon_inv != comparison_horizon_op:
            for step in steps_horizon:
                StringUtils.print_optimization_progress(scenario, steps_horizon, step, system=config.system)
                # 1) Investment optimization (fs_type = investment):
                if scenario == '':
                    logging.info(f"--- Base Scenario; Investment Opt --- \n")
                else:
                    logging.info(f"--- Investment Opt --- \n")

                # set foresight horizon to investment
                optimization_setup.set_fs_type("Investment")
                # get rolling horizon years
                optimization_setup.get_optimization_horizon()
                # overwrite time indices
                optimization_setup.overwrite_time_indices(step)
                # create optimization problem
                optimization_setup.construct_optimization_problem()
                # from new code
                if config.solver["use_scaling"]:
                    optimization_setup.scaling.run_scaling()
                else:
                    optimization_setup.scaling.analyze_numerics()
                # SOLVE THE OPTIMIZATION PROBLEM
                optimization_setup.solve()
                # break if infeasible
                if not optimization_setup.optimality:
                    # write IIS
                    if len(scenarios) > 1:
                        optimization_setup.write_IIS(scenario)
                    else:
                        optimization_setup.write_IIS()
                    raise OptimizationError(optimization_setup.model.termination_condition)
                # from new code
                if config.solver["use_scaling"]:
                    optimization_setup.scaling.re_scale()
                scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                    config=config, scenario=scenario, scenario_dict=scenario_dict, steps_horizon=steps_horizon,
                    step=step, operation=optimization_setup.fs_type_operation
                )
                # write results
                Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)

                # 2) Operation optimization (fs_type = operation):
                StringUtils.print_optimization_progress(scenario, steps_horizon, step, system=config.system)
                if scenario == '':
                    logging.info(f"--- Base Scenario; Operation Opt --- \n")
                else:
                    logging.info(f"--- Operation Opt --- \n")
                # set foresight horizon to operation and set flag
                optimization_setup.set_fs_type("Operation")
                # get rolling horizon years
                optimization_setup.get_optimization_horizon()
                # overwrite time indices
                optimization_setup.overwrite_time_indices(step)
                # set foresight horizon to investment
                # fix newly capacity_addition of first years in operation horizon
                optimization_setup.fix_new_capacity_addition_for_operation(step)
                # create optimization problem
                optimization_setup.construct_optimization_problem()
                # from new code
                if config.solver["use_scaling"]:
                    optimization_setup.scaling.run_scaling()
                else:
                    optimization_setup.scaling.analyze_numerics()
                # SOLVE THE OPTIMIZATION PROBLEM
                optimization_setup.solve()
                # break if infeasible
                if not optimization_setup.optimality:
                    # write IIS
                    optimization_setup.write_IIS()
                    raise OptimizationError(optimization_setup.model.termination_condition)

                #  from new code
                if config.solver["use_scaling"]:
                    optimization_setup.scaling.re_scale()

                # save new capacity additions and cumulative carbon emissions for next time step
                optimization_setup.add_results_of_optimization_step(step)
                # EVALUATE RESULTS
                # create scenario name, subfolder and param_map for postprocessing
                scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                    config=config, scenario=scenario, scenario_dict=scenario_dict, steps_horizon=steps_horizon,
                    step=step, operation=optimization_setup.fs_type_operation
                )
                # write results
                Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)
            pass

        else:
            # iterate through horizon steps
            for step in steps_horizon:
                StringUtils.print_optimization_progress(scenario,steps_horizon,step, system=config.system)
                # overwrite time indices
                optimization_setup.overwrite_time_indices(step)
                # create optimization problem
                optimization_setup.construct_optimization_problem()

                # from new code
                if config.solver["use_scaling"]:
                    optimization_setup.scaling.run_scaling()
                else:
                    optimization_setup.scaling.analyze_numerics()

                # SOLVE THE OPTIMIZATION PROBLEM
                optimization_setup.solve()
                # break if infeasible
                if not optimization_setup.optimality:
                    # write IIS
                    if len(scenarios) > 1:
                        optimization_setup.write_IIS(scenario)
                    else:
                        optimization_setup.write_IIS()
                    break
                # from new code
                if config.solver["use_scaling"]:
                    optimization_setup.scaling.re_scale()
                # save new capacity additions and cumulative carbon emissions for next time step
                optimization_setup.add_results_of_optimization_step(step)
                # EVALUATE RESULTS
                # create scenario name, subfolder and param_map for postprocessing
                scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                    config = config,scenario = scenario,scenario_dict=scenario_dict,steps_horizon=steps_horizon,step=step
                )
                # write results
                Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                                model_name=model_name, scenario_name=scenario_name, param_map=param_map)
    logging.info("--- Optimization finished ---")
    return optimization_setup
