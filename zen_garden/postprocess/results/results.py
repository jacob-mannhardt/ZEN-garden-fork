import pandas as pd
from typing import Optional, Any
from zen_garden.postprocess.results.solution_loader import (
    SolutionLoader,
    Scenario,
    Component,
    TimestepType,
    ComponentType,
)
from zen_garden.postprocess.results.multi_hdf_loader import MultiHdfLoader
from functools import cache


class Results:
    def __init__(self, path: str):
        self.solution_loader: SolutionLoader = MultiHdfLoader(path)

    def _get_annuity(self, scenario: Scenario, discount_to_first_step: bool = True):
        """discounts the duals

        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param scenario: scenario name whose results are assessed
        :return: #TODO describe parameter/return
        """
        system = scenario.system
        discount_rate_component = self.solution_loader.components["discount_rate"]
        # calculate annuity
        discount_rate = self.solution_loader.get_component_data(
            scenario, discount_rate_component
        ).squeeze()

        years = list(range(0, system["optimized_years"]))

        annuity = pd.Series(index=years, dtype=float)
        for year in years:
            interval_between_years = system.interval_between_years
            if year == years[-1]:
                interval_between_years_this_year = 1
            else:
                interval_between_years_this_year = system.interval_between_years
            if self.solution_loader.has_rh:
                if discount_to_first_step:
                    annuity[year] = interval_between_years_this_year * (
                        1 / (1 + discount_rate)
                    )
                else:
                    annuity[year] = sum(
                        ((1 / (1 + discount_rate)) ** (_intermediate_time_step))
                        for _intermediate_time_step in range(
                            0, interval_between_years_this_year
                        )
                    )
            else:
                if discount_to_first_step:
                    annuity[year] = interval_between_years_this_year * (
                        (1 / (1 + discount_rate))
                        ** (interval_between_years * (year - years[0]))
                    )
                else:
                    annuity[year] = sum(
                        (
                            (1 / (1 + discount_rate))
                            ** (
                                interval_between_years * (year - years[0])
                                + _intermediate_time_step
                            )
                        )
                        for _intermediate_time_step in range(
                            0, interval_between_years_this_year
                        )
                    )
        return annuity

    def get_df(
        self, component_name: str, scenario_name: Optional[str] = None  
    ) -> dict[str, "pd.DataFrame | pd.Series[Any]"]:
        component = self.solution_loader.components[component_name]

        scenario_names = (
            self.solution_loader.scenarios.keys()
            if scenario_name is None
            else [scenario_name]
        )

        ans = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            ans[scenario_name] = self.solution_loader.get_component_data(
                scenario, component
            )

        return ans

    def get_full_ts_per_scenario(
        self,
        scenario: Scenario,
        component: Component,
        year: Optional[int] = None,
        discount_to_first_step: bool = True,
        element_name: Optional[str] = None,
    ) -> "pd.Series[Any]":
        assert component.timestep_type is not None
        series = self.solution_loader.get_component_data(scenario, component)

        if element_name is not None:
            series = series.loc[element_name]

        if year is None:
            years = [i for i in range(0, scenario.system.optimized_years)]
        else:
            years = [year]

        if isinstance(series.index, pd.MultiIndex):
            series = series.unstack(component.timestep_name)

        if component.timestep_type is TimestepType.yearly:
            if component.component_type not in [
                ComponentType.parameter,
                ComponentType.variable,
            ]:
                annuity = self._get_annuity(scenario, discount_to_first_step)
                ans = series / annuity
            else:
                ans = series

            try:
                ans = ans[years]
            except KeyError:
                pass

            return ans

        sequence_timesteps = self.solution_loader.get_sequence_time_steps(
            scenario, component.timestep_type
        )

        if (
            component.component_type is ComponentType.dual
            and component.timestep_type is not None
        ):
            timestep_duration = self.solution_loader.get_timestep_duration(
                scenario, component
            )

            annuity = self._get_annuity(scenario)

            series = series.div(timestep_duration, axis=1)
            for year_temp in annuity.index:
                time_steps_year = self.solution_loader.get_timesteps_of_year(
                    scenario, component.timestep_type, year_temp
                )
                series[time_steps_year] = series[time_steps_year] / annuity[year_temp]

        try:
            output_df = series[sequence_timesteps]
        except KeyError:
            output_df = series

        output_df = output_df.T.reset_index(drop=True).T
        return output_df

    def get_full_ts(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        discount_to_first_step: bool = True,
        year: Optional[int] = None,
        element_name: Optional[str] = None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        component = self.solution_loader.components[component_name]

        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"] = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]

            scenarios_dict[scenario_name] = self.get_full_ts_per_scenario(
                scenario,
                component,
                discount_to_first_step=discount_to_first_step,
                year=year,
                element_name=element_name,
            )

        return self._concat_scenarios_dict(scenarios_dict)

    @cache
    def get_total_per_scenario(
        self,
        scenario: Scenario,
        component: Component,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Calculates the total values of a component for a specific scenario.
        """
        assert (
            component.component_type is not ComponentType.sets
        ), "Cannot calculate Total for Sets"

        series = self.solution_loader.get_component_data(scenario, component)

        if year is None:
            years = [i for i in range(0, scenario.system.optimized_years)]
        else:
            years = [year]

        if component.timestep_type is None:
            return series

        if type(series.index) is not pd.MultiIndex:
            return series

        if component.timestep_type is TimestepType.yearly:
            ans = series.unstack(component.timestep_name)
            return ans[years]

        timestep_duration = self.solution_loader.get_timestep_duration(
            scenario, component
        )

        unstacked_series = series.unstack(component.timestep_name)
        total_value = unstacked_series.multiply(timestep_duration, axis=1)  # type: ignore

        ans = pd.DataFrame(index=unstacked_series.index)

        for y in years:
            timesteps = self.solution_loader.get_timesteps(scenario, component, int(y))
            try:
                ans.insert(int(y), y, total_value[timesteps].sum(axis=1))  # type: ignore
            except KeyError:
                timestep_list = [i for i in timesteps if i in total_value]
                ans.insert(year, year, total_value[timestep_list].sum(axis=1))  # type: ignore # noqa

        return ans

    @cache
    def get_total(
        self,
        component_name: str,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
        scenario_name: Optional[str] = None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Calculates the total values of a component for a all scenarios.
        """
        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        component = self.solution_loader.components[component_name]

        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"] = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            current_total = self.get_total_per_scenario(
                scenario, component, element_name, year
            )

            if type(current_total) is pd.Series:
                current_total = current_total.rename(component_name)

            scenarios_dict[scenario_name] = current_total

        return self._concat_scenarios_dict(scenarios_dict)

    def _concat_scenarios_dict(
        self, scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"]
    ):
        scenario_names = list(scenarios_dict.keys())

        if len(scenario_names) == 1:
            ans = scenarios_dict[scenario_names[0]]
            return ans

        if isinstance(scenarios_dict[scenario_names[0]], pd.Series):
            total_value = pd.concat(
                scenarios_dict, keys=scenarios_dict.keys(), axis=1
            ).T
        else:
            try:
                total_value = pd.concat(scenarios_dict, keys=scenarios_dict.keys())  # type: ignore # noqa
            except Exception:
                total_value = pd.concat(
                    scenarios_dict, keys=scenarios_dict.keys(), axis=1
                ).T
        return total_value