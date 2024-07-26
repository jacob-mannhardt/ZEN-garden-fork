import copy
import os

import numpy as np
import pandas as pd
from zen_garden.postprocess.results.results import Results
# from zen_garden.postprocess.old_results import Results
# from zen_garden.postprocess.results.results import Results as NewResults

class PostResult(Results):
    def __init__(self, path):
        """ initialize the result object"""
        super().__init__(path)
        # r = NewResults(path)
        # general settings
        self.set_settings()

    def set_settings(self):
        """ set the settings for the postprocessing"""
        self.analysis = self.get_this_analysis()
        self.system = self.get_this_system()
        self.opti_steps = self.get_years()
        self.opti_years = self.calculate_years(reference_year=self.system.reference_year, opti_steps=self.opti_steps)
        # techs
        self.conversion_techs = self.system.set_conversion_technologies
        self.transport_techs = self.system.set_transport_technologies
        self.storage_techs = self.system.set_storage_technologies

    def calculate_years(self,reference_year,opti_steps):
        """ calculate the years for the optimization"""
        opti_years = [reference_year + self.system.interval_between_years * step for step in opti_steps]
        return opti_years

    def get_this_analysis(self,scenario_name = None):
        """ get the analysis for this scenario"""
        if scenario_name is not None:
            return self.get_analysis(scenario = scenario_name)
        else:
            return self.get_analysis()

    def get_this_system(self,scenario_name = None):
        """ get the system for this scenario"""
        if scenario_name is not None:
            return self.get_system(scenario = scenario_name)
        else:
            return self.get_system()

    def _get_input_carriers(self):
        """ get the input carriers for a carrier"""
        input_carriers = self.get_df("set_input_carriers")
        input_carriers = input_carriers[next(iter(input_carriers))]
        input_carriers = input_carriers.str.split(",").map(lambda c: [] if c == [""] else c)
        return input_carriers

    def _get_output_carriers(self):
        """ get the output carriers for a carrier"""
        output_carriers = self.get_df("set_output_carriers")
        output_carriers = output_carriers[next(iter(output_carriers))]
        output_carriers = output_carriers.str.split(",").map(lambda c: [] if c == [""] else c)
        return output_carriers

    def _get_reference_carriers(self):
        """ get the reference carriers for a carrier"""
        reference_carriers = self.get_df("set_reference_carriers")
        reference_carriers = reference_carriers[next(iter(reference_carriers))]
        reference_carriers = reference_carriers.str.split(",").map(lambda c: [] if c == [""] else c)
        return reference_carriers

    def sector_techs(self,carrier,carrier_type = "output"):
        """ get the sector technologies for a carrier
        :param carrier: the carrier
        :param carrier_type: the carrier type
        """
        if carrier_type == "output":
            tech2carrier = self._get_output_carriers()
        elif carrier_type == "input":
            tech2carrier = self._get_input_carriers()
        elif carrier_type == "reference":
            tech2carrier = self._get_reference_carriers()
        else:
            raise ValueError(f"Carrier type {carrier_type} not known")
        sector_techs = []
        for tech in tech2carrier.index:
            if carrier in tech2carrier.loc[tech]:
                sector_techs.append(tech)
        return sector_techs

    def get_sector_type_techs(self,carrier,technology_type = "conversion",carrier_type = "reference"):
        """ get the technologies for a sector and type
        :param carrier: the carrier
        :param technology_type: the technology type
        """
        if technology_type == "conversion":
            type_techs = self.conversion_techs
        elif technology_type == "transport":
            type_techs = self.transport_techs
            carrier_type = "reference"
        elif technology_type == "storage":
            type_techs = self.storage_techs
            carrier_type = "reference"
        else:
            raise ValueError(f"Technology type {technology_type} not known")
        sector_techs = self.sector_techs(carrier,carrier_type = carrier_type)
        techs = []
        for tech in sector_techs:
            if tech in type_techs:
                techs.append(tech)
        return techs

    def get_all_sector_emissions(self,scenario = None):
        """ get the emissions for all sectors"""
        d = self.get_total("demand", scenario_name=scenario).groupby(level=0).sum()
        dcs = d[(d != 0).all(axis=1)].index.unique()
        e = {}
        set_dict = self.get_set_dict(scenario)
        var_dict = self.get_var_dict(scenario_name=scenario)
        for dc in dcs:
            e[dc] = self.get_sector_emissions(carrier=dc,scenario = scenario,var_dict = var_dict,set_dict = set_dict)
        # add DAC
        if "DAC" in self.system.set_conversion_technologies:
            share = pd.Series(index=self.opti_steps, name="year", data=1)
            e["DAC"] = self.recursive_emissions(tech="DAC",parent_carrier="none",emissions={},var_dict = var_dict,set_dict = set_dict,share=share)
        em = self.emissions_from_tree(e, scenario_name=scenario)
        # tot_e = self.recursive_sum(e)
        cec = self.get_total("carbon_emissions_carrier", scenario_name=scenario).groupby(level=0).sum()
        cec = cec[(cec != 0).all(axis=1)]
        emc = em.groupby(level=1).sum()
        delc = cec - emc
        return e, em

    def get_sector_emissions(self,carrier,scenario = None,var_dict = None,set_dict = None):
        """ get the emissions for a sector"""
        print(f"Calculate annual emissions for carrier {carrier} for scenario {self.name}")
        # sets
        set_dict = self.get_set_dict(scenario_name=scenario, set_dict=set_dict)
        var_dict = self.get_var_dict(scenario_name=scenario, var_dict=var_dict)
        emissions_empty = {}
        emissions = {}
        # direct emissions of carrier
        ec = var_dict["emissions_carrier"].loc[carrier]
        if ec.abs().sum() > 0:
            emissions_empty["fuel"] = ec
        # conversion technologies, remove those techs that only have the carrier as a secondary output
        techs = [t for t,c in self._get_output_carriers().items() if carrier in c and (len(self._get_output_carriers()[t]) == 1 or carrier in self._get_reference_carriers()[t])]
        # if carrier in var_dict["flow_in"].index.get_level_values(1):
        #     share = 1-(var_dict["flow_in"].loc[(slice(None),carrier),:].sum() / (var_dict["demand"].loc[carrier]+var_dict["flow_in"].loc[(slice(None),carrier),:].sum()))
        # else:
        #     share = pd.Series(index=self.opti_steps, name="year", data=1.0)
        # share = var_dict["demand"].loc[carrier] / var_dict["flow_out"].loc[(slice(None),carrier),:].sum()
        # if carrier == "electricity":
        #     a=1
        # TODO double counting of electricity emissions -> e.g. diff lignite in other sectors is exactly lignite from other sectors
        share = pd.Series(index=self.opti_steps, name="year", data=1.0)
        for tech in techs:
            em = self.recursive_emissions(tech,carrier, emissions_empty, var_dict, set_dict,share=share)
            if em:
                emissions[tech] = em
        return emissions

    def get_var_dict(self, scenario_name=None, var_dict = None):
        if var_dict is None:
            # tech emissions
            emissions_tech = self.get_total("carbon_emissions_technology", scenario_name=scenario_name).groupby("technology").sum()
            # opex
            emissions_carrier = self.get_total("carbon_emissions_carrier", scenario_name=scenario_name).groupby("carrier").sum()
            # input flow, output flow
            flow_out = self.get_total("flow_conversion_output", scenario_name=scenario_name).groupby(["technology", "carrier"]).sum()
            flow_in = self.get_total("flow_conversion_input", scenario_name=scenario_name).groupby(["technology", "carrier"]).sum()
            demand = self.get_total("demand", scenario_name=scenario_name).groupby(["carrier"]).sum()
            # import
            flow_import = self.get_total("flow_import", scenario_name=scenario_name).groupby("carrier").sum()
            var_dict = {"emissions_tech": emissions_tech, "emissions_carrier": emissions_carrier, "flow_in": flow_in,
                        "flow_out": flow_out, "demand": demand,
                        "flow_import": flow_import}
        return var_dict

    def get_npc(self):
        """ calculates the NPV for an individual scenario """
        # system = self.system
        years = np.array(range(self.opti_years[0], self.opti_years[-1] + 1))
        steps = np.floor(np.array(range(len(years))) / 2).astype(int)
        discount_rate = self.get_total("discount_rate").squeeze()
        NPC_OPEX = {}
        NPC_CAPEX = {}
        costs = ["cost_opex_total", "cost_carrier_total", "cost_capex_total"]

        df = {}
        for c in costs:
            df[c] = self.get_total(c)
        df = pd.concat(df, keys=df.keys())
        df.loc['cost_opex_total'] = (df.loc['cost_opex_total'].values + df.loc['cost_carrier_total'].values)
        df = df.drop('cost_carrier_total')
        cost_opex = df.loc["cost_opex_total"]
        cost_capex = df.loc["cost_capex_total"]

        for idx_year, year in enumerate(years):
            NPC_OPEX[year] = cost_opex[steps[idx_year]] * (1 / (1 + discount_rate)) ** (year - years[0])
            NPC_CAPEX[year] = cost_capex[steps[idx_year]] * (1 / (1 + discount_rate)) ** (year - years[0])
        dfs_OPEX = [pd.DataFrame({year: data}) for year, data in NPC_OPEX.items()]
        df_OPEX = pd.concat(dfs_OPEX, axis=1)
        dfs_CAPEX = [pd.DataFrame({year: data}) for year, data in NPC_CAPEX.items()]
        df_CAPEX = pd.concat(dfs_CAPEX, axis=1)

        return df_OPEX, df_CAPEX

    def get_set_dict(self, scenario_name=None, set_dict = None):
        if set_dict is None:
            ref_carrier = self.get_df("set_reference_carriers", scenario_name=scenario_name)
            input_carrier = self.get_df("set_input_carriers", scenario_name=scenario_name)
            output_carrier = self.get_df("set_output_carriers", scenario_name=scenario_name)
            if "set_retrofitting_base_technologies" in self.solution_loader.components:
                retrofit_base_techs = self.get_df("set_retrofitting_base_technologies", scenario_name=scenario_name)
            else:
                retrofit_base_techs = {next(iter(self.solution_loader.scenarios)): pd.Series()}
                # this here is pretty hacky:
                if scenario_name is not None:
                    retrofit_base_techs[scenario_name] = retrofit_base_techs.pop('scenario_')
            if scenario_name is not None:
                ref_carrier = ref_carrier[scenario_name]
                input_carrier = input_carrier[scenario_name]
                output_carrier = output_carrier[scenario_name]
                retrofit_base_techs = retrofit_base_techs[scenario_name]
            else:
                ref_carrier = ref_carrier[next(iter(ref_carrier))]
                input_carrier = input_carrier[next(iter(input_carrier))]
                output_carrier = output_carrier[next(iter(output_carrier))]
                retrofit_base_techs = retrofit_base_techs[next(iter(retrofit_base_techs))]
            set_dict = {"ref_carrier": ref_carrier, "input_carrier": input_carrier, "output_carrier": output_carrier,
                        "retrofit_base_techs": retrofit_base_techs}
        return set_dict

    def recursive_emissions(self,tech, parent_carrier,emissions, var_dict, set_dict,share,downstream=False,share_threshold = 1e-4):
        """ recursive calculation of emissions for a technology
        :param tech: the technology
        :param parent_carrier: the parent carrier
        :param emissions: the emissions
        :param var_dict: the variable dictionary
        :param set_dict: the set dictionary
        TODO clean up
        """
        emissions = copy.deepcopy(emissions)
        et = var_dict["emissions_tech"].loc[tech]
        # direct emissions
        if et.abs().sum() > 0:
            emissions["tech"] = et*share
        index_in = var_dict["flow_in"].index
        index_out = var_dict["flow_out"].index
        # those carriers that are an input to tech but not the parent carrier
        input_carrier_tech = pd.Index(self._get_input_carriers()[tech]).difference([parent_carrier])
        # those carriers that are an output to tech but not the parent carrier
        output_carrier_tech = pd.Index(self._get_output_carriers()[tech]).difference([parent_carrier])

        # inputs
        for input_carrier_pre in input_carrier_tech:
            emissions[input_carrier_pre] = {}
            consumption_share = (
                    var_dict["flow_in"].loc[tech, input_carrier_pre] /
                    (var_dict["flow_in"].loc[(slice(None),input_carrier_pre),:]
                     + var_dict["demand"].loc[input_carrier_pre]).sum())
            new_share = consumption_share * share
            # direct fuel emissions
            if not downstream:
                fuel_emissions = new_share * var_dict["emissions_carrier"].loc[input_carrier_pre]
                if fuel_emissions.abs().sum() > 0:
                    emissions[input_carrier_pre]["fuel"] = fuel_emissions
                # transport and storage emissions
                transport_techs = self.get_sector_type_techs(input_carrier_pre,"transport")
                storage_techs = self.get_sector_type_techs(input_carrier_pre,"storage")
                aux_techs = transport_techs + storage_techs
                aux = pd.Series(index=self.opti_steps, name="year", data=0)
                for aux_tech in aux_techs:
                    aux += new_share*var_dict["emissions_tech"].loc[aux_tech]
                if aux.abs().sum() > 0:
                    emissions[input_carrier_pre]["aux"] = aux
            else:
                fe = new_share * var_dict["emissions_carrier"].loc[input_carrier_pre]
            if (new_share <= share_threshold).all():
                emissions = {}
                continue
            techs_pre = index_out.get_level_values("technology")[index_out.get_level_values("carrier") == input_carrier_pre]
            # don't allocate emissions to those that have multiple outputs and input_carrier_pre is not the reference carrier
            techs_pre = [t for t in techs_pre if
                     len(self._get_output_carriers()[t]) == 1 or input_carrier_pre in self._get_reference_carriers()[t]]
            emissions_empty = {}
            for tech_pre in techs_pre:
                emissions_pre = self.recursive_emissions(tech_pre,input_carrier_pre, emissions_empty, var_dict, set_dict,share=new_share,downstream=downstream)
                if emissions_pre:
                    emissions[input_carrier_pre][tech_pre] = emissions_pre
            if not emissions[input_carrier_pre]:
                del emissions[input_carrier_pre]
        # outputs
        for output_carrier_pre in output_carrier_tech:
            emissions[output_carrier_pre] = {}
            production_share = (
                    var_dict["flow_out"].loc[tech, output_carrier_pre] /
                    (var_dict["flow_out"].loc[(slice(None),output_carrier_pre),:]
                    + var_dict["flow_import"].loc[output_carrier_pre]).sum()
                     )
            new_share = production_share * share
            # transport and storage emissions
            transport_techs = self.get_sector_type_techs(output_carrier_pre,"transport")
            storage_techs = self.get_sector_type_techs(output_carrier_pre,"storage")
            aux_techs = transport_techs + storage_techs
            aux = pd.Series(index=self.opti_steps, name="year", data=0)
            for aux_tech in aux_techs:
                aux += new_share*var_dict["emissions_tech"].loc[aux_tech]
            if aux.abs().sum() > 0:
                emissions[output_carrier_pre]["aux"] = aux
            if (new_share <= share_threshold).all():
                emissions = {}
                continue
            techs_pre = index_in.get_level_values("technology")[index_in.get_level_values("carrier") == output_carrier_pre]
            emissions_empty = {}
            for tech_pre in techs_pre:
                emissions_pre = self.recursive_emissions(tech_pre,output_carrier_pre, emissions_empty, var_dict, set_dict,share=new_share,downstream=True)
                if emissions_pre:
                    emissions[output_carrier_pre][tech_pre] = emissions_pre
            if not emissions[output_carrier_pre]:
                del emissions[output_carrier_pre]
        # retrofitting
        rt = [r for r,b in set_dict["retrofit_base_techs"].items() if b == tech]
        assert len(rt) <= 1, f"More than one retrofit base technology for {tech}. Currently not supported"
        for retrofit_tech in rt:
            if (share <= share_threshold).all():
                continue
            emissions_empty = {}
            emissions_retrofit = self.recursive_emissions(retrofit_tech, None, emissions_empty, var_dict, set_dict,share=share,downstream=downstream)
            if emissions_retrofit:
                emissions[retrofit_tech] = emissions_retrofit
        return copy.deepcopy(emissions)

    def recursive_sum(self,d):
        # Base case: if the value is a Pandas Series, return it
        if isinstance(d, pd.Series):
            return d

        # Recursive case: if the value is a dictionary, sum the values
        elif isinstance(d, dict):
            return sum(self.recursive_sum(value) for value in d.values())

        # If the value is neither a Pandas Series nor a dictionary, return None
        else:
            return None

    def recursive_lookup(self,e,em,c="none"):
        for k in e:
            if k == "tech" or k == "aux":
                em.loc["tech"] += e[k]
            elif k == "fuel":
                em.loc[c] += e[k]
            else:
                em = self.recursive_lookup(e[k],em,c=k)
        return em

    def emissions_from_tree(self,e, scenario_name=None):
        em_sec = pd.DataFrame(index=self.system.set_carriers+["tech"],columns=self.opti_steps,data=0,dtype=float)
        em = {}
        for sector in e:
            em[sector] = self.recursive_lookup(e[sector],em_sec.copy())
        em = pd.concat(em,keys=em.keys())
        em = em[(em != 0).any(axis=1)]
        cc = self.get_total("carbon_emissions_carrier", scenario_name=scenario_name).groupby(level=0).sum()
        cc = cc[(cc != 0).any(axis=1)]
        tc = self.get_total("carbon_emissions_technology", scenario_name=scenario_name).groupby(level=0).sum().sum()
        cc.loc["tech"] = tc
        return em
