from plotting.helpers import *



# model_name = "PC_ct_pass_tra_base"
# model_name = "PC_ct_pass_tra_ETS1only"
# model_name = "PC_ct_pass_tra_ETS1and2"

# model_name = ("PC_ct_pass_tra_cap_add_target")
# model_name = ("PC_ct_pass_tra_gen_target")

# model_name = ("PC_ct_pass_tra_coal_cap_phaseout")

# model_name = ("PC_ct_pass_tra_coal_gen_phaseout")
#
# model_name = ("old_PC_ct_pass_tra_coal_gen_phaseout")



desired_scenarios = ['15_1']
technology = "BEV"

r, config = read_in_special_results(model_name)

shed_demand = r.get_total("shed_demand", scenario_name='scenario_2')
shed_demand = shed_demand.groupby(level=['carrier']).sum()

flow_output = r.get_total("flow_conversion_output",scenario_name="scenario_2")
flow_output = flow_output.groupby(level=["carrier"]).sum()

flow_input = r.get_total("flow_conversion_input",scenario_name="scenario_2")
flow_input = flow_input.groupby(level=["carrier"]).sum()

# cumulative emissions:
em_df = r.get_total("carbon_emissions_cumulative", keep_raw=True)
# em_df = indexmapping(em_df, special_model_name=model_name)

# capacity
cap_df = r.get_total("capacity", keep_raw=True)
cap_df.reset_index(inplace=True)
print("reset complete")
# The 'level_0' column contains the scenario information, so we will rename it
cap_df.rename(columns={'level_0': 'scenario'}, inplace=True)

# Now we set the multi-index again, this time including the 'scenario'
cap_df.set_index(['scenario', 'technology', 'capacity_type', 'location', 'mf'], inplace=True)
print("re-index complete")
# Perform the aggregation by summing over the 'node' level of the index
cap_df = cap_df.groupby(level=['scenario', 'technology', 'capacity_type', 'mf']).sum()
print("aggregation complete")
cap_df = indexmapping(cap_df, special_model_name=model_name)

cap_df = cap_df.loc[cap_df.index.get_level_values('scenario').isin(desired_scenarios)]
# get rid of capacit_type "energy"
cap_df = cap_df[cap_df.index.get_level_values('capacity_type') != 'energy']
cap_df.index = cap_df.index.droplevel('capacity_type')

tech_cap_df = cap_df.xs(key=technology, level="technology")
tech_cap_df = sort_dataframe(tech_cap_df)



print("read in")
