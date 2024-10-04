from plotting.helpers import *


def manipulate_df(df):
    df.reset_index(inplace=True)
    df.rename(columns={'level_0': 'scenario'}, inplace=True)
    df.set_index(['scenario', 'technology', 'capacity_type', 'location'], inplace=True)
    df = df.groupby(level=['scenario', 'technology', 'capacity_type']).sum()
    df = df[df.index.get_level_values('scenario') == 'scenario_2']
    df = df[df.index.get_level_values('capacity_type') == 'power']
    df.index = df.index.droplevel('capacity_type')
    df.index = df.index.droplevel('scenario')

    return df


model_name = "PC_ct_pass_tra_base"
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


model_name = "PC_ct_pass_tra_base"
r, config = read_in_special_results(model_name)
cost_capex_base = r.get_full_ts("cost_capex")
cost_capex_base = manipulate_df(cost_capex_base)
base_hybrid = cost_capex_base.loc[['PHEV', 'HEV', "BEV"]]


model_name = "PC_ct_pass_tra_cap_add_target"
r, config = read_in_special_results(model_name)
cost_capex_cap_add = r.get_full_ts("cost_capex")
cost_capex_cap_add = manipulate_df(cost_capex_cap_add)
cap_add_hybrid = cost_capex_cap_add.loc[['PHEV', 'HEV', "BEV"]]


# Create the x-axis (years 0 to 14)
years = np.arange(15)

# Set the width of the bars
bar_width = 0.12

# Plot the bars for each series
fig, ax = plt.subplots()

# Base Hybrid HEV (Black)
ax.bar(years - bar_width*2.5, base_hybrid.loc['HEV'], bar_width, label='Base HEV', color='black')

# Base Hybrid PHEV (Lighter black)
ax.bar(years - bar_width*1.5, base_hybrid.loc['PHEV'], bar_width, label='Base PHEV', color='gray')

# Base Hybrid BEV (Dark Gray)
ax.bar(years - bar_width/2, base_hybrid.loc['BEV'], bar_width, label='Base BEV', color='darkgray')

# Cap Add HEV (Blue)
ax.bar(years + bar_width/2, cap_add_hybrid.loc['HEV'], bar_width, label='Cap Add HEV', color='blue')

# Cap Add PHEV (Lighter blue)
ax.bar(years + bar_width*1.5, cap_add_hybrid.loc['PHEV'], bar_width, label='Cap Add PHEV', color='lightblue')

# Cap Add BEV (Light Blue)
ax.bar(years + bar_width*2.5, cap_add_hybrid.loc['BEV'], bar_width, label='Cap Add BEV', color='skyblue')

# Add labels and legend
ax.set_xlabel('Year')
ax.set_ylabel('Values')
ax.set_title('PHEV, HEV, and BEV over Time')
ax.set_xticks(years)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()



diff = cost_capex_base - cost_capex_cap_add

df = diff.T.sum()

# Create a bar chart
plt.figure(figsize=(10, 6))
df.plot(kind='bar')

# Adding labels and title
plt.xlabel('Technology')
plt.ylabel('Sum of Differences')
plt.title('Differences in CAPEX (Base vs Investment Policy) until 2050 (5_1)')

# Rotating x-axis labels for better readability
plt.xticks(rotation=90)

# Show the plot
plt.tight_layout()
plt.show()

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
