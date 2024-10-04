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
bar_width = 0.4

# Create the stacked bar plot
fig, ax = plt.subplots()

# Base Hybrid (Stacked bar for each year)
ax.bar(years - bar_width/2, base_hybrid.loc['HEV'], bar_width, label='Base HEV', color='black')
ax.bar(years - bar_width/2, base_hybrid.loc['PHEV'], bar_width, bottom=base_hybrid.loc['HEV'], label='Base PHEV', color='gray')
ax.bar(years - bar_width/2, base_hybrid.loc['BEV'], bar_width, bottom=base_hybrid.loc['HEV'] + base_hybrid.loc['PHEV'], label='Base BEV', color='darkgray')

# Cap Add Hybrid (Stacked bar for each year)
ax.bar(years + bar_width/2, cap_add_hybrid.loc['HEV'], bar_width, label='Cap Add HEV', color='blue')
ax.bar(years + bar_width/2, cap_add_hybrid.loc['PHEV'], bar_width, bottom=cap_add_hybrid.loc['HEV'], label='Cap Add PHEV', color='lightblue')
ax.bar(years + bar_width/2, cap_add_hybrid.loc['BEV'], bar_width, bottom=cap_add_hybrid.loc['HEV'] + cap_add_hybrid.loc['PHEV'], label='Cap Add BEV', color='skyblue')

# Modify x-ticks to show years 2022 to 2050
xticks_major = np.arange(0, 16, 2)  # Every 2nd year for clarity
ax.set_xticks(xticks_major)
new_labels = [2022 + 2 * int(tick) for tick in xticks_major]
ax.set_xticklabels(new_labels)

# Add labels and legend
ax.set_xlabel('Year')
ax.set_ylabel('CAPEX')
ax.set_title('Transport Scetor Investment\nWith and without investment policy')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


# Show the plot
plt.tight_layout()
plt.show()

