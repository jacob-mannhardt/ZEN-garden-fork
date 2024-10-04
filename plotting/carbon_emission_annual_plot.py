from plotting.helpers import *




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
model_name = "PC_ct_pass_tra_ETS1and2"
# model_name = "PC_ct_pass_tra_base"
# model_name = ("PC_ct_pass_tra_cap_add_target")
# model_name = ("PC_ct_pass_tra_gen_target")

r, config = read_in_special_results(model_name)
carbon_emissions_annual = r.get_total("carbon_emissions_annual")
carbon_emissions_annual = indexmapping(carbon_emissions_annual, special_model_name=model_name)
annual_limit = r.get_total("carbon_emissions_annual_limit")
annual_limit = indexmapping(annual_limit, special_model_name=model_name)
if (annual_limit.eq(annual_limit.iloc[0])).all(axis=None):
    # If all rows are equal, create a new Series from the first row
    annual_limit_series = pd.Series(annual_limit.iloc[0], index=annual_limit.columns)

annual_limit_series = pd.Series(annual_limit.iloc[0], index=annual_limit.columns)


ETS1onlylimit = pd.Series(
    np.array([
        1894.32819089, 1802.39019386, 1680.74915163, 1559.1081094,
        1437.46706718, 1315.82602495, 1194.18498273, 1072.5439405,
        1027.01018022, 1027.01018022, 1027.01018022, 1027.01018022,
        1027.01018022, 1027.01018022, 1027.01018022
    ]),
    index=range(15),  # Index from 0 to 14
)
originalETS1and2limit = pd.Series(
    np.array([
        1894.32819089, 1802.39019386, 1680.74915163, 1277.93643597,
        1038.98750387, 800.03857177, 561.08963966, 322.14070756,
        159.2990574, 41.99116752, 0.0, 0.0, 0.0, 0.0, 0.0
    ]),
    index=range(15),  # Index from 0 to 14
)

ETS1and2limit = pd.Series(
    np.array([1721.2654343, 1623.4284652, 1410.0533978, 1116.0731706,
        897.1561892,  678.2392078,  459.3222264,  247.4078532,
        119.0760563,   19.2354483,    0.       ,    0.       ,
          0.       ,    0.       ,    0.       ]),
    index=range(15),  # Index from 0 to 14
)

if model_name.endswith("_ETS1and2") == True:
    assert np.all(ETS1and2limit.round(7).values == annual_limit_series.round(7).values)
elif model_name.endswith("_ETS1only") == True:
    assert np.all(ETS1onlylimit.round(7).values == annual_limit_series.round(7).values)
else: pass

annual_overshoot = r.get_total("carbon_emissions_annual_overshoot")
annual_overshoot = indexmapping(annual_overshoot, special_model_name=model_name)


# Start plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Top plot: Line plot for annual emissions with the annual limit in bold black
colors = ['blue', 'green', 'red', 'purple']  # Define a list of colors for different scenarios
for i, scenario in enumerate(carbon_emissions_annual.index):
    ax1.plot(carbon_emissions_annual.columns, carbon_emissions_annual.loc[scenario], label=scenario, color=colors[i])

ax1.plot(ETS1onlylimit.index, ETS1onlylimit.values, label='ETS1 Annual Limit', color='black', linewidth=2, linestyle=':')
ax1.plot(ETS1and2limit.index, ETS1and2limit.values, label='ETS1&ETS2 Annual Limit', color='black', linewidth=2, linestyle='--')
ax1.set_ylabel('Annual Emissions [Mt CO2]')
ax1.legend()

title = f"Annual Emissions and Annual Emissions Overshoot: {model_name}"
ax1.set_title(title)

# Bottom plot: Side-by-side bar chart for annual overshoot with a secondary y-axis
x = np.arange(len(annual_overshoot.columns))  # the label locations
width = 0.2  # the width of the bars

for i, scenario in enumerate(annual_overshoot.index):
    ax2.bar(x + (i - 1.5) * width, annual_overshoot.loc[scenario], width, label=scenario, color=colors[i])

ax2.set_ylabel('Annual Overshoot [Mt CO2]')
ax2.legend(loc='upper left')

# Adjust x-axis labels
ax2.set_xticks(x)
ax2.set_xticklabels(annual_overshoot.columns)

plt.show()

print("done")

