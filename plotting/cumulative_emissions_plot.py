import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors  # this is what I recently added
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np
import importlib
import os
import json
import pandas as pd
import math
from pandas import IndexSlice as idx
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.lines import Line2D
from postresults import PostResult
from matplotlib.patches import Rectangle


from zen_garden.postprocess.results.results import Results
from zen_garden.model.default_config import Config,Analysis,Solver,System
from plotting.helpers import *




def cumulative_emissions_plot():
    print("Plotting Cumulative Emissions")
    # Set the color palette from Seaborn
    sns.set_palette('deep')

    # Initialize an empty list to store DataFrames from each model
    dfs = []
    if type(model_name) is not str:
        for name in model_name:
            r, config = read_in_results(name)
            temp_df = r.get_total("carbon_emissions_cumulative")
            # indexing
            temp_df = indexmapping(temp_df, special_model_name=name)
            dfs.append(temp_df)
    else:
        r, config = read_in_results(model_name)
        temp_df = r.get_total("carbon_emissions_cumulative")
        # indexing
        if desired_scenarios != "dont_index":
            temp_df = indexmapping(temp_df, special_model_name=model_name)
        dfs.append(temp_df)

    # Concatenate all DataFrames in the list to a single DataFrame
    df = pd.concat(dfs, axis=0, join='outer')

    # Select only the desired rows
    if desired_scenarios != None:
        if desired_scenarios != "dont_index":
            df = df.loc[desired_scenarios]
    df_T = df.T

    index_years = np.arange(15)
    data = {
        "15_1": np.cumsum(np.random.normal(500, 100, size=15)) + 1000,
        "15_3": np.cumsum(np.random.normal(520, 150, size=15)) + 1200,
        "15_5": np.cumsum(np.random.normal(550, 80, size=15)) + 950,
        "10_1": np.cumsum(np.random.normal(530, 90, size=15)) + 1100,
        "10_3": np.cumsum(np.random.normal(510, 110, size=15)) + 1050,
        "10_5": np.cumsum(np.random.normal(550, 80, size=15)) + 950,
        "5_1": np.cumsum(np.random.normal(530, 90, size=15)) + 1100,
        "5_3": np.cumsum(np.random.normal(510, 110, size=15)) + 1050,
        "5_5": np.cumsum(np.random.normal(550, 80, size=15)) + 950,
    }

    df_T = pd.DataFrame(data, index=index_years)


    df_T.index = 2022 + (df_T.index * 2)


    # Dictionary to store dataframes
    investment_foresight_dfs = {}
    # Extract unique prefixes and create individual dataframes
    for col in df_T.columns:
        prefix = col.split('_')[0]
        if prefix not in investment_foresight_dfs:
            investment_foresight_dfs[prefix] = df_T.filter(regex=f'^{prefix}_')

    # New dictionary to store min/max series
    min_max_series = {}
    # Calculate min and max for each dataframe across all columns row-wise
    for key, df in investment_foresight_dfs.items():
        min_series = df.min(axis=1)
        max_series = df.max(axis=1)
        combined_series = pd.concat([min_series, max_series], axis=1)
        combined_series.columns = ['Min', 'Max']
        min_max_series[key] = combined_series

    print("stop here")

    # Setting up the plot
    plt.figure(figsize=(10, 6))

    carbon_budget = 16.58
    j = 0  # Color index

    # Base color
    base_color = 'purple'
    # Mapping base colors to matplotlib colormaps
    color_maps = {
        'blue': 'Blues',
        'green': 'Greens',
        'yellow': 'YlOrBr',  # Yellow to orange/brown gradient
        'orange': 'Oranges',
        'red': 'Reds',
        'purple': 'Purples',
        'turquoise': 'GnBu'  # Green to blue gradient, for a turquoise effect
    }


    # Determine the number of keys in the dictionary
    num_keys = len(min_max_series)

    # Base RGB color from a name (using matplotlib's color table)
    rgb_color = mcolors.to_rgb(base_color)


    # # Iterate over each key in the investment_foresight_dfs dictionary
    # for key, df in investment_foresight_dfs.items():
    #     num_columns = len(df.columns)
    #     color = colors[j % len(colors)]
    #     # line_style = line_styles[i % len(line_styles)]
    #     for i, column in enumerate(df.columns):
    #         # Select color and line style based on column index
    #         # Plot each column
    #         plt.plot(df.index, df[column], label=f'{key} {column}', color=color)  # ,linestyle=line_style
    #     j += 1

    # Plotting loop
    for j, (key, df) in enumerate(min_max_series.items()):
        # Calculate alpha, increasing with each key
        alpha_value = (j + 1) / float(num_keys)

        # Fill the area between min and max with the base color and increasing alpha
        plt.fill_between(df.index, df['Min']/1000, df['Max']/1000, label=f'{str(int(key)*2) + "_years"}', color=rgb_color[:3] + (alpha_value,), alpha=alpha_value)

    # Draw a horizontal line at y = carbon_budget
    plt.axhline(y=carbon_budget, color='black', linestyle='--', label='Carbon Budget')
    # Adding a shaded rectangle above the carbon budget line
    current_axis = plt.gca()  # Get current axis
    height = plt.ylim()[1] - carbon_budget  # Calculate height of the rectangle
    rectangle = Rectangle((df.index[0]-30, carbon_budget), df.index[-1] - df.index[0] + 50, height, color='grey', alpha=0.1)
    current_axis.add_patch(rectangle)

    tick_positions = np.linspace(start=df_T.index.min(), stop=df_T.index.max(), num=len(df_T.index))
    plt.xticks(ticks=tick_positions, labels=[str(int(x)) for x in tick_positions])

    # Adding titles and labels
    plt.title('Base Scenario')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Emissions [Gt CO2]')
    plt.legend(title="Investment foresight horizon")

    # Show the plot
    plt.show()

    #
    # fig, ax = plt.subplots(figsize=(12, 8))
    # carbon_budget = 16.58
    #
    # num_years = len(df_T)
    # # Creating an array for the x-axis positions
    # x = np.arange(num_years)
    # if desired_scenarios != "dont_index":
    #     for i, scenario in enumerate(df_T.columns):
    #         plt.plot(x, df_T[scenario] / 1000, label=df_T.columns[i])
    # else:
    #     plt.plot(x, df_T / 1000)
    # # Adding labels and title
    # plt.xlabel('Year', fontsize=14)
    # plt.ylabel('Total Carbon Emissions [Gt CO₂]', fontsize=14)
    #
    # # generate figure title:
    # #fig_title = get_fig_title(plot_name)
    # # plt.title(fig_title)
    #
    # # Adjusting x-ticks
    # plt.xticks(x, (df_T.index) * 2 + 2022)
    # # Adding a legend
    # plt.legend(title="Operation Foresight Horizon:")
    # # Adding a thin red horizontal line at y = carbon_emissions_budget
    # plt.axhline(y=carbon_budget, color='black', linestyle='-', linewidth=0.75, xmin=0.25, zorder=1, alpha=0.5)
    # # Labeling the red line
    # plt.text(14, carbon_budget - 0.04 * plt.ylim()[1], "Carbon Emission Budget", va='bottom', ha='right', color='black',
    #          fontsize=12)
    # # Show the plot
    # plt.show(block=True)
    # return


if __name__ == "__main__":
    model_name = "PI_small_drastic_coal_capacity_phaseout"
    # model_name = ["cons_nolead_1to4", "cons_nolead_init", "cons_lead_1to4",
    #               "cons_lead_init", "lesscons_lead_1to4", "lesscons_lead_init",
    #               "varcons_lead_1to4", "varcons_lead_init"]

    desired_scenarios = ['15_1', '15_3', '15_5', '10_1', '10_3', '10_1', '5_1', '5_3', '5_1']
    desired_scenarios = "dont_index"
    cumulative_emissions_plot()
