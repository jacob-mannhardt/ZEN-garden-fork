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
    print("Plotting Cumulative Emissions (individual scenarios)")
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
    df.columns = 2022 + (df.columns * 2)
    plt.figure(figsize=(10, 6))
    for label in df.index:
        plt.plot(df.columns, df.loc[label]/1000, label=label)
    carbon_budget = 16.58
    # carbon_budget = 12.23. # budget without the transport sector
    plt.axhline(y=carbon_budget, color='black', linestyle='--', label='Carbon Budget')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Emissions [Gt CO2]')
    plt.title(model_name + ', TSA:50')
    plt.legend(title="Series")
    plt.grid(True)
    plt.ylim(top=18.5)

    plt.show()

def cumulative_emissions_plot_grouped():
    print("Plotting Cumulative Emissions")
    # Set the color palette from Seaborn
    sns.set_palette('deep')

    dfs = []
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
    df_T.index = 2022 + (df_T.index * 2)


    # Dictionary to store dataframes
    investment_foresight_dfs = {}
    # Extract unique prefixes and create individual dataframes
    for col in df_T.columns:
        prefix = col.split('_')[0]
        if prefix not in investment_foresight_dfs:
            investment_foresight_dfs[prefix] = df_T.filter(regex=f'^{prefix}_')

    # Dictionary to store dataframes
    operation_foresight_dfs = {}
    # Extract unique prefixes and create individual dataframes
    for col in df_T.columns:
        second_prefix = col.split('_')[1]
        if second_prefix not in operation_foresight_dfs:
            operation_foresight_dfs[second_prefix] = df_T.filter(regex=f'_{second_prefix}')

    # foresight_dfs = investment_foresight_dfs
    foresight_dfs = operation_foresight_dfs


    # New dictionary to store min/max series
    min_max_series = {}
    # Calculate min and max for each dataframe across all columns row-wise
    for key, df in foresight_dfs.items():
        min_series = df.min(axis=1)
        max_series = df.max(axis=1)
        combined_series = pd.concat([min_series, max_series], axis=1)
        combined_series.columns = ['Min', 'Max']
        min_max_series[key] = combined_series

    # Setting up the plot
    plt.figure(figsize=(10, 6))

    carbon_budget = 16.58
    # carbon_budget = 12.23

    j = 0  # Color index

    # Base color
    base_color = 'green'
    # Mapping base colors to matplotlib colormaps
    color_maps = {
        'blue': 'Blues',
        'green': 'Greens',
        'yellow': 'YlOrBr',  # Yellow to orange/brown gradient
        'orange': 'Oranges',
        'red': 'Reds',
        'purple': 'Purples',
        'turquoise': 'GnBu',  # Green to blue gradient, for a turquoise effect
        'black': 'Greys'
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

    if foresight_dfs == investment_foresight_dfs:
        legend_title = "Investment foresight horizon"
    elif foresight_dfs == operation_foresight_dfs:
        legend_title = "Operation foresight horizon"

    # Adding titles and labels
    plt.title(model_name + ', TSA:20')  # 'Base Scenario', 'Renewable Capacity Policy', 'Renewable Generation Policy', 'ETS 1 only Policy'
    plt.xlabel('Year')
    plt.ylabel('Cumulative Emissions [Gt CO2]')
    plt.legend(title=legend_title)
    plt.ylim(top=18.5)

    # Show the plot
    plt.show()


def cumulative_emissions_plot_grouped2x2(model_names):
    print("Plotting Cumulative Emissions for Multiple Models")

    # Setting up the figure for 2x2 subplots with shared x and y axes
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 10), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the 2x2 grid of axes for easy indexing

    # Set the color palette from Seaborn
    sns.set_palette('deep')

    for idx, model_name in enumerate(model_names):
        dfs = []
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
        df_T.index = 2022 + (df_T.index * 2)

        # Dictionary to store dataframes
        investment_foresight_dfs = {}
        for col in df_T.columns:
            prefix = col.split('_')[0]
            if prefix not in investment_foresight_dfs:
                investment_foresight_dfs[prefix] = df_T.filter(regex=f'^{prefix}_')

        # Dictionary to store dataframes
        operation_foresight_dfs = {}
        for col in df_T.columns:
            second_prefix = col.split('_')[1]
            if second_prefix not in operation_foresight_dfs:
                operation_foresight_dfs[second_prefix] = df_T.filter(regex=f'_{second_prefix}')

        foresight_dfs = operation_foresight_dfs

        # New dictionary to store min/max series
        min_max_series = {}
        for key, df in foresight_dfs.items():
            min_series = df.min(axis=1)
            max_series = df.max(axis=1)
            combined_series = pd.concat([min_series, max_series], axis=1)
            combined_series.columns = ['Min', 'Max']
            min_max_series[key] = combined_series

        # Choose the current subplot axis
        ax = axs[idx]

        # Set the base color
        if model_name == "PC_ct_pass_tra_base":
            base_color = 'black'
        elif model_name == "PC_ct_pass_tra_cap_add_target":
            base_color = 'blue'
        elif model_name == "PC_ct_pass_tra_ETS1and2":
            base_color = 'red'
        elif model_name == "PC_ct_pass_tra_gen_target":
            base_color = 'green'
        num_keys = len(min_max_series)
        rgb_color = mcolors.to_rgb(base_color)

        # Plot each min/max series in the current subplot
        for j, (key, df) in enumerate(min_max_series.items()):
            alpha_value = (j + 1) / float(num_keys)
            fill_color = rgb_color[:3] + (alpha_value,)  # Adjust the alpha for the color
            ax.fill_between(df.index, df['Min'] / 1000, df['Max'] / 1000, label=f'{str(int(key) * 2) + "_years"}',
                            color=fill_color, alpha=alpha_value)

            # Connect the first point of the series to (2021, 0)
            first_year = df.index[0]
            first_min = df['Min'].iloc[0] / 1000
            first_max = df['Max'].iloc[0] / 1000
            ax.plot([2021, first_year], [0, first_min], color=fill_color, linewidth=0.8)  # Connecting to the min series
            ax.plot([2021, first_year], [0, first_max], color=fill_color, linewidth=0.8)  # Connecting to the max series

        # Draw the carbon budget line and add the shaded area
        carbon_budget = 16.58
        ax.set_ylim(0, 19.5)  # Fix the y-axis max to 19.5
        ax.axhline(y=carbon_budget, color='black', linestyle='--', label='Carbon Budget')
        height = ax.get_ylim()[1] - carbon_budget
        rectangle = Rectangle((df.index[0] - 30, carbon_budget), df.index[-1] - df.index[0] + 50, height, color='grey',
                              alpha=0.1)
        ax.add_patch(rectangle)

        if model_name == "PC_ct_pass_tra_base":
            model_name_str = "Base Scenario"
        elif model_name == "PC_ct_pass_tra_ETS1and2":
            model_name_str = "Emission Policy"
        elif model_name == "PC_ct_pass_tra_cap_add_target":
            model_name_str = "Investment Policy"
        elif model_name == "PC_ct_pass_tra_gen_target":
            model_name_str = "Operation Policy"

        # Set titles and labels for the subplots
        ax.set_title(model_name_str, loc='left', x=0.01 ,y=0.93, fontsize=12)
        ax.set_xlabel('Year')
        ax.set_ylabel('Cumulative Emissions [Gt CO2]')

        # Define x-axis ticks every two years from 2022 to 2050
        ax.set_xticks(np.arange(2022, 2052, 2))
        ax.set_xlim([2020, 2052])
        # Get the current tick labels and only display every second one
        tick_labels = [label if i % 2 == 0 else '' for i, label in enumerate(np.arange(2022, 2052, 2))]
        ax.set_xticklabels(tick_labels)

        # Enable minor ticks
        ax.minorticks_on()

        # Define the position of the minor ticks (every year)
        ax.set_xticks(np.arange(2021, 2050, 2), minor=True)

        # Optional: You can customize the appearance of minor ticks if needed (e.g., make them shorter)
        ax.tick_params(axis='x', which='minor', length=2)

        # Move the legend to the bottom right of each subplot
        ax.legend(title='Operation foresight horizon', loc='lower right')

    # Hide x-axis ticks on the upper plots
    for ax in axs[:2]:
        ax.xaxis.set_visible(False)

    # Hide y-axis ticks on the right-side plots
    for ax in axs[1::2]:
        ax.yaxis.set_visible(False)

    # Adjust the layout so the subplots are touching (no space between them)
    plt.subplots_adjust(hspace=0, wspace=0)

    # Show the plot
    plt.show()



if __name__ == "__main__":
    desired_scenarios = ['15_1', '15_5', '5_1', '5_5']
    model_names = ["PC_ct_pass_tra_base", "PC_ct_pass_tra_cap_add_target",
                   "PC_ct_pass_tra_gen_target", "PC_ct_pass_tra_ETS1and2"]
    cumulative_emissions_plot_grouped2x2(model_names)


    # individual plots (not in overleaf doc):
    # model_name = "PC_ct_pass_tra_base"
    # model_name = "PC_ct_pass_tra_cap_add_target"
    # model_name = "PC_ct_pass_tra_gen_target"
    # model_name = "PC_ct_pass_tra_ETS1and2"

    # cumulative_emissions_plot()
    # cumulative_emissions_plot_grouped()
