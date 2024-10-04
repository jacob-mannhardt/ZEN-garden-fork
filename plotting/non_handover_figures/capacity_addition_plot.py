from plotting.helpers import *

def capacity_plot(technology):
    # df = r.get_total("capacity_addition")
    df = r.get_total("capacity")
    df = indexmapping(df, special_model_name=model_name)
    df.columns = rename_columns(df.columns)

    df.index = df.index.rename('scenario', level=0)

    tech_df = df.loc[df.index.get_level_values('technology') == technology]
    grouped_df = tech_df.groupby(['scenario', 'technology']).sum()
    grouped_df_reset = grouped_df.reset_index()
    df = grouped_df_reset.drop(columns=['technology'])

    for scenario in df['scenario'].unique():
        # Filter the DataFrame for the current scenario
        scenario_data = df[df['scenario'] == scenario].drop(columns='scenario')
        # Plot the series
        plt.plot(scenario_data.T, label=scenario)

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Capacity [megapkm/hour]')
    plt.title('Capacity: ' + technology + '\n' + model_name)

    # Add a legend to differentiate between the scenarios
    plt.legend(title='Scenario')

    # Show the plot
    plt.show()

    print("complete")
    pass


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Beginning of code:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# model_name = "PC_ct_pass_tra_ETS1only"
# model_name = "PC_ct_pass_tra_ETS1and2"
model_name = "PC_ct_pass_tra_base"
technology = "BEV"

r, config = read_in_special_results(model_name)
capacity_plot(technology)
