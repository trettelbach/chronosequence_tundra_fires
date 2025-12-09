import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import numpy as np
# import pdb
# Show full lists/arrays in printout
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

plt.rcParams["font.family"] = "Arial"


# Function to convert memory usage to MB
def convert_memory_to_mb(memory_str):
    if 'GB' in memory_str:
        return float(memory_str.replace('GB', '')) * 1024
    elif 'MB' in memory_str:
        return float(memory_str.replace('MB', ''))
    else:
        return float(memory_str)  # Assuming the value is already in MB if no unit is specified

# Function to convert CPU usage to float and remove '%' symbol if present
def convert_cpu_to_float(cpu_str):
    if isinstance(cpu_str, str) and '%' in cpu_str:
        return float(cpu_str.replace('%', ''))
    return float(cpu_str)


# Function to convert realtime to seconds
def convert_duration_to_seconds(duration_str):
    # Check for milliseconds format like '200ms'
    if re.match(r'^\d+(\.\d+)?ms$', duration_str):
        return float(duration_str.replace('ms', '')) / 1000

    # Check for seconds format like '2.2s'
    if re.match(r'^\d+(\.\d+)?s$', duration_str):
        return float(duration_str.replace('s', ''))

    # Check for formats like '1h 16m 27s', '1h 9m 23s', '13m 9s'
    hours = minutes = seconds = 0
    match = re.search(r'(\d+)h', duration_str)
    if match:
        hours = int(match.group(1))
    match = re.search(r'(\d+)m', duration_str)
    if match:
        minutes = int(match.group(1))
    match = re.search(r'(\d+(\.\d+)?)s', duration_str)
    if match:
        seconds = float(match.group(1))

    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds

# # Apply the conversion to the 'perc_cpu' column
# combined_metrics['perc_cpu'] = combined_metrics['perc_cpu'].apply(convert_cpu_to_float)


def prep_dfs(trace_single_machine, trace_multi_cpu_cluster, trace_local):
    metrics_single_machine = trace_single_machine[['process', 'perc_cpu', 'peak_rss', 'realtime', 'duration']].copy()
    metrics_multi_cpu_cluster = trace_multi_cpu_cluster[['process', 'perc_cpu', 'peak_rss', 'realtime', 'duration']].copy()
    metrics_local = trace_local[['process', 'perc_cpu', 'peak_rss', 'realtime', 'duration']].copy()

    # # Convert memory usage to MB
    # metrics_single_machine['peak_rss'] = metrics_single_machine['peak_rss'].apply(convert_memory_to_mb)
    # metrics_multi_cpu_cluster['peak_rss'] = metrics_multi_cpu_cluster['peak_rss'].apply(lambda x: x / (1024 * 1024))

    # metrics_single_machine['realtime'] = metrics_single_machine['realtime'].apply(convert_duration_to_seconds)
    # metrics_multi_cpu_cluster['realtime'] = metrics_multi_cpu_cluster['realtime'].apply(lambda x: x / 1000)

    # Combine the three datasets for comparison
    metrics_single_machine['run'] = 'Single-CPU machine (1 node with 16 cores, 32 threads)'
    metrics_multi_cpu_cluster['run'] = 'Multi-CPU cluster (11 nodes with 16 cores, 32 threads)'
    metrics_local['run'] = 'Single-CPU machine (1 node with 8 cores, 16 threads)'
    combined_metrics = pd.concat([metrics_single_machine, metrics_multi_cpu_cluster, metrics_local])
    return combined_metrics

def do_the_plotting(combined_metrics, metric):
    # Create the violin plot
    plt.figure(figsize=(4, 5))
    # sns.violinplot(x='process', y='realtime', hue='run', data=combined_metrics, split=True, inner='quartile')
    # sns.boxplot(y='process', x=metric, hue='run', data=combined_metrics, palette=['#2CB58E', '#DAA520', '#D64045'], width=0.75, whis=[0, 100], linewidth=1.5, zorder=1)
    sns.stripplot(y='process', x=metric, data=combined_metrics, size=3.5, hue='run', color='black', alpha=0.5, dodge=True, zorder=2)
    # sns.despine(trim=True, left=True)
    if metric == 'realtime':
        plt.xscale("log")
        plt.xlabel('Job duration [s]')
        vlines = [1, 10, 100, 1000]
    elif metric == "perc_cpu":
        plt.xlabel('Single-core CPU usage [%]')
        vlines = [0, 200, 400, 600, 800, 1000, 1200]
    for x in vlines:
        plt.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=.5, zorder=0)
    plt.legend('')
    # plt.savefig(rf'E:\02_macs_fire_sites\00_working\06_for_documentation\04_result_plots\{metric}_three.svg')
    # plt.savefig(rf'E:\02_macs_fire_sites\00_working\06_for_documentation\04_result_plots\{metric}_three.png')
    plt.show()

def do_plotting_new(combined_metrics, metric):
    # Create a figure with two subplots: the boxplot on the left and the barplot on the right
    fig, axes = plt.subplots(ncols=2, figsize=(6.5, 5))  # Total width 6.5 inches: 4 + 2.5 inches

    # Boxplot (left subplot)
    plt.sca(axes[0])  # Set the current axis to the first subplot
    sns.boxplot(y='process', x=metric, hue='run', data=combined_metrics, palette=['#2CB58E', '#DAA520', '#D64045'], width=0.75, whis=[0, 100], linewidth=1.5, zorder=1)
    sns.stripplot(y='process', x=metric, data=combined_metrics, size=2.5, hue='run', color='black', alpha=0.15, dodge=True, zorder=2)
    sns.despine(trim=True, left=True)

    if metric == 'realtime':
        plt.xscale("log")
        plt.xlabel('Job duration [s]')
        vlines = [1, 10, 100, 1000]
    elif metric == "perc_cpu":
        plt.xlabel('Single-core CPU usage [%]')
        vlines = [0, 200, 400, 600, 800, 1000, 1200]
        
    for x in vlines:
        plt.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=.5, zorder=0)

    plt.legend('')
    axes[0].set_title('Boxplot')  # Title for the left plot

    # Barplot (right subplot)
    plt.sca(axes[1])  # Set the current axis to the second subplot
    data = {
        'Index': [0, 1, 2, 3, 4, 5],
        'Task': [
            'demToGraph', 'extractTroughTransects', 'transectAnalysis',
            'networkAnalysis', 'graphToShapefile', 'mergeAnalysisCSVs'
        ],
        'Cluster': [49.0, 61.0, 1666.0, 1194.0, 898.0, 8.0],
        'CPU15': [203.619, 2086.462, 13666.147, 13661.568, 13659.446, 1.007],
        'Efficiency': ['x4.16', 'x34.20', 'x8.20', 'x11.44', 'x15.21', 'x0.13']
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars='Task', value_vars=['CPU15', 'Cluster'], var_name='Environment', value_name='Runtime')

    sns.barplot(y='Runtime', x='Task', hue='Environment', palette=['#2CB58E', '#DAA520'], data=df_melted)
    sns.despine(left=True, bottom=True)
    plt.title('Runtime Comparison by Task and Environment')
    plt.xlabel('Task')
    plt.ylabel('Runtime (sec)')
    plt.legend(title='Environment')
    plt.tight_layout()

    # Display the combined plot
    plt.show()

# Example usage (assuming combined_metrics is already defined and populated with data)
# do_the_plotting(combined_metrics, 'realtime')

# combined_metrics.to_csv(fr'E:\02_macs_fire_sites\00_working\01_processed-data\06_workflow_outputs\run_20240527\all_metrics_to_plot.csv')
# print(combined_metrics)
# do_the_plotting('realtime')

def makespan_barplot():
    # Data to be formatted into a DataFrame
    data = {
        'Index': [0, 1, 2, 3, 4, 5],
        'Task': [
            'demToGraph', 'extractTroughTransects', 'transectAnalysis',
            'networkAnalysis', 'graphToShapefile', 'mergeAnalysisCSVs'
        ],
        'Cluster': [49.0, 61.0, 1666.0, 1194.0, 898.0, 8.0],
        'CPU15': [203.619, 2086.462, 13666.147, 13661.568, 13659.446, 1.007],
        'Efficiency': ['x4.16', 'x34.20', 'x8.20', 'x11.44', 'x15.21', 'x0.13']
    }
    # Creating the DataFrame
    df = pd.DataFrame(data)
    # Display the DataFrame
    print(df)

    df['Cluster'] = df['Cluster'] / 60
    df['CPU15'] = df['CPU15'] / 60

    # Melt the DataFrame to have a long-form data structure suitable for seaborn
    df_melted = df.melt(id_vars='Task', value_vars=['CPU15', 'Cluster'], var_name='Environment', value_name='Runtime')


    # Create the seaborn bar plot
    plt.figure(figsize=(2.5, 4))
    ax = sns.barplot(y='Runtime', x='Task', hue='Environment', palette=['#2CB58E', '#DAA520'], data=df_melted, width=0.9)
    
    # Add black outline to each bar
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.9)
    hlines = [0, 50, 100, 150, 200]
    for x in hlines:
        plt.axhline(y=x, color='gray', linestyle='-', linewidth=0.5, alpha=.5, zorder=0)
    # sns.despine(left=True, bottom=True)
    # plt.title('Runtime Comparison by Task and Environment')
    # plt.xlabel('Task')
    plt.ylabel('Task runtime [min]')
    # plt.yscale("log")
    plt.xticks(rotation=45)
    plt.legend('')
    # plt.tight_layout()

    plt.savefig(rf'E:\02_macs_fire_sites\00_working\06_for_documentation\04_result_plots\task_runtimes_cluster_cpu15.svg')
    plt.savefig(rf'E:\02_macs_fire_sites\00_working\06_for_documentation\04_result_plots\task_runtimes_cluster_cpu15.png')
    # Display the plot
    plt.show()



# added after Sep 2025:
def subset_trace_by_good_aois(
    good_aois_path="/Users/stare/Documents/99_personal_things/04_remaining_AWI_things/01_fires/aois_and_fires_good.csv",
    trace_full_path="/Users/stare/Documents/99_personal_things/04_remaining_AWI_things/01_fires/nextflow_reports/trace_local.csv"
):
    # Read the good AOIs table
    good_aois = pd.read_csv(good_aois_path)
    # Extract the integer AOI numbers from the 'name' column
    good_aois_list = []
    for name in good_aois['name']:
        match = re.search(r"arf_aoi_(\d{3})", name)
        if match:
            good_aois_list.append(int(match.group(1)))

    # Read the trace_full table
    trace_full = pd.read_csv(trace_full_path)
    # Function to extract AOI number from the end of the 'name' column (in brackets)
    def extract_aoi_from_name(name):
        match = re.search(r"\((\d+)\)$", str(name))
        if match:
            return int(match.group(1))
        return None

    trace_full['aoi_num'] = trace_full['name'].apply(extract_aoi_from_name)
    # Subset rows where aoi_num is in good_aois_list
    trace_full_good = trace_full[trace_full['aoi_num'].isin(good_aois_list)].copy()
    # Optionally drop the helper column
    trace_full_good = trace_full_good.drop(columns=['aoi_num'])
    # Save to same directory as trace_full_path
    out_path = os.path.join(os.path.dirname(trace_full_path), "trace_local_good.csv")
    mask_good_aoi = trace_full['aoi_num'].isin(good_aois_list)
    mask_merge = trace_full['name'].str.contains(r"mergeAnalysisCSVs \(1\)")
    trace_full_good = trace_full[mask_good_aoi | mask_merge].copy()
    trace_full_good.to_csv(out_path, index=False)
    print(f"Subsetted trace_full_good saved to: {out_path}")
    
    trace_full = pd.read_csv(trace_full_path)
    count_aoi_occurrences(trace_full)

    return good_aois_list, trace_full_good

def count_aoi_occurrences(trace_full):
    # Extract AOI number from the end of the 'name' column (in brackets)
    def extract_aoi_from_name(name):
        match = re.search(r"\((\d{3})\)$", str(name))
        if match:
            return int(match.group(1))
        return None

    trace_full['aoi_num'] = trace_full['name'].apply(extract_aoi_from_name)
    counts = trace_full['aoi_num'].value_counts().sort_index()
    print("AOI number occurrence counts:")
    print(counts)
    return counts


def make_ridge_plot(combined_metrics):
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=combined_metrics, 
        x="perc_cpu", 
        y="process", 
        scale="width", 
        inner=None, 
        cut=0, 
        bw=0.2
    )
    sns.despine(left=True, bottom=True)
    plt.xlabel('Single-core CPU usage [%]')
    plt.ylabel('Process')
    plt.title('CPU Usage Distribution by Task (Ridge/Violin Style)')
    plt.show()


def make_joy_plot(combined_metrics):
    import joypy
    plt.figure(figsize=(6, 5))
    fig, axes = joypy.joyplot(
        combined_metrics, 
        by="process", 
        column="perc_cpu", 
        ylim='own', 
        figsize=(6, 5), 
        kind="kde", 
        bins=20, 
        overlap=0.5, 
        colormap=plt.cm.viridis
    )
    plt.xlabel('Single-core CPU usage [%]')
    plt.title('CPU Usage Distribution by Task (Joy Plot)')
    plt.show()


def makespan_barplot_from_combined(combined_metrics):
    """
    Plots a barplot of total runtime per task for each environment using combined_metrics.
    Each bar shows the sum of 'realtime' per task, divided by 60000 (to get minutes).
    """
    # Desired order
    task_order = [
        'demToGraph', 'extractTroughTransects', 'transectAnalysis',
        'networkAnalysis', 'graphToShapefile', 'mergeAnalysisCSVs'
    ]

    # Clean process names (remove AOI number in brackets)
    combined_metrics['Task'] = combined_metrics['process'].apply(lambda x: re.sub(r'\s*\(\d+\)', '', str(x)))

    # Group and sum runtimes per task and environment, convert to minutes
    runtime = (
        combined_metrics
        .groupby(['Task', 'run'])['duration']
        .sum()
        .unstack('run')
        / 60000
    )

    # Reset index for plotting
    df_melted = runtime.reset_index().melt(id_vars='Task', var_name='Environment', value_name='Runtime')

    # Set Task as categorical with the desired order
    df_melted['Task'] = pd.Categorical(df_melted['Task'], categories=task_order, ordered=True)

    # Plot
    plt.figure(figsize=(3.5, 4.5))
    ax = sns.barplot(y='Runtime', x='Task', hue='Environment',
                     palette=['#DAA520', '#2CB58E', '#D64045'], data=df_melted, width=0.9, order=task_order)

    # Add black outline to each bar
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.9)
    hlines = [0, 50, 100, 150, 200]
    for x in hlines:
        plt.axhline(y=x, color='gray', linestyle='-', linewidth=0.5, alpha=.5, zorder=0)
    plt.ylabel('Task runtime [min]')
    plt.xticks(rotation=45)
    plt.legend(title='Environment')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load the trace files
    trace_single_machine = pd.read_csv(fr'/Users/stare/Documents/99_personal_things/04_remaining_AWI_things/01_fires/nextflow_reports/trace_cpu15_good.csv', sep=',')
    trace_multi_cpu_cluster = pd.read_csv(fr'/Users/stare/Documents/99_personal_things/04_remaining_AWI_things/01_fires/nextflow_reports/trace_full_good.csv', sep=',')
    trace_local = pd.read_csv(fr'/Users/stare/Documents/99_personal_things/04_remaining_AWI_things/01_fires/nextflow_reports/trace_local_good.csv', sep=',')
    combined_metrics = prep_dfs(trace_single_machine, trace_multi_cpu_cluster, trace_local) 
    # Only keep multi-CPU cluster rows
    # combined_metrics = combined_metrics[combined_metrics['run'] == 'Multi-CPU cluster (11 nodes with 16 cores, 32 threads)']
    # do_the_plotting(combined_metrics, "perc_cpu")
    # make_ridge_plot(combined_metrics)
    # make_joy_plot(combined_metrics)
    # makespan_barplot()

    # good_aois_list, trace_full_good = subset_trace_by_good_aois()
    # print("number of good aois:", len(good_aois_list))
    # print(good_aois_list)
    # do_plotting_new(combined_metrics, 'perc_cpu')
    makespan_barplot_from_combined(combined_metrics)










# Old stuff

# # Extract relevant columns and clean 'name' (removes the number in brackets at the end)
# trace_single_machine['process'] = trace_single_machine['name'].apply(lambda x: re.sub(r'\s*\(\d+\)', '', x))

# metrics_single_machine = trace_single_machine[['process', 'perc_cpu', 'peak_rss', 'realtime']].copy()
# metrics_multi_cpu_cluster = trace_multi_cpu_cluster[['process', 'perc_cpu', 'peak_rss', 'realtime']].copy()

# # Convert memory usage to MB
# metrics_single_machine['peak_rss'] = metrics_single_machine['peak_rss'].apply(convert_memory_to_mb)
# metrics_multi_cpu_cluster['peak_rss'] = metrics_multi_cpu_cluster['peak_rss'].apply(lambda x: x / (1024 * 1024))

# # Function to convert realtime to seconds
# def convert_duration_to_seconds(duration_str):
#     # Check for milliseconds format like '200ms'
#     if re.match(r'^\d+(\.\d+)?ms$', duration_str):
#         return float(duration_str.replace('ms', '')) / 1000

#     # Check for seconds format like '2.2s'
#     if re.match(r'^\d+(\.\d+)?s$', duration_str):
#         return float(duration_str.replace('s', ''))

#     # Check for formats like '1h 16m 27s', '1h 9m 23s', '13m 9s'
#     hours = minutes = seconds = 0
#     match = re.search(r'(\d+)h', duration_str)
#     if match:
#         hours = int(match.group(1))
#     match = re.search(r'(\d+)m', duration_str)
#     if match:
#         minutes = int(match.group(1))
#     match = re.search(r'(\d+(\.\d+)?)s', duration_str)
#     if match:
#         seconds = float(match.group(1))

#     total_seconds = hours * 3600 + minutes * 60 + seconds
#     return total_seconds

# metrics_single_machine['realtime'] = metrics_single_machine['realtime'].apply(convert_duration_to_seconds)
# metrics_multi_cpu_cluster['realtime'] = metrics_multi_cpu_cluster['realtime'].apply(lambda x: x / 1000)

# # Combine the two datasets for comparison
# metrics_single_machine['run'] = 'Single Machine'
# metrics_multi_cpu_cluster['run'] = 'Multi-CPU Cluster'
# combined_metrics = pd.concat([metrics_single_machine, metrics_multi_cpu_cluster])
