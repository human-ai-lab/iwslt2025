import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

''' 
Example usage: 
    python {PATH}/plot_stream_scores.py -r {PATH}/exp_root_dir -l StreamLAAL_CA -o tradeoff
'''

parser = argparse.ArgumentParser(description="trade-off curve")
parser.add_argument(
    "--root-dir",
    "-r",
    type=str,
    default="",
    help="Plot all exps under this root directory if exists"
)
parser.add_argument(
    "--exp-dirs",
    "-e",
    type=str,
    nargs='+',
    help="Path to the exp root directory"
)
parser.add_argument(
    "--out-name",
    "-o",
    type=str,
)
parser.add_argument(
    "--latency-metric",
    "-l",
    type=str,
    default="StreamLAAL",
    help="StreamLAAL, StreamLAAL_CA"
)
parser.add_argument(
    "--n-col",
    "-n",
    type=int,
    help="Number of columns in the legend",
    default=3
)
parser.add_argument(
    "--quality-metric",
    "-q",
    type=str,
    default="BLEU",
    help="BLEU, xcomet"
)
parser.add_argument(
    "--markers",
    type=str,
    nargs='+',
    default=["o", "s", "p", "h", "^", "d", "+", "*", "x", "v"],
    help="Data points markers"
)

args = parser.parse_args()

# Read TSV file
def read_tsv(file_path):
    return pd.read_csv(file_path, delimiter='\t')

# Function to plot BLEU-LAAL curve
def plot_curve(plot_files, data_marker, line_style, col, lab):
    quality_metric = []
    latency = []
    # sort the plot_files according to the latency metric
    plot_files = sorted(plot_files, key=lambda x: x[args.latency_metric][0])
    for file in plot_files:
        quality_metric.append(file[args.quality_metric][0])
        latency.append(file[args.latency_metric][0])
    # set marker size
    plt.plot(latency, quality_metric, marker=data_marker, linestyle=line_style, color=col, label=lab, markersize=12, linewidth=2.5)

# Obtain meta results data
if os.path.exists(getattr(args, "root_dir", "")):
    root_paths = [os.path.join(args.root_dir, x) for x in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, x))]
else:
    root_paths = args.exp_dirs

# Sort the experiments with a certain pattern
model_priority = {"Baseline":0, "llama": 1, "qwen": 2}
def sort_key(path):
    # extract the last part of the path like 'llama_LA'
    name = path.split("/")[-1]
    model = name.split("_")[0]
    return (model_priority.get(model, 999))
root_paths = sorted(root_paths, key=sort_key)

data = []  # Tuple[exp_name, exp_data]
for root_path in root_paths:
    # get the name of the root_path
    exp_name = os.path.basename(root_path)
    # get the data points from the sub directories of the root_path
    sub_dirs = [x[0] for x in os.walk(root_path)][1:]

    # Read the scores.tsv file from each sub directory
    exp_data = []
    for sub_dir in sub_dirs:
        if os.path.exists(os.path.join(sub_dir, "stream_scores.tsv")):
            data_point = read_tsv(os.path.join(sub_dir, "stream_scores.tsv"))
            exp_data.append(data_point)
        else:
            Warning ("scores.tsv file not found in {}".format(sub_dir))
            continue
            # raise FileNotFoundError("scores.tsv file not found in {}".format(sub_dir))
    data.append((exp_name, exp_data))

# Use stronger, more vibrant colors instead of darkened Set3
# Define a custom color palette with strong, distinct colors
strong_colors = [
    '#2196F3',  # Bright blue
    '#009688',  # Teal
    '#E74C3C',  # Intense red
    '#9C27B0',  # Deep purple
    '#00BCD4',  # Cyan
    '#4CAF50',  # Vivid green
    '#FFC107',  # Amber
    '#FF9800',  # Orange
    '#673AB7',  # Deep violet
    '#3F51B5'   # Indigo
]

# Plotting the curves for each file
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
for i, (exp_name, exp_data) in enumerate(data):
    color = strong_colors[i % len(strong_colors)]  # Cycle through colors if more experiments than colors
    plot_curve(exp_data, data_marker=args.markers[i], line_style='-', col=color, lab=exp_name)

# Adding labels and title
plt.xlabel(args.latency_metric, fontsize=24)
plt.ylabel(args.quality_metric.upper(), fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Set fixed vertical axis scale from 10 to 45
plt.ylim(10, 45)

# Adding legend
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=args.n_col, borderaxespad=0., frameon=False, fontsize=22)
plt.legend(fontsize=22, loc='lower right', bbox_to_anchor=(1, 0))
plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave some space on the right

# Note: If the legend overlaps with the plot, uncomment the following
xmin, xmax = plt.xlim()
factor=0.1 # Increase the factor to move the x-axis limits further
plt.xlim(xmin, xmax + factor * (xmax - xmin))  

# horizontal grid lines
plt.gca().yaxis.grid(True)

out_name = getattr(args, "out_name", None)
assert out_name is not None, "Please provide the output name for the plot using --out-name"
if args.latency_metric.split("_")[-1] == "CA":
    out_name = out_name + "_CA"
out_name = out_name + ".pdf"

save_path = os.path.join(os.path.dirname(root_paths[0]), out_name)
# check if the directory exists
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))
print("Saving the plot to: {}".format(save_path))
plt.savefig(save_path, format='pdf', bbox_inches='tight')