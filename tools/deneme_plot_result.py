import matplotlib.pyplot as plt
import h5py
import numpy as np

# Model list should be defined as a list of strings
MODEL = ['padim', 'cflow', 'ganomaly', 'dfkde', 'patchcore', 'cfa', 'dfm', 'efficient_ad', 'fastflow']

def result(model):
    f1 = h5py.File(f"{model}_result_datas.h5", 'r+')

    auc_result = f1["auc_result"]
    auc = auc_result[()]
    print(f"{model} auc:", auc)

    mean_time = f1["mean_time"]
    mean_time_all = mean_time[()]
    print(f"{model} mean_time:", mean_time_all)

    elapsed_time = f1["elapsed_time"]
    elapsed_time_all = elapsed_time[()]
    print(f"{model} elapsed_time:", elapsed_time_all)

    accuracy_result = f1["accuracy_result"]
    accuracy = accuracy_result[()]
    print(f"{model} accuracy:", accuracy)

    f1.close()

    return auc, mean_time_all, accuracy, elapsed_time_all

def plot_boxplots(elapsed_time_all, model):
    data = [elapsed_time_all]
    fig, ax = plt.subplots()
    boxplot = ax.boxplot(data)
    ax.set_title(f'{model} Anomalib Models Precition Time Graph', color='red', fontweight='bold', fontsize=15)
    ax.set_xticks([1])
    ax.set_xticklabels([model])

auc_val = []
time_val = []
accuracy_val = []
elapsed_time_val = []

for model in MODEL:
    print(model)
    auc, mean_time_all, accuracy, elapsed_time_all = result(model)
    # plot_boxplots(elapsed_time_all, model)
    auc_val.append(auc)
    time_val.append(mean_time_all)
    accuracy_val.append(accuracy)
    elapsed_time_val.append(elapsed_time_all)
marker_list = ["o", "s", "D", "v", "^", ">", "p", "H", "X"]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
ax1.set_title("Anomalib (Boyteks Dataset 400)", color='red')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Predict Time(ms)')
for i, model in enumerate(MODEL):
    ax1.scatter(time_val[i], accuracy_val[i], label=model, marker=marker_list[i])

ax1.legend()

# İkinci subplot
ax2.set_title("Anomalib (Boyteks Dataset 400)", color='red')
ax2.set_ylabel('Area Under the Curve')
ax2.set_xlabel('Predict Time(ms)')
for i, model in enumerate(MODEL):
    ax2.scatter(time_val[i], auc_val[i], label=model, marker=marker_list[i])

ax2.legend()

plt.tight_layout()
plt.show()
