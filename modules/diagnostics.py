import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

# ------------------------------------------------------------------------------------------------

# MODEL EVALUATION FUNCTION.

def evaluate_model(dataset_name, y_train, y_test, y_pred, kernel_name, folder_name, best_model_K_train):

    # IMPORTING THE REPORT DICTIONARY.
    
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # BUILDING THE TABLE PRINTED IN THE TERMINAL.

    table_data = []
    
    for key, values in report_dict.items():

        if key in ["accuracy", "macro avg", "weighted avg"]: 

            continue

        table_data.append([
            f"Class {key}", 
            f"{values['precision'] * 100:.2f}%", 
            f"{values['recall'] * 100:.2f}%", 
            f"{values['f1-score'] * 100:.2f}%", 
            int(values['support'])
        ])

    table_data.append([
        "Macro Avg", 
        f"{report_dict['macro avg']['precision'] * 100:.2f}%", 
        f"{report_dict['macro avg']['recall'] * 100:.2f}%", 
        f"{report_dict['macro avg']['f1-score'] * 100:.2f}%", 
        int(report_dict['macro avg']['support'])
    ])

    headers = ["", "Precision", "Recall", "F1-Score", "Support (Actual Samples)"]
    formatted_table = tabulate(table_data, headers=headers, tablefmt="fancy_grid", stralign="center", numalign="center")

    # RESULTS (printing the formatted table).
    
    print(f"\nFINAL DIAGNOSIS ON THE TEST SET FOR \"{dataset_name.upper()}\" DATASET\n")
    print(formatted_table)

    # BUILDING THE PLOTS.

    conf_matrix = confusion_matrix(y_test, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix plot.

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds", cbar=False, ax=ax1, annot_kws={"size": 14})
    ax1.set_title(f"Confusion Matrix\n{kernel_name}", fontsize=14)
    ax1.set_xlabel("QSVM Prediction", fontsize=12)
    ax1.set_ylabel("True Class", fontsize=12)

    # Gram matrix plot (sorted by class).

    sorted_indices = np.argsort(y_train)
    sorted_gram_matrix = best_model_K_train[sorted_indices, :][:, sorted_indices]

    sns.heatmap(sorted_gram_matrix, cmap="viridis", vmin=0, vmax=1, ax=ax2, cbar=True)
    ax2.set_title(f"Gram Matrix (Sorted by Class)\n{kernel_name}", fontsize=14)
    ax2.set_xlabel("Sorted Indices", fontsize=12)
    ax2.set_ylabel("Sorted Indices", fontsize=12)

    plt.tight_layout()
    
    # RESULTS (saving and showing the plots).

    plot_path = os.path.join(folder_name, f"best_model_plots_{dataset_name}.svg")
    plt.savefig(plot_path, dpi=300)
    plt.show() 