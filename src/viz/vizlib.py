import os
import numpy as np
from matplotlib import pyplot as plt


def basic_metrics_per_epoch_unspec(metrics_list, labels, epochs, save_path, savename: str = ""):
    plt.figure(figsize=(10, 6))
    for metric_values, label in zip(metrics_list, labels):
        plt.plot(range(1, epochs + 1), metric_values, marker='o', label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{savename}_metrics.png'))
    plt.close()


def plot_calibration_analysis(bin_confidences, bin_accuracies, bin_counts, cw_bins, cw_accuracies, cw_counts,
                               ece_list_cw, mce_list_cw, class_names, save_path, 
                               confidences=None, savename='', classwise=True,):
    num_classes = len(class_names)
    basic_calcurve_binhist(bin_confidences, bin_accuracies, bin_counts, save_path, savename, num_classes=num_classes)
    if classwise:
        classwise_calibration_plots(cw_bins, cw_accuracies, cw_counts, save_path, savename, class_names,
                                     ece_list_cw, mce_list_cw)


def basic_calcurve_binhist(bin_confidences, bin_accuracies, bin_counts, save_path, savename: str = "", num_classes: int = 10):
    mask = np.array(bin_counts) > 0
    bin_confidences = np.array(bin_confidences)[mask]
    bin_accuracies = np.array(bin_accuracies)[mask]
    order = np.argsort(bin_confidences)
    bin_confidences = bin_confidences[order]
    bin_accuracies = bin_accuracies[order]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(bin_confidences, bin_accuracies, 'b-o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve on Test Set')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    bins = np.linspace(0, 1, len(bin_counts) + 1)
    plt.bar(bins[:-1], bin_counts, width=1 / len(bin_counts), align='edge', color='skyblue', edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Samples')
    plt.title('Histogram of Confidence Bins on Test Set')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{savename}_cali_metrics.png'))
    plt.close()


def classwise_calibration_plots(all_bin_confidences, all_bin_accuracies, all_bin_counts, save_path,
                                 savename: str = "", class_names=None, ece_list=None, mce_list=None):
    n_classes = len(all_bin_confidences)
    cols = min(3, n_classes)
    calibration_rows = 2 * ((n_classes + cols - 1) // cols)
    add_extra = 1 if ece_list is not None and mce_list is not None else 0
    total_rows = calibration_rows + add_extra
    plt.figure(figsize=(6 * cols, 5 * total_rows))

    for i in range(n_classes):
        row = (i // cols) * 2
        col = i % cols
        subplot_idx = row * cols + col + 1

        plt.subplot(total_rows, cols, subplot_idx)
        plt.plot(all_bin_confidences[i], all_bin_accuracies[i], 'b-o', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Frequency')
        class_name = class_names[i] if class_names else f'Class {i}'
        plt.title(f'Calibration - {class_name}')
        plt.grid(True)
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        subplot_idx = (row + 1) * cols + col + 1
        plt.subplot(total_rows, cols, subplot_idx)
        bins = np.linspace(0, 1, len(all_bin_counts[i]) + 1)
        plt.bar(bins[:-1], all_bin_counts[i], width=1 / len(all_bin_counts[i]), align='edge', color='skyblue', edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Number of Samples')
        plt.title(f'Distribution - {class_name}')
        plt.grid(True)

    if ece_list is not None and mce_list is not None:
        plt.subplot(total_rows, 1, total_rows)
        indices = np.arange(n_classes)
        bar_width = 0.4
        plt.bar(indices, ece_list, width=bar_width, label='ECE', color='skyblue')
        plt.bar(indices + bar_width, mce_list, width=bar_width, label='MCE', color='salmon')
        if class_names is not None:
            plt.xticks(indices + bar_width / 2, class_names, rotation=45, ha='right')
        else:
            plt.xticks(indices + bar_width / 2, [f'Class {i}' for i in range(n_classes)])
        plt.ylabel("Calibration Error")
        plt.title("ECE and MCE per Class", fontweight='bold')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(save_path, f'{savename}_classwise_calibration.png'), bbox_inches='tight', dpi=300)
    plt.close()
