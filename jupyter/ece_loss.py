import torch
import numpy as np
from matplotlib import pyplot as plt

"""
source: https://gist.github.com/gpleiss/0b17bc4bd118b49050056cfcd5446c71
"""

def make_model_diagrams(probabilities, predictions, confidences, labels, model, n_bins=10):
    """
    outputs - a torch tensor (size n x num_classes) with the outputs from the final linear layer
    - NOT the softmaxes
    labels - a torch tensor (size n) with the labels
    """
    # softmaxes = torch.nn.functional.softmax(outputs, 1)
    # confidences, predictions = softmaxes.max(1)
    accuracies = torch.eq(predictions, labels)
    overall_accuracy = (predictions==labels).sum().item()/len(labels)

    # Reliability diagram
    bins = torch.linspace(0, 1, n_bins + 1)
    width = 1.0 / n_bins
    bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    
    bin_corrects = np.array([ torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices])
    bin_scores = np.array([ torch.mean(confidences[bin_index].float()) for bin_index in bin_indices])
    bin_corrects = np.nan_to_num(bin_corrects)
    bin_scores = np.nan_to_num(bin_scores)
    
    print(bin_corrects)
    print(type(bin_corrects))

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    confs = plt.bar(bin_centers, bin_corrects, color=[0, 0, 1], width=width, ec='black')
    # bin_corrects = np.nan_to_num(np.array([bin_correct.cpu().numpy()  for bin_correct in bin_corrects]))
    # gaps = plt.bar(bin_centers, gap, bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.legend([confs], ['Accuracy'], loc='upper left', fontsize='x-large')

    # Clean up
    bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
    # ax.text(0.17, 0.82, "ECE: {:.4f}".format(ece), ha="center", va="center", size=20, weight = 'normal', bbox=bbox_props)

    ax.set_title("Reliability Diagram - {}".format(model, size=20))
    ax.set_ylabel("Accuracy",  size=18)
    ax.set_xlabel("Confidence",  size=18)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    # plt.savefig('reliability_diagram.png')
    plt.show()
