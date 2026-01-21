import torch
from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import h5py
import os
import itertools
from collections import Counter
import logging
from src.hoi_anesthesia.io import load_covariance_dict

device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_measures_accross_states(
    nplet: list,
    selected_measure: str = "O",
    title: str = "",
    custom_order: list = [
        "MA_awake",
        "ts_selv2",
        "ts_selv4",
        "moderate_propofol",
        "deep_propofol",
        "ketamine",
        "DBS_awake",
        "ts_on_5V",
        "ts_on_3V",
        "ts_on_5V_control",
        "ts_on_3V_control",
        "ts_off",
    ],
    font_size: float = 14,
    gc_path: str = "C:/Camilo/Brain_Entropy/RESULTS/covariance_matrices_gc.h5",
    num: int = 0,
):

    # Example usage
    all_covs = load_covariance_dict(gc_path)

    O_list = []

    # con state_dict tenemos las npletas y su puntaje
    # Después hay que calcular las métricas para cada sujeto y estado:
    for dataset_name, dataset in all_covs.items():
        for state_name, state in dataset.items():
            for subject in range(state.shape[0]):
                measures_ = nplets_measures(
                    state[subject], [nplet], covmat_precomputed=True, device=device
                )
                measures = measures_[0, 0].cpu().numpy()

                O_list.append(
                    [
                        dataset_name,
                        state_name,
                        subject,
                        measures[0],
                        measures[1],
                        measures[2],
                        measures[3],
                        nplet,
                    ]
                )

    cols = ["Dataset", "State", "Subject", "TC", "DTC", "O", "S", "nplet"]
    O_df = pd.DataFrame(O_list, columns=cols)

    plt.rcParams.update({"font.size": font_size})

    nplet_sample = O_df
    nplet = nplet_sample.iloc[0]["nplet"]
    # Initialize plot
    plt.figure(figsize=(14, 6), num=num)
    sns.set_style("whitegrid")
    ax = sns.violinplot(
        data=nplet_sample,
        x="State",
        y=selected_measure,
        hue="State",  # 👈 add this
        legend=False,
        inner="box",
        palette="tab20",
        dodge=False,
        alpha=0.7,
        cut=0,
        order=custom_order,
    )
    strip = sns.stripplot(
        data=nplet_sample,
        x="State",
        y=selected_measure,
        hue="State",  # 👈 add this
        legend=False,
        dodge=False,
        jitter=True,
        alpha=1,
        size=3,
        palette="tab20",
        order=custom_order,
        ax=ax,
    )

    # Annotate number of samples on top of each violin
    y_max = nplet_sample[selected_measure].max()  # Adjust spacing for bars
    y_min = nplet_sample[selected_measure].min()  # Adjust spacing for bars
    bar_height = (y_max - y_min) * 0.1  # Vertical space between bars
    # Annotate sample sizes
    for i, state in enumerate(custom_order):
        n_samples = len(nplet_sample[nplet_sample["State"] == state])
        plt.text(
            i,
            y_min - bar_height * 0.8,
            rf"$n={n_samples}$",
            ha="center",
            fontweight="bold",
        )

    # Beautify plot
    plt.suptitle(title)
    plt.xlabel("")
    plt.legend(fontsize=10)
    plt.ylabel(selected_measure)
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.ylim(y_min - bar_height, y_max * 1.2)
    plt.tight_layout()
    # plt.savefig(f"{folder_path}/{selected_measure}_{nplet_name}.png")
    # plt.close("all")
    # Show plot
    # plt.show()
