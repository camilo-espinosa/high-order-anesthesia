import torch
from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from collections import defaultdict
import h5py
import os
import itertools
from collections import Counter
import logging
import nibabel as nib
from nilearn import surface
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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
    gc_path: str = "data/covariance_matrices_gc.h5",
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


_super = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci_label(x):
    if x == 0:
        return "0"
    return f"{x:.1f}"


cocomac_table = {
    1: ("TCpol", "R"),
    2: ("TCs", "R"),
    3: ("Amyg", "R"),
    4: ("PFCoi", "R"),
    5: ("Ia", "R"),
    6: ("PFCom", "R"),
    7: ("TCc", "R"),
    8: ("PFCol", "R"),
    9: ("TCi", "R"),
    10: ("PHC", "R"),
    11: ("G", "R"),
    12: ("PMCvl", "R"),
    13: ("VACv", "R"),
    14: ("Ip", "R"),
    15: ("PFCpol", "R"),
    16: ("HC", "R"),
    17: ("CCs", "R"),
    18: ("PFCvl", "R"),
    19: ("V2", "R"),
    20: ("PFCm", "R"),
    21: ("TCv", "R"),
    22: ("VACd", "R"),
    23: ("V1", "R"),
    24: ("PFCcl", "R"),
    25: ("A2", "R"),
    26: ("CCr", "R"),
    27: ("CCp", "R"),
    28: ("CCa", "R"),
    29: ("S2", "R"),
    30: ("S1", "R"),
    31: ("A1", "R"),
    32: ("M1", "R"),
    33: ("PCi", "R"),
    34: ("PCm", "R"),
    35: ("PFCdm", "R"),
    36: ("PCip", "R"),
    37: ("PCs", "R"),
    38: ("FEF", "R"),
    39: ("PFCdl", "R"),
    40: ("PMCm", "R"),
    41: ("PMCdl", "R"),
    42: ("TCpol", "L"),
    43: ("TCs", "L"),
    44: ("Amyg", "L"),
    45: ("PFCoi", "L"),
    46: ("Ia", "L"),
    47: ("PFCom", "L"),
    48: ("TCc", "L"),
    49: ("PFCol", "L"),
    50: ("TCi", "L"),
    51: ("PHC", "L"),
    52: ("G", "L"),
    53: ("PMCvl", "L"),
    54: ("VACv", "L"),
    55: ("Ip", "L"),
    56: ("PFCpol", "L"),
    57: ("HC", "L"),
    58: ("CCs", "L"),
    59: ("PFCvl", "L"),
    60: ("V2", "L"),
    61: ("PFCm", "L"),
    62: ("TCv", "L"),
    63: ("VACd", "L"),
    64: ("V1", "L"),
    65: ("PFCcl", "L"),
    66: ("A2", "L"),
    67: ("CCr", "L"),
    68: ("CCp", "L"),
    69: ("CCa", "L"),
    70: ("S2", "L"),
    71: ("S1", "L"),
    72: ("A1", "L"),
    73: ("M1", "L"),
    74: ("PCi", "L"),
    75: ("PCm", "L"),
    76: ("PFCdm", "L"),
    77: ("PCip", "L"),
    78: ("PCs", "L"),
    79: ("FEF", "L"),
    80: ("PFCdl", "L"),
    81: ("PMCm", "L"),
    82: ("PMCdl", "L"),
}


def plot_cocomac_regions(
    regions,
    colors,
    names=None,
    cocomac_names=False,
    file_path="data/visualization/",
    show=True,
    distance=1,
):
    """
    Plot selected CoCoMac regions on the macaque cortical surface using per-vertex RGB colors.

    Parameters
    ----------
    regions : list[int] (0-based)
    colors  : list[str] (hex colors, one per region)
    names   : list[str] or None
        Legend labels. Empty list + cocomac_names=True → use CoCoMac abbreviations.
        Empty list + cocomac_names=False → no legend.
    cocomac_names : bool
    """
    if names is None:
        names = []
    if len(names) not in (0, len(regions)):
        raise ValueError("names must be empty or the same length as regions")

    regions_1b = (np.array(regions) + 1).tolist()

    if len(names) == 0:
        final_names = (
            [f"{cocomac_table[rid][0]} ({cocomac_table[rid][1]})" for rid in regions_1b]
            if cocomac_names
            else None
        )
    else:
        final_names = list(names)

    surf_file = file_path + "f99_surface_147k.gii"
    label_file = file_path + "f99_regionMapping_147k_84.gii"

    coords, faces = surface.load_surf_mesh(surf_file)
    region_map = nib.load(label_file).darrays[0].data.astype(int)

    def _hex_to_rgb255(c):
        return (np.array(mcolors.to_rgb(c)) * 255).astype(np.uint8)

    region_color_rgb = {
        rid: _hex_to_rgb255(col) for rid, col in zip(regions_1b, colors)
    }

    n_vertices = coords.shape[0]
    vec2coco = np.concatenate([np.arange(0, 41), np.arange(42, 83)])
    bg = np.array([245, 245, 220], dtype=np.uint8)
    vertex_rgb = np.tile(bg, (region_map.shape[0], 1)).astype(np.uint8)

    for idx, coco_val in enumerate(vec2coco):
        rid = idx + 1
        if rid in region_color_rgb:
            vertex_rgb[region_map == coco_val] = region_color_rgb[rid]

    coords_L = coords[: n_vertices // 2]
    faces_L = faces[np.all(faces < n_vertices // 2, axis=1)]
    rgb_L = vertex_rgb[: n_vertices // 2]

    coords_R = coords[n_vertices // 2 :]
    faces_R = faces[np.all(faces >= n_vertices // 2, axis=1)] - n_vertices // 2
    rgb_R = vertex_rgb[n_vertices // 2 :]

    views = {
        "lateral_L": dict(eye=dict(x=-3.0 * distance, y=0, z=0)),
        "medial_L": dict(eye=dict(x=2.6 * distance, y=0, z=0)),
        "medial_R": dict(eye=dict(x=-2.6 * distance, y=0, z=0)),
        "lateral_R": dict(eye=dict(x=3.0 * distance, y=0, z=0)),
        "top": dict(eye=dict(x=0, y=0, z=2.2 * distance), up=dict(x=0, y=1, z=0)),
        "bottom": dict(eye=dict(x=0, y=0, z=-2.2 * distance), up=dict(x=0, y=1, z=0)),
    }

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "surface"}] * 3, [{"type": "surface"}] * 3],
        horizontal_spacing=0.02,
        vertical_spacing=0.01,
    )

    def _add(row, col, crd, fcs, rgb, cam):
        fig.add_trace(
            go.Mesh3d(
                x=crd[:, 0],
                y=crd[:, 1],
                z=crd[:, 2],
                i=fcs[:, 0],
                j=fcs[:, 1],
                k=fcs[:, 2],
                vertexcolor=[f"rgb({r},{g},{b})" for r, g, b in rgb],
                showscale=False,
                lighting=dict(ambient=0.8, diffuse=0.6, roughness=0.9),
            ),
            row=row,
            col=col,
        )
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=cam,
                aspectmode="data",
            ),
            row=row,
            col=col,
        )

    _add(1, 1, coords_R, faces_R, rgb_R, views["medial_R"])
    _add(1, 3, coords_L, faces_L, rgb_L, views["medial_L"])
    _add(2, 3, coords_L, faces_L, rgb_L, views["lateral_L"])
    _add(2, 1, coords_R, faces_R, rgb_R, views["lateral_R"])
    _add(1, 2, coords, faces, vertex_rgb, views["top"])
    _add(2, 2, coords, faces, vertex_rgb, views["bottom"])

    if final_names is not None:
        for name, col in zip(final_names, colors):
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=16, color=col),
                    name=name,
                )
            )

    fig.update_layout(height=800, width=1000)
    if show:
        fig.show()
    return fig


def plot_cocomac_region_values(
    values,
    file_path="data/visualization/",
    cmap_name="viridis",
    vmin=None,
    vmax=None,
    colorbar_title=None,
    show=True,
    fontsize=18,
    height=800,
    width=1000,
    distance=1,
    title="",
):
    """
    Plot a scalar value per CoCoMac region (82 regions) on the f99 surface
    using a continuous colormap and a colorbar.

    Parameters
    ----------
    values : array-like, shape (82,)
        One value per CoCoMac region, 0-based index:
        values[0] -> region 1, values[81] -> region 82.
    file_path : str
        Folder containing f99_surface_147k.gii and f99_regionMapping_147k_84.gii.
    cmap_name : str
        Matplotlib colormap name (e.g. 'viridis', 'magma', 'coolwarm', 'RdBu_r', ...).
    vmin, vmax : float or None
        Color scale limits. If None, use min/max of `values`.
    colorbar_title : str or None
        Title for the colorbar (e.g. 'count', 'Ω', etc.).
    show : bool
        If True, call fig.show().
    """

    values = np.asarray(values, dtype=float)
    if values.shape[0] != 82:
        raise ValueError("`values` must have length 82 (one per CoCoMac region).")

    # --- Load surface and labels ---
    surf_file = file_path + "f99_surface_147k.gii"
    label_file = file_path + "f99_regionMapping_147k_84.gii"

    coords, faces = surface.load_surf_mesh(surf_file)
    region_map = nib.load(label_file).darrays[0].data.astype(int)

    n_vertices = coords.shape[0]

    # f99 mapping: CoCoMac indices (1..82) -> region_map values (vec2coco)
    vec2coco = np.concatenate([np.arange(0, 41), np.arange(42, 83)])
    hole_mask = ~np.isin(
        region_map, vec2coco
    )  # vertices not belonging to the 82 regions (e.g., label 41)

    # Map region index (1..82) -> value from `values`
    region_values = {rid: values[rid - 1] for rid in range(1, 83)}

    # --- Build per-vertex scalar array ---
    # vertex_vals = np.zeros_like(region_map, dtype=float)
    vertex_vals = np.full(region_map.shape, np.nan, dtype=float)
    for idx, coco_val in enumerate(vec2coco):
        rid = idx + 1  # CoCoMac region ID
        v = region_values[rid]
        vertex_vals[region_map == coco_val] = v
    vertex_vals[hole_mask] = 0
    # --- Hemisphere split ---
    coords_L = coords[: n_vertices // 2]
    faces_L = faces[np.all(faces < n_vertices // 2, axis=1)]
    vals_L = vertex_vals[: n_vertices // 2]

    coords_R = coords[n_vertices // 2 :]
    faces_R = faces[np.all(faces >= n_vertices // 2, axis=1)] - n_vertices // 2
    vals_R = vertex_vals[n_vertices // 2 :]

    coords_all = coords
    faces_all = faces
    vals_all = vertex_vals

    # --- Color scale limits ---
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    # --- Build Plotly colorscale from matplotlib colormap ---
    cmap = cm.get_cmap(cmap_name, 256)
    colorscale = [[i / 255, mcolors.to_hex(cmap(i))] for i in range(256)]

    # --- Camera views (same as before) ---
    views = {
        "lateral_L": dict(eye=dict(x=-3.0 * distance, y=0, z=0)),
        "medial_L": dict(eye=dict(x=2.6 * distance, y=0, z=0)),
        "medial_R": dict(eye=dict(x=-2.6 * distance, y=0, z=0)),
        "lateral_R": dict(eye=dict(x=3.0 * distance, y=0, z=0)),
        "top": dict(eye=dict(x=0, y=0, z=2.2 * distance), up=dict(x=0, y=1, z=0)),
        "bottom": dict(eye=dict(x=0, y=0, z=-2.2 * distance), up=dict(x=0, y=1, z=0)),
    }

    # --- Figure layout (2x3: medial R, top, medial L / lateral R, bottom, lateral L) ---
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.00,
    )

    def add_brain(row, col, crd, fcs, vals, camera, showscale=False):
        tickvals = np.linspace(vmin, vmax, 5)
        ticktext = [sci_label(v) for v in tickvals]
        fig.add_trace(
            go.Mesh3d(
                x=crd[:, 0],
                y=crd[:, 1],
                z=crd[:, 2],
                i=fcs[:, 0],
                j=fcs[:, 1],
                k=fcs[:, 2],
                intensity=vals.astype(float),
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=showscale,
                colorbar=(
                    dict(
                        title=dict(
                            text=colorbar_title,
                            font=dict(size=fontsize),
                        ),
                        tickfont=dict(size=fontsize),
                        tickvals=tickvals.tolist(),
                        ticktext=ticktext,
                        thickness=15,
                        len=0.7,
                        x=1.15,
                        xanchor="center",
                        y=0.5,
                    )
                    if showscale
                    else None
                ),
                lighting=dict(ambient=0.8, diffuse=0.6, roughness=0.9),
            ),
            row=row,
            col=col,
        )
        fig.update_scenes(
            dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=camera,
                aspectmode="data",
            ),
            row=row,
            col=col,
        )

    # Add hemispheres and top/bottom; show colorbar only once (e.g., top view)
    add_brain(1, 1, coords_R, faces_R, vals_R, views["medial_R"], showscale=False)
    add_brain(1, 3, coords_L, faces_L, vals_L, views["medial_L"], showscale=False)
    add_brain(2, 3, coords_L, faces_L, vals_L, views["lateral_L"], showscale=False)
    add_brain(2, 1, coords_R, faces_R, vals_R, views["lateral_R"], showscale=False)
    add_brain(1, 2, coords_all, faces_all, vals_all, views["top"], showscale=True)
    add_brain(2, 2, coords_all, faces_all, vals_all, views["bottom"], showscale=False)

    fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=0, r=0, t=90, b=0),
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=fontsize),
        ),
    )

    if show:
        fig.show()

    return fig
