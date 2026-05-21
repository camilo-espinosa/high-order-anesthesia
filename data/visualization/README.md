# Visualization assets

The surface mesh files required for brain map rendering are **not included** in
this repository due to unclear redistribution licensing.

## Required files

Place the following files in this directory (`data/visualization/`):

| File | Description |
|------|-------------|
| `f99_surface_147k.gii` | F99 macaque cortical surface mesh (~4.7 MB) |
| `f99_regionMapping_147k_84.gii` | CoCoMac region label mapping for 147k vertices (~0.2 MB) |

## Where to obtain them

These files originate from the **F99 macaque atlas** and the **CoCoMac parcellation**,
distributed as part of [The Virtual Brain (TVB)](https://www.thevirtualbrain.org) dataset.

1. Download the TVB demo datasets from:  
   https://zenodo.org/record/10'

2. Look for `Macaque_47` or `Macaque_80` connectivity datasets which include
   the f99 surface and region mapping files.

Alternatively, they may be obtained from the
[Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench)
macaque atlases or the original Caret distribution.

> **Note**: Check the applicable license before redistributing these files.
