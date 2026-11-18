#!/bin/bash
# Independent environments for the whole-cell segmentation study (the cell-rg / stardist envs are not touched).
set -x
C=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/bin/conda
E=/mnt/mib-dbia08_1/users/noa.ganeko/miniconda3/envs
$C create -y -q -n cellseg-cp python=3.11 pip
$E/cellseg-cp/bin/pip install -q "cellpose>=4.0" tifffile imagecodecs scikit-image shapely pandas matplotlib
$E/cellseg-cp/bin/python -c "import cellpose, torch; print('cellpose', cellpose.version if hasattr(cellpose,'version') else '?', 'torch', torch.__version__, 'cuda', torch.version.cuda)"
$C create -y -q -n cellseg-mesmer python=3.10 pip
$E/cellseg-mesmer/bin/pip install -q deepcell tifffile imagecodecs
$E/cellseg-mesmer/bin/python -c "import deepcell, tensorflow as tf; print('deepcell', deepcell.__version__, 'tf', tf.__version__)"
echo SETUP_DONE
