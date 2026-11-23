# LegSeg

LegSeg is a desktop GUI for segmenting Dixon MRI images of the leg with
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). It loads Fat and Water DICOM series,
runs a trained nnU-Net model, shows the segmentation over the images, and reports
the share of each tissue per slice.

## Installation

Requires Python 3.14. The install scripts create a `.venv` virtual environment,
install PyTorch (CUDA 12.8 build by default), then install nnU-Net, pydicom and PyQt6,
and finally download the trained model (~790 MB) into `nnUNet_results`.

Linux:

```bash
./install.sh            # CUDA 12.8
CUDA=cpu ./install.sh   # CPU only
```

Windows:

```bat
install.bat             :: CUDA 12.8
install.bat cpu         :: CPU only
```

For a different CUDA version, pass its PyTorch wheel tag, e.g. `cu126`.

## Usage

```bash
.venv/bin/python legseg.py          # Linux
.venv\Scripts\python.exe legseg.py  # Windows
```

1. **File > Load Patients...** – select one or more patient folders. The Fat and Water
   series subfolders are detected automatically and can be corrected in a dialog.
2. Set the slice range and click **Run** to segment the selected patients.
3. Use the sliders to change overlay opacity and to switch between slices and patients.
4. **Save** exports colored PNG overlays and the raw segmentation files.

The computation device (CPU/GPU) and the model are selected in the **Run** menu.

## Model

The default model `nnUNet_results/Dataset015_SARXfinal/nnUNetTrainer_100epochs__nnUNetPlans__2d`
is a 2D 5-fold ensemble trained on Fat/Water Dixon slices. It segments these labels:

| Label | Name                         |
|-------|------------------------------|
| 1     | visceral fat                 |
| 2     | muscle                       |
| 3     | bone                         |
| 4     | nerves and vessels           |
| 5     | intra- and intermuscular fat |

The model is published as a
[GitHub Release asset](https://github.com/isi-nmr/LegSeg/releases/tag/model-v1)
(`Dataset015_SARXfinal.zip`, full checkpoints including optimizer state) and is downloaded
by the install scripts. To install it manually, unzip it into `nnUNet_results/`.

It is loaded at startup. Other trained nnU-Net models can be copied into `nnUNet_results`
using the standard `<Dataset>/<Trainer>__<Plans>__<Configuration>/fold_X/checkpoint_final.pth`
layout and selected through **Run > Select Model...**.

## License

LegSeg is licensed under the [GNU General Public License v3.0](LICENSE), as required by
its PyQt6 dependency.
