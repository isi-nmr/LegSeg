"""Discovery of trained nnU-Net models in nnUNet_results and their labels/colors.

Each trained model carries its own dataset.json (labels, file format) and plans.json,
so any model in the standard nnU-Net layout can be used:
    nnUNet_results/<DatasetID_Name>/<Trainer>__<Plans>__<Configuration>/
"""

import glob
import os
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap

from batchgenerators.utilities.file_and_folder_operations import load_json

# Model bundled in nnUNet_results, selected at startup.
DEFAULT_DATASET_NAME = "Dataset015_SARXfinal"
DEFAULT_TRAINER = "nnUNetTrainer_100epochs"
DEFAULT_PLANS = "nnUNetPlans"
DEFAULT_CONFIGURATION = "2d"

# Fixed colors for the "Default" palette, keyed by label name (as in dataset.json).
# Labels not listed here fall back to tab20.
DEFAULT_LABEL_COLORS = {
    "muscle": [230, 25, 75],
    "visceral fat": [255, 255, 0],
    "bone": [255, 255, 255],
    "nerves and vessels": [0, 130, 200],
    "intra- and intermuscular fat": [144, 238, 144],
}

_PALETTE_TO_CMAP = {
    "Default": "tab20",
    "Contrast": "Set1",
    "Viridis": "viridis",
    "Paired": "Paired",
    "Dark2": "Dark2",
}

_QUALITATIVE_CMAPS = {"tab10", "tab20", "Set1", "tab20b", "Paired", "Dark2"}


@dataclass(frozen=True)
class ModelInfo:
    path: str
    dataset_name: str
    trainer: str
    plans: str
    configuration: str
    folds: tuple
    dataset_json: dict
    plans_json: dict

    @property
    def file_ending(self):
        return self.dataset_json["file_ending"]

    @property
    def labels(self):
        return self.dataset_json["labels"]

    @property
    def is_3d(self):
        return not self.configuration.startswith("2d")

    @property
    def display_name(self):
        return f"{self.dataset_name}  |  {self.configuration}  |  {self.trainer}  (folds: {len(self.folds)})"


def _parse_leaf(leaf_dir):
    """Build a ModelInfo from a candidate trained-model folder, or None if it isn't one."""
    dataset_json_path = os.path.join(leaf_dir, "dataset.json")
    plans_json_path = os.path.join(leaf_dir, "plans.json")
    if not (os.path.isfile(dataset_json_path) and os.path.isfile(plans_json_path)):
        return None

    leaf_name = os.path.basename(leaf_dir.rstrip(os.sep))
    parts = leaf_name.split("__")
    if len(parts) != 3:
        return None
    trainer, plans, configuration = parts

    folds = []
    for fold_dir in sorted(glob.glob(os.path.join(leaf_dir, "fold_*"))):
        if not os.path.isfile(os.path.join(fold_dir, "checkpoint_final.pth")):
            continue
        try:
            folds.append(int(os.path.basename(fold_dir).split("_")[1]))
        except (IndexError, ValueError):
            continue
    if not folds:
        return None

    try:
        dataset_json = load_json(dataset_json_path)
        plans_json = load_json(plans_json_path)
    except (OSError, ValueError):
        return None
    if "labels" not in dataset_json or "file_ending" not in dataset_json:
        return None

    # In the standard layout <root>/<Dataset>/<Trainer>__<Plans>__<Config>/ the dataset
    # name is the parent folder name.
    dataset_name = os.path.basename(os.path.dirname(leaf_dir.rstrip(os.sep))) or leaf_name

    return ModelInfo(
        path=leaf_dir,
        dataset_name=dataset_name,
        trainer=trainer,
        plans=plans,
        configuration=configuration,
        folds=tuple(sorted(folds)),
        dataset_json=dataset_json,
        plans_json=plans_json,
    )


def discover_models(results_dir):
    """Recursively scan results_dir for valid trained model instances."""
    models = []
    if not os.path.isdir(results_dir):
        return models
    for dirpath, _dirnames, filenames in os.walk(results_dir):
        if "dataset.json" in filenames and "plans.json" in filenames:
            model = _parse_leaf(dirpath)
            if model is not None:
                models.append(model)
    return sorted(models, key=lambda m: (m.dataset_name, m.configuration, m.trainer))


def resolve_default_model(models):
    """The bundled default model if present, otherwise the first model found."""
    default_key = (DEFAULT_DATASET_NAME, DEFAULT_TRAINER, DEFAULT_PLANS, DEFAULT_CONFIGURATION)
    for model in models:
        if (model.dataset_name, model.trainer, model.plans, model.configuration) == default_key:
            return model
    return models[0] if models else None


def get_label_names(labels):
    """{label_id: name} for all non-background labels, sorted by id."""
    return {
        label_id: name
        for name, label_id in sorted(labels.items(), key=lambda item: item[1])
        if label_id != 0
    }


def build_colormap(labels, palette_name):
    """{label_id: [r, g, b]} for all non-background labels, sized to fit any label count."""
    label_names = get_label_names(labels)
    colors = _sample_palette(palette_name, len(label_names))
    colormap = {label_id: colors[i] for i, label_id in enumerate(label_names)}
    if palette_name == "Default":
        for label_id, name in label_names.items():
            if name in DEFAULT_LABEL_COLORS:
                colormap[label_id] = list(DEFAULT_LABEL_COLORS[name])
    return colormap


def _sample_palette(palette_name, n):
    if n == 0:
        return []

    if palette_name == "Grayscale":
        values = np.linspace(0.15, 0.9, n)
        return [[int(v * 255)] * 3 for v in values]

    cmap_name = _PALETTE_TO_CMAP.get(palette_name, "tab10")
    if cmap_name == "tab10" and n > 10:
        cmap_name = "tab20"
    elif cmap_name == "Set1" and n > 9:
        cmap_name = "tab20b"

    cmap = colormaps[cmap_name]
    # Qualitative palettes are short lists of distinct colors, so take the first n.
    # Continuous colormaps (e.g. viridis) are 256-entry gradients, so sample them evenly.
    if isinstance(cmap, ListedColormap) and cmap_name in _QUALITATIVE_CMAPS and n <= len(cmap.colors):
        rgba_values = cmap.colors[:n]
    else:
        rgba_values = [cmap(x) for x in np.linspace(0, 1, n)]

    return [[int(channel * 255) for channel in rgba[:3]] for rgba in rgba_values]
