"""Format-aware export of raw images and read-back of segmentation results.

2D nnU-Net models work on one file per slice (e.g. .tif), 3D models on a single
volumetric file (e.g. .nii.gz). The model's dataset.json decides which layout is used,
and nnU-Net's reader/writer registry reads the segmentation results back.
"""

import os

import imageio.v3 as imageio
import numpy as np
import SimpleITK as sitk
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json


def get_reader_writer_cls(dataset_json):
    """The nnU-Net BaseReaderWriter subclass appropriate for this model's file format."""
    return determine_reader_writer_from_dataset_json(dataset_json, verbose=False)


def export_case_images(model, fat_slices, water_slices, pixel_spacing, slice_spacing, output_dir,
                       first_slice_number=1):
    """Write the loaded Fat/Water slices as nnU-Net raw input, in the layout the model expects.

    2D configurations: one case per slice, named by its ABSOLUTE 1-based slice number
    (dataset_{n:04d}_0000/_0001, n starting at first_slice_number), because the viewer
    looks predictions up as dataset_{absolute slice:04d}.
    3D configurations: one case for the whole slice range, as a single multi-slice volume.
    """
    os.makedirs(output_dir, exist_ok=True)
    ext = model.file_ending

    if not model.is_3d:
        for i, (fat, water) in enumerate(zip(fat_slices, water_slices)):
            slice_number = first_slice_number + i
            fat_filename = os.path.join(output_dir, f"dataset_{slice_number:04d}_0000{ext}")
            water_filename = os.path.join(output_dir, f"dataset_{slice_number:04d}_0001{ext}")
            imageio.imwrite(fat_filename, fat.astype(np.uint16))
            imageio.imwrite(water_filename, water.astype(np.uint16))
        return

    row_spacing, col_spacing = pixel_spacing
    for channel_index, slices in enumerate((fat_slices, water_slices)):
        volume = np.stack(slices, axis=0).astype(np.float32)
        image = sitk.GetImageFromArray(volume)
        image.SetSpacing((float(col_spacing), float(row_spacing), float(slice_spacing)))
        filename = os.path.join(output_dir, f"dataset_0001_{channel_index:04d}{ext}")
        sitk.WriteImage(image, filename)


def read_segmentation_array(reader_writer, prediction_path):
    """Read a segmentation file, dropping the channel dimension nnU-Net's readers add.

    Returns shape (1, H, W) for a single 2D slice, or (Z, H, W) for a 3D volume.
    """
    seg, _ = reader_writer.read_seg(prediction_path)
    return seg[0]
