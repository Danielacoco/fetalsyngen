"""Resamples CHUV fetal T2w and dseg to 0.75mm isotropic, center-cropped/padded to 256³.

Native images are 256x256x256 @ 1.125mm -> 384x384x384 @ 0.75mm, then center-cropped to 256³.
Logs any subjects where brain tissue is lost in the crop.
"""

from tqdm import tqdm
from pathlib import Path

import monai

res = 0.75
target_size = (256, 256, 256)

bids_path = Path("/media/danielaco/HDD2TB/DataDaniela/CHUV_fetal")
out_path = bids_path / "resampled75"

loader = monai.transforms.LoadImaged(keys=["T2w", "label"], allow_missing_keys=True)
resample_transform = monai.transforms.Spacingd(
    keys=["T2w", "label"],
    pixdim=(res, res, res),
    mode=("bilinear", "nearest"),
    allow_missing_keys=True,
)
orientation = monai.transforms.Orientationd(
    keys=["T2w", "label"], axcodes="RAS", allow_missing_keys=True
)
cropper = monai.transforms.CenterSpatialCropd(
    keys=["T2w", "label"], roi_size=target_size, allow_missing_keys=True
)
padder = monai.transforms.SpatialPadd(
    keys=["T2w", "label"],
    spatial_size=target_size,
    mode="constant",
    allow_missing_keys=True,
)


def check_brain_crop(label, target_size, res):
    """Return list of strings describing brain tissue lost per axis, empty if none."""
    shape = label.shape[1:]  # (X, Y, Z), skip channel dim
    axes = ["X", "Y", "Z"]
    losses = []
    for i, (s, t) in enumerate(zip(shape, target_size)):
        if s <= t:
            continue
        n_before = (s - t) // 2
        n_after = (s - t) - n_before

        # collapse all dims except axis i to get per-slice occupancy
        dims_to_reduce = [0] + [d + 1 for d in range(len(shape)) if d != i]

        slab_before = label[tuple([slice(None) if d != i + 1 else slice(0, n_before)
                                   for d in range(label.ndim)])]
        slab_after = label[tuple([slice(None) if d != i + 1 else slice(s - n_after, s)
                                  for d in range(label.ndim)])]

        occupied_before = slab_before.any(dim=dims_to_reduce)  # shape [n_before]
        occupied_after = slab_after.any(dim=dims_to_reduce)    # shape [n_after]

        # depth = how many voxels from the edge actually contain brain
        depth_before = (occupied_before.nonzero(as_tuple=False)[-1].item() + 1
                        if occupied_before.any() else 0)
        depth_after = (occupied_after.shape[0] - occupied_after.flip(0).nonzero(as_tuple=False)[-1].item()
                       if occupied_after.any() else 0)

        lost_mm_before = depth_before * res
        lost_mm_after = depth_after * res
        if lost_mm_before or lost_mm_after:
            losses.append(f"{axes[i]}: {lost_mm_before + lost_mm_after:.1f}mm lost "
                          f"(front: {lost_mm_before:.1f}mm, back: {lost_mm_after:.1f}mm)")
    return losses


subjects = sorted(bids_path.glob("sub-*"))
print(f"Found {len(subjects)} subjects in {bids_path}")

for sub in tqdm(subjects):
    for ses in sorted(sub.glob("ses-*")):
        session = ses.name

        saver = monai.transforms.SaveImaged(
            keys=["T2w", "label"],
            output_dir=out_path / f"{sub.name}/{session}/anat/",
            output_postfix="",
            resample=False,
            separate_folder=False,
            print_log=False,
            allow_missing_keys=True,
            mode="nearest",
        )

        try:
            data = {}

            t2_files = list(sub.glob(f"{session}/anat/*_rec-mial_T2w.nii.gz"))
            label_files = list(sub.glob(f"{session}/anat/*_rec-mial_dseg.nii.gz"))

            if t2_files:
                data["T2w"] = str(t2_files[0])
            if label_files:
                data["label"] = str(label_files[0])

            if not data:
                print(f"No files found for {sub.name}/{session}, skipping.")
                continue

            data = loader(data)

            for key in ["T2w", "label"]:
                if key in data:
                    data[key] = data[key].unsqueeze(0)

            data = resample_transform(data)
            data = orientation(data)

            if "label" in data:
                losses = check_brain_crop(data["label"], target_size, res)
                if losses:
                    print(f"  CROP WARNING {sub.name}/{session}: {', '.join(losses)}")

            data = cropper(data)
            data = padder(data)
            saver(data)

        except Exception as e:
            print(f"Error processing {sub.name}/{session}: {e}")
            continue
