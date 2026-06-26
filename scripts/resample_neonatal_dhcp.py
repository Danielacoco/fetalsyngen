"""Resamples neonatal dHCP T1w, T2w and all segmentations to 0.75mm isotropic, crop-padded to 256³.

Source is resampled05 (0.5mm, 256³ = 128mm FOV) — original dHCP data not stored locally.
After resampling to 0.75mm the volume shrinks to ~171³; padder brings it back to 256³.
"""

from tqdm import tqdm
from pathlib import Path

import monai

res = 0.75
target_size = (256, 256, 256)

bids_path = Path("/media/danielaco/HDD2TB/DataDaniela/neonates_dHCP/resampled05")
out_path = Path("/media/danielaco/HDD2TB/DataDaniela/neonates_dHCP/resampled75")

ALL_KEYS = ["T1w", "T2w", "drawem9"]
IMAGE_KEYS = ["T1w", "T2w"]
LABEL_KEYS = ["drawem9"]

loader = monai.transforms.LoadImaged(keys=ALL_KEYS, allow_missing_keys=True)
resample_transform = monai.transforms.Spacingd(
    keys=ALL_KEYS,
    pixdim=(res, res, res),
    mode=("bilinear", "bilinear", "nearest"),
    allow_missing_keys=True,
)
orientation = monai.transforms.Orientationd(
    keys=ALL_KEYS, axcodes="RAS", allow_missing_keys=True
)
cropper = monai.transforms.CenterSpatialCropd(
    keys=ALL_KEYS, roi_size=target_size, allow_missing_keys=True
)
padder = monai.transforms.SpatialPadd(
    keys=ALL_KEYS,
    spatial_size=target_size,
    mode="constant",
    allow_missing_keys=True,
)


def check_brain_crop(label, target_size, res):
    """Return list of strings describing brain tissue lost per axis, empty if none."""
    shape = label.shape[1:]
    axes = ["X", "Y", "Z"]
    losses = []
    for i, (s, t) in enumerate(zip(shape, target_size)):
        if s <= t:
            continue
        n_before = (s - t) // 2
        n_after = (s - t) - n_before
        dims_to_reduce = [0] + [d + 1 for d in range(len(shape)) if d != i]
        slab_before = label[tuple([slice(None) if d != i + 1 else slice(0, n_before)
                                   for d in range(label.ndim)])]
        slab_after = label[tuple([slice(None) if d != i + 1 else slice(s - n_after, s)
                                  for d in range(label.ndim)])]
        occupied_before = slab_before.any(dim=dims_to_reduce)
        occupied_after = slab_after.any(dim=dims_to_reduce)
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
            keys=ALL_KEYS,
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
            anat = ses / "anat"

            t1_files = list(anat.glob("*_T1w.nii.gz"))
            t2_files = [f for f in anat.glob("*_T2w.nii.gz") if "reoriented" not in f.name]
            drawem9_files = list(anat.glob("*_desc-drawem9_dseg.nii.gz"))

            if t1_files:
                data["T1w"] = str(t1_files[0])
            if t2_files:
                data["T2w"] = str(t2_files[0])
            if drawem9_files:
                data["drawem9"] = str(drawem9_files[0])

            if not data:
                print(f"No files found for {sub.name}/{session}, skipping.")
                continue

            data = loader(data)

            for key in ALL_KEYS:
                if key in data:
                    data[key] = data[key].unsqueeze(0)

            data = resample_transform(data)
            data = orientation(data)

            ref_label = next((data[k] for k in LABEL_KEYS if k in data), None)
            if ref_label is not None:
                losses = check_brain_crop(ref_label, target_size, res)
                if losses:
                    print(f"  CROP WARNING {sub.name}/{session}: {', '.join(losses)}")

            data = cropper(data)
            data = padder(data)
            saver(data)

        except Exception as e:
            print(f"Error processing {sub.name}/{session}: {e}")
            continue
