from collections import defaultdict
from pathlib import Path
from monai.transforms import (
    ScaleIntensity,
    Orientation,
)
from monai.transforms import Compose
from fetalsyngen.generator.model import FetalSynthGen
from hydra.utils import instantiate
from fetalsyngen.utils.image_reading import SimpleITKReader
import time
import torch
import numpy as np
from monai.data import MetaTensor
import torchio as tio
import pandas as pd


class FetalDataset:
    """Abstract class defining a dataset for loading fetal data."""

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        load_segmentations: bool = True,
        apply_mri_augm: bool = False,
    ) -> dict:
        """
        Args:
            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use. If None, all subjects are used.


        """
        super().__init__()
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.load_segmentations = load_segmentations
        self.bids_path = Path(bids_path)
        self.subjects = self.find_subjects(sub_list)
        if self.subjects is None:
            self.subjects = [x.name for x in self.bids_path.glob("sub-*")]
        self.sub_ses = [
            (x, y) for x in self.subjects for y in self._get_ses(self.bids_path, x)
        ]
        self.loader = SimpleITKReader()
        self.scaler = ScaleIntensity(minv=0, maxv=1)
        self.orientation = Orientation(axcodes="RAS")
        self.apply_mri_augm = apply_mri_augm
        self.img_paths, img_subject_sessions = self._load_bids_path(
            self.bids_path, self.img_suffix
        )

        if load_segmentations:
            # load the segmentation paths and ensure they are in the same order as images
            self.segm_paths, segm_subject_sessions = self._load_bids_path(
                self.bids_path, self.seg_suffix
            )

            if len(self.img_paths) != 0 and len(self.segm_paths) != 0:
                # ensure that image and segmentation paths are in the same order and have only overlapping subjects
                overlapping_subjects = set(img_subject_sessions) & set(
                    segm_subject_sessions
                )
                if len(overlapping_subjects) == 0:
                    raise ValueError(
                        f"No overlapping subjects found between image and segmentation paths."
                    )

                # realign img_paths and segm_paths to the filtered self.sub_ses
                img_map = {
                    sess: path
                    for sess, path in zip(img_subject_sessions, self.img_paths)
                }
                seg_map = {
                    sess: path
                    for sess, path in zip(segm_subject_sessions, self.segm_paths)
                }
                sorted_keys = sorted(overlapping_subjects)
                self.img_paths = [img_map[key] for key in sorted_keys]
                self.segm_paths = [seg_map[key] for key in sorted_keys]
                self.sub_ses = sorted_keys
                self.subjects = list(set([x[0] for x in self.sub_ses]))
        else:
            self.segm_paths = None
            self.sub_ses = img_subject_sessions
            self.subjects = list(set([x[0] for x in img_subject_sessions]))

    def find_subjects(self, sub_list):
        subj_found = [x.name for x in Path(self.bids_path).glob("sub-*")]
        return (
            sorted(list(set(subj_found) & set(sub_list)))
            if sub_list is not None
            else None
        )

    def _sub_ses_string(self, sub, ses):
        return f"{sub}_{ses}" if ses is not None else sub

    def _sub_ses_idx(self, idx):
        sub, ses = self.sub_ses[idx]
        return self._sub_ses_string(sub, ses)

    def _get_ses(self, bids_path, sub):
        """Get the session names for the subject."""
        sub_path = bids_path / sub
        ses_dir = [x for x in sub_path.iterdir() if x.is_dir()]
        ses = []
        for s in ses_dir:
            if "anat" in s.name:
                ses.append(None)
            else:
                ses.append(s.name)

        return sorted(ses, key=lambda x: x or "")

    def _get_pattern(self, sub, ses, suffix, extension=".nii.gz"):
        """Get the pattern for the file name."""
        if ses is None:
            return f"{sub}/anat/{sub}*_{suffix}{extension}"
        else:
            return f"{sub}/{ses}/anat/{sub}*_{suffix}{extension}"

    def _load_bids_path(self, path, suffix):
        """
        "Check that for a given path, all subjects have a file with the provided suffix
        """
        files_paths = []
        subject_sessions = []
        for sub, ses in self.sub_ses:

            pattern = self._get_pattern(sub, ses, suffix)
            files = list(path.glob(pattern))
            if len(files) == 0:
                print(
                    f"No files found for requested subject {sub} in {path} "
                    f"({pattern} returned nothing)"
                )
                continue
            elif len(files) > 1:
                raise ValueError(
                    f"Multiple files found for requested subject {sub} in {path} "
                    f"({pattern} returned {files}). "
                    "Please ensure that there is only one file per subject/session."
                )
            files_paths.append(files[0])
            subject_sessions.append((sub, ses))

        return files_paths, subject_sessions

    def __len__(self):
        return len(self.sub_ses)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )


class FetalTestDataset(FetalDataset):
    """Dataset class for loading fetal images offline.
    Used to load test/validation data.

    Use the `transforms` argument to pass additional processing steps
    (scaling, resampling, cropping, etc.).
    """

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        transforms: Compose | None = None,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        load_segmentations: bool = True,
    ):
        """
        Args:
            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use. If None, all subjects are used.
            transforms: Compose object with the transformations to apply.
                Default is None, no transformations are applied.

        !!! Note
            We highle recommend using the `transforms` arguments with at
            least the re-oriented transform to RAS and the intensity scaling
            to `[0, 1]` to ensure the data consistency.

            See [inference.yaml](https://github.com/Medical-Image-Analysis-Laboratory/fetalsyngen/blob/dev/configs/dataset/transforms/inference.yaml) for an example of the transforms configuration.
        """
        super().__init__(
            bids_path,
            sub_list,
            img_suffix,
            seg_suffix,
            load_segmentations,
        )
        self.transforms = transforms

    def _load_data(self, idx):
        # load the image and segmentation
        image = self.loader(self.img_paths[idx])
        segm = self.loader(self.segm_paths[idx]) if self.load_segmentations else None
        if len(image.shape) == 3:
            # add channel dimension
            image = image.unsqueeze(0)
            segm = segm.unsqueeze(0) if segm is not None else None
        elif len(image.shape) != 4:
            raise ValueError(f"Expected 3D or 4D image, got {len(image.shape)}D image.")

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        data = {"image": image, "name": name}
        if segm is not None:
            data["label"] = segm.long()

        return data

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            Dictionary with the `image` , `label` and the `name`
                keys. `image` and `label` are  `torch.float32`
                [`monai.data.meta_tensor.MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
                instances  with dimensions `(1, H, W, D)` and `name` is a string
                of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.


        """
        data = self._load_data(idx)

        if self.transforms:
            data = self.transforms(data)
        if "label" in data:
            # ensure label is long tensor
            # to avoid issues with collate_fn
            # and loss functions expecting long tensors
            data["label"] = data["label"].long()
        return data

    def reverse_transform(self, data: dict) -> dict:
        """Reverse the transformations applied to the data.

        Args:
            data: Dictionary with the `image` and `label` keys,
                like the one returned by the `__getitem__` method.

        Returns:
            Dictionary with the `image` and `label` keys where
                the transformations are reversed.
        """
        if self.transforms:
            data = self.transforms.inverse(data)
        return data


class FetalSynthDataset(FetalDataset):
    """Dataset class for generating/augmenting on-the-fly fetal images" """

    def __init__(
        self,
        bids_path: str,
        generator: FetalSynthGen,
        seed_path: str | None,
        sub_list: list[str] | None,
        load_image: bool = False,
        image_as_intensity: bool = False,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        apply_mri_augm: bool = False,
    ):
        """

        Args:
            bids_path: Path to the bids-formatted folder with the data.
            seed_path: Path to the folder with the seeds to use for
                intensity sampling. See `scripts/seed_generation.py`
                for details on the data formatting. If seed_path is None,
                the intensity  sampling step is skipped and the output image
                intensities will be based on the input image.
            generator: a class object defining a generator to use.
            sub_list: List of the subjects to use. If None, all subjects are used.
            load_image: If **True**, the image is loaded and passed to the generator,
                where it can be used as the intensity prior instead of a random
                intensity sampling or spatially deformed with the same transformation
                field as segmentation and the syntehtic image. Default is **False**.
            image_as_intensity: If **True**, the image is used as the intensity prior,
                instead of sampling the intensities from the seeds. Default is **False**.
        """
        super().__init__(
            bids_path, sub_list, img_suffix, seg_suffix, apply_mri_augm=apply_mri_augm
        )
        self.seed_path = Path(seed_path) if isinstance(seed_path, str) else None
        self.load_image = load_image
        self.generator = generator
        self.image_as_intensity = image_as_intensity
        # parse seeds paths
        if not self.image_as_intensity and isinstance(self.seed_path, Path):
            if not self.seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {self.seed_path} does not exist."
                )
            else:
                self._load_seed_path()

        self.transforms = tio.Compose(
            [
                # SPATIAL
                tio.RandomMotion(degrees=10, translation=5, num_transforms=2, p=0.2),
                tio.RandomGhosting(num_ghosts=(1, 10), intensity=(0.1, 0.5), p=0.2),
                tio.RandomSpike(num_spikes=1, intensity=0.3, p=0.2),
                # INTENSITY
                tio.RandomNoise(mean=0.0, std=(0.0, 0.25), p=0.2),
                tio.RescaleIntensity(out_min_max=(0.0, 1.0)),
            ]
        )

    def apply_torchio_augmentations(self, data: dict) -> dict:
        """
        data: {
            "image": torch.Tensor of shape [C, …],
            "label": torch.Tensor of shape [C, …],
            "name":  any (e.g. str)
        }
        returns the same dict with image/label replaced by the augmented ones.
        """
        # wrap into a Subject
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=data["image"]),
            label=tio.LabelMap(tensor=data["label"]),
        )

        # apply all transforms
        transformed = self.transforms(subject)

        # extract back to torch.Tensor
        return {
            "image": transformed["image"].data,
            "label": transformed["label"].data.long(),
            "name": data["name"],
        }

    def _load_seed_path(self):
        """Load the seeds for the subjects."""
        self.seed_paths = {
            self._sub_ses_string(sub, ses): defaultdict(dict)
            for (sub, ses) in self.sub_ses
        }
        avail_seeds = [
            int(x.name.replace("subclasses_", ""))
            for x in self.seed_path.glob("subclasses_*")
        ]
        min_seeds_available = min(avail_seeds)
        max_seeds_available = max(avail_seeds)
        for n_sub in range(
            min_seeds_available,
            max_seeds_available + 1,
        ):
            seed_path = self.seed_path / f"subclasses_{n_sub}"
            if not seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {seed_path} does not exist."
                )
            # load the seeds for the subjects for each meta label 1-4
            for i in range(1, self.generator.intensity_generator.meta_labels + 1):
                files, __ = self._load_bids_path(seed_path, f"mlabel_{i}")
                for (sub, ses), file in zip(self.sub_ses, files):
                    sub_ses_str = self._sub_ses_string(sub, ses)
                    self.seed_paths[sub_ses_str][n_sub][i] = file

    def sample(self, idx, genparams: dict = {}) -> tuple[dict, dict]:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.
            genparams (dict): Dictionary with generation parameters.
                Used for fixed generation. Should follow exactly the same structure
                and be of the same type as the returned generation parameters.
                Can be used to replicate the augmentations (power)
                used for the generation of a specific sample.
        Returns:
            Dictionaries with the generated data and the generation parameters.
                First dictionary contains the `image`, `label` and the `name` keys.
                The second dictionary contains the parameters used for the generation.

        !!! Note
            The `image` is scaled to `[0, 1]` and oriented with the `label` to **RAS**
            and returned on the device  specified in the `generator` initialization.
        """
        # use generation_params to track the parameters used for the generation
        generation_params = {}

        image = self.loader(self.img_paths[idx]) if self.load_image else None
        segm = self.loader(self.segm_paths[idx]).astype(torch.long)

        # orient to RAS for consistency
        image = (
            self.orientation(image.unsqueeze(0)).squeeze(0) if self.load_image else None
        )
        segm = self.orientation(segm.unsqueeze(0)).squeeze(0)

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        # initialize seeds as dictionary
        # with paths to the seeds volumes
        # or None if image is to be used as intensity prior
        if self.seed_path is not None:
            seeds = self.seed_paths[name]
        if self.image_as_intensity:
            seeds = None

        # log input data
        generation_params["idx"] = idx
        generation_params["img_paths"] = str(self.img_paths[idx])
        generation_params["segm_paths"] = str(self.img_paths[idx])
        generation_params["seeds"] = str(self.seed_path)
        generation_time_start = time.time()

        # generate the synthetic data
        gen_output, segmentation, image, synth_params = self.generator.sample(
            image=image, segmentation=segm, seeds=seeds, genparams=genparams
        )

        # scale the images to [0, 1]
        gen_output = self.scaler(gen_output)
        image = self.scaler(image) if image is not None else None

        # ensure image and segmentation are on the cpu
        gen_output = gen_output.cpu()
        segmentation = segmentation.cpu()
        image = image.cpu() if image is not None else None

        generation_params = {**generation_params, **synth_params}
        generation_params["generation_time"] = time.time() - generation_time_start
        data_out = {
            "image": gen_output.unsqueeze(0),
            "label": segmentation.unsqueeze(0).long(),
            "name": name,
        }

        if self.transforms is not None and self.apply_mri_augm:
            # apply torchio augmentations
            data_out = self.apply_torchio_augmentations(data_out)

        return data_out, generation_params

    def __getitem__(self, idx) -> dict:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dictionary with the `image`, `label` and the `name` keys.
                `image` and `label` are `torch.float32`
                [`monai.data.meta_tensor.MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
                and `name` is a string of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.

        !!!Note
            The `image` is scaled to `[0, 1]` and oriented to **RAS** and returned on the device
            specified in the `generator` initialization.
        """
        data_out, generation_params = self.sample(idx)
        self.generation_params = generation_params
        return data_out

    def sample_with_meta(self, idx: int, genparams: dict = {}) -> dict:
        """
        Retrieve a sample along with its generation parameters
        and store them in the same dictionary.

        Args:
            idx: The index of the sample to retrieve.
            genparams: Dictionary with generation parameters.
                Used for fixed generation. Should follow exactly the same structure
                and be of the same type as the returned generation parameters from the `sample()` method.
                Can be used to replicate the augmentations (power)
                used for the generation of a specific sample.

        Returns:
            A dictionary with `image`, `label`, `name` and `generation_params` keys.
        """

        data, generation_params = self.sample(idx, genparams=genparams)
        data["generation_params"] = generation_params
        return data


class MultiProtocolDataset:
    """Composes multiple `FetalDataset` instances and returns training
    samples tagged with a protocol conditioning vector for CoNeMOS-style FiLM
    conditioning.

    Each dataset entry is associated with a named annotation protocol
    (e.g. ``"drawem9_albert"`` or ``"feta"``).  Segmentation labels are remapped
    from their raw dataset-specific integers to a shared set of output channels
    defined by ``label_map_csv``.

    Args:
        dataset_entries: List of dicts, each with keys:
            - ``bids_path`` (str): Path to the BIDS folder.
            - ``split_file`` (str): Path to a CSV with columns
              ``participant_id`` and ``splits``.
            - ``protocol_name`` (str): Annotation protocol name.
        split: Split to load (e.g. ``"train"``).  Rows in each split CSV
            where ``splits == split`` are kept.
        label_map_csv: Path to a CSV with columns ``protocol``,
            ``raw_label`` (int), ``channel`` (int).  Defines how raw
            segmentation integers map to output channel indices per protocol.
        img_suffix: Default image file suffix used when an entry does not
            specify ``"img_suffix"`` (default ``"T2w"``).
        seg_suffix: Default segmentation file suffix used when an entry does
            not specify ``"seg_suffix"`` (default ``"dseg"``).
        transforms: Optional MONAI :class:`~monai.transforms.Compose` applied
            to each sample after loading and remapping.

    Per-entry suffix override
        Add ``"img_suffix"`` or ``"seg_suffix"`` keys to any entry dict to
        override the defaults for that dataset only.
    """

    def __init__(
        self,
        dataset_entries: list[dict],
        label_map_csv: str,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        transforms: Compose | None = None,
    ):

        label_df = pd.read_csv(label_map_csv)
        for col in ("protocol", "raw_label", "channel"):
            if col not in label_df.columns:
                raise ValueError(
                    f"label_map_csv is missing required column '{col}'. "
                    f"Found: {label_df.columns.tolist()}"
                )

        self.transforms = transforms

        # protocol_registry: protocol_name -> integer index (first-seen order)
        self.protocol_registry: dict[str, int] = {}

        self._datasets: list[FetalDataset] = []
        # per-dataset label remap: raw_label_int -> channel_int
        self._label_maps: list[dict[int, int]] = []
        self._label_luts: list[torch.Tensor] = []
        # per-dataset protocol one-hot tensor (built after all protocols seen)
        self._protocol_names: list[str] = []
        # per-dataset train type: "synth" or "real"
        self._train_types: list[str] = []
        # per-dataset flag: True means FetalTestDataset (no generator)
        self._is_test: list[bool] = []

        self.sample_index: list[tuple[int, int]] = []

        for ds_idx, entry in enumerate(dataset_entries):
            bids_path = entry["bids_path"]
            sub_list = entry["sub_list"]
            protocol_name = entry["protocol_name"]
            is_test = entry.get("is_test", False)
            train_type = entry.get("train_type", "real")
            entry_img_suffix = entry.get("img_suffix", img_suffix)
            entry_seg_suffix = entry.get("seg_suffix", seg_suffix)
            assert train_type in ("synth", "real"), (
                f"train_type must be 'synth' or 'real', got '{train_type}'"
            )
            self._train_types.append(train_type)
            self._is_test.append(is_test)

            # Register protocol
            if protocol_name not in self.protocol_registry:
                self.protocol_registry[protocol_name] = len(self.protocol_registry)
            self._protocol_names.append(protocol_name)

            if is_test:
                ds = FetalTestDataset(
                    bids_path=bids_path,
                    sub_list=sub_list,
                    transforms=self.transforms,
                    img_suffix=entry_img_suffix,
                    seg_suffix=entry_seg_suffix,
                )
            elif train_type == "synth":
                if "generator" not in entry:
                    raise ValueError(
                        f"Entry for protocol '{protocol_name}' has train_type='synth' "
                        "but is missing required key 'generator'."
                    )
                ds = FetalSynthDataset(
                    bids_path=bids_path,
                    seed_path=entry.get("seed_path", None),
                    sub_list=sub_list,
                    load_image=False,
                    image_as_intensity=False,
                    generator=entry["generator"],
                    img_suffix=entry_img_suffix,
                    seg_suffix=entry_seg_suffix,
                    apply_mri_augm=entry.get("apply_mri_augm", False),
                )
            else:
                # train_type="real": real images through FetalSynthDataset
                # (load_image=True, image_as_intensity=True) so spatial deformation
                # still applies, consistent with original DataModule.
                if entry.get("generator") is None:
                    raise ValueError(
                        f"Entry for protocol '{protocol_name}' has train_type='real' "
                        "but is missing required key 'generator'. Pass a generator "
                        "(with augmentations nulled out if needed) as in the existing experiments."
                    )
                ds = FetalSynthDataset(
                    bids_path=bids_path,
                    seed_path=None,
                    sub_list=sub_list,
                    load_image=True,
                    image_as_intensity=True,
                    generator=entry["generator"],
                    img_suffix=entry_img_suffix,
                    seg_suffix=entry_seg_suffix,
                    apply_mri_augm=entry.get("apply_mri_augm", False),
                )
            self._datasets.append(ds)

            # Build label remap for this protocol
            proto_rows = label_df[label_df["protocol"] == protocol_name]
            label_map: dict[int, int] = {
                int(r["raw_label"]): int(r["channel"])
                for _, r in proto_rows.iterrows()
            }
            self._label_maps.append(label_map)

            # Build LUT for vectorised remapping in __getitem__.
            # size256 covers all uint8 label values
            # any wild value not in label_map stays 0 (background), matching
            # the original zeros_like behaviour.
            lut = torch.zeros(256, dtype=torch.long)
            for raw_val, channel in label_map.items():
                lut[raw_val] = channel
            self._label_luts.append(lut)

            # Extend flat sample index
            for local_idx in range(len(ds)):
                self.sample_index.append((ds_idx, local_idx))

        # Build one-hot tensors now that all protocols are registered
        num_protocols = len(self.protocol_registry)
        self._protocol_vecs: list[torch.Tensor] = []
        for name in self._protocol_names:
            vec = torch.zeros(num_protocols, dtype=torch.float32)
            vec[self.protocol_registry[name]] = 1.0
            self._protocol_vecs.append(vec)

        # Number of output channels = max channel index across all label maps + 1
        # Exposed so the model can be instantiated with the correct output size.
        all_channels = [c for lm in self._label_maps for c in lm.values()]
        self.num_channels: int = max(all_channels) + 1 if all_channels else 0

        self._scaler = ScaleIntensity(minv=0, maxv=1)
        self._orientation = Orientation(axcodes="RAS")

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict:
        """Return a training sample with protocol conditioning.

        Returns:
            Dictionary with keys:

            - ``"image"``: ``torch.float32`` tensor ``(1, H, W, D)``.
            - ``"label"``: ``torch.long`` tensor ``(1, H, W, D)`` with
              channel indices from ``label_map_csv`` (unmapped labels → 0).
            - ``"protocol_vec"``: ``torch.float32`` one-hot tensor of length
              ``num_protocols``.
            - ``"protocol_name"``: str.
            - ``"name"``: str in ``sub_ses`` (or just ``sub``) format.
        """
        ds_idx, local_idx = self.sample_index[idx]
        ds = self._datasets[ds_idx]

        if self._is_test[ds_idx] or self._train_types[ds_idx] == "real":
            # FetalTestDataset or FetalSynthDataset(real mode):
            # delegate entirely to ds[local_idx] — orientation, scaling,
            # and any transforms are handled inside the dataset.
            data_out = ds[local_idx]
            image = data_out["image"]
            segm = data_out["label"]
            name = data_out["name"]
        else:
            # train_type="synth": use sample() to get image + raw label integers.
            # FetalSynthDataset.sample() handles orientation, scaling, augmentation.
            data_out, _ = ds.sample(local_idx)
            image = data_out["image"]
            segm = data_out["label"]
            name = data_out["name"]

        # Remap segmentation labels to shared output channels via prebuilt LUT.
        # Values outside the LUT range (not in label_map) clamp to 0 = background.
        lut = self._label_luts[ds_idx]
        segm = lut[segm.clamp(min=0, max=255)]

        data = {
            "image": image.float(),
            "label": segm.long(),
            "protocol_vec": self._protocol_vecs[ds_idx],
            "protocol_name": self._protocol_names[ds_idx],
            "name": name,
        }

        if self.transforms is not None:
            data = self.transforms(data)

        return data


if __name__ == "__main__":
    # Example usage
    dataset = FetalTestDataset(
        bids_path="/media/vzalevskyi/data/FETA_challenge/merged_feta_spinabifida/derivatives/resampled05",
        # generator=None,
        # seed_path="/media/vzalevskyi/data/FETA_challenge/merged_feta_spinabifida/derivatives/seeds",
        sub_list=None,
        # load_image=True,
        # image_as_intensity=False,
    )
    print(f"Number of subjects: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample: {sample}")
