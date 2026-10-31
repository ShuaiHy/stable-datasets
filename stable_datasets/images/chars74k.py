import io
import re
import tarfile

import numpy as np
from PIL import Image as PILImage
from scipy.io import loadmat

from stable_datasets.schema import (
    ClassLabel,
    DatasetInfo,
    DatasetSource,
    DownloadInfo,
    Features,
    Version,
)
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.splits import SplitGenerator
from stable_datasets.utils import BaseDatasetBuilder, bulk_download


class Chars74K(BaseDatasetBuilder):
    """Chars74K English handwritten character dataset."""

    VERSION = Version("1.1.0")

    LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

    BASE_URL = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/"

    # Select the first of the 30 official splits.
    OFFICIAL_SPLIT_INDEX = 0

    IMAGE_PATTERN = re.compile(
        r"^English/Hnd/Img/Sample(\d{3})/"
        r"img(\d{3})-(\d{3})\.png$",
        re.IGNORECASE,
    )

    SOURCE = DatasetSource(
        homepage="https://teodecampos.github.io/chars74k/",
        assets={
            "train": DownloadInfo(url=BASE_URL + "EnglishHnd.tgz"),
            "test": DownloadInfo(url=BASE_URL + "EnglishHnd.tgz"),
        },
        citation="""@inproceedings{deCampos2009chars74k,
            title={Character Recognition in Natural Images},
            author={de Campos, Teofilo E. and Babu, Bodla Rakesh and Varma, Manik},
            booktitle={VISAPP},
            year={2009}
        }""",
    )

    def _info(self):
        return DatasetInfo(
            description=(
                "Chars74K EnglishHnd contains 3,410 handwritten "
                "character images from 62 classes: digits, uppercase "
                "letters, and lowercase letters. This builder uses "
                "the first official train/test split provided by "
                "the Chars74K dataset."
            ),
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=self.LABELS),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
        )

    def _split_generators(self):
        """Download images and official split definitions."""

        image_info = self.SOURCE["assets"]["train"]

        split_info = DownloadInfo(url=self.BASE_URL + "Lists.tgz")

        download_dir = getattr(self, "_raw_download_dir", None)

        if download_dir is None:
            from stable_datasets.utils import _default_dest_folder

            download_dir = _default_dest_folder()

        image_path, list_path = bulk_download(
            [image_info, split_info],
            dest_folder=download_dir,
        )

        return [
            SplitGenerator(
                name="train",
                gen_kwargs={
                    "data_path": image_path,
                    "list_path": list_path,
                    "split": "train",
                },
            ),
            SplitGenerator(
                name="test",
                gen_kwargs={
                    "data_path": image_path,
                    "list_path": list_path,
                    "split": "test",
                },
            ),
        ]

    @staticmethod
    def _matlab_string(value):
        """Convert MATLAB string data into a Python string."""

        value = np.asarray(value)

        while value.size == 1 and value.dtype == object:
            value = np.asarray(value.item())

        if value.dtype.kind in ("U", "S"):
            return "".join(str(x) for x in value.flat).strip()

        raise ValueError(f"Unsupported MATLAB filename format: {value.dtype}")

    @classmethod
    def _normalized_image_key(cls, name):
        """Normalize official filenames and archive image paths."""

        name = str(name).replace("\\", "/").strip()
        name = name.removeprefix("./")

        match = re.search(
            r"(Sample\d{3}/img\d{3}-\d{3})(?:\.png)?$",
            name,
            flags=re.IGNORECASE,
        )

        if match is None:
            raise ValueError(f"Unexpected official image name: {name}")

        return match.group(1).lower() + ".png"

    def _load_official_split(self, list_path, split):
        """Read the selected official EnglishHnd split."""

        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split: {split}")

        target = "Lists/English/Hnd/lists_var_size.mat"

        # Read the official MATLAB file.
        with tarfile.open(list_path, "r:gz") as archive:
            try:
                member = archive.getmember(target)
            except KeyError as exc:
                raise ValueError(f"Official split file not found: {target}") from exc

            file = archive.extractfile(member)

            if file is None:
                raise ValueError("Could not read the official split file.")

            with file:
                mat_data = loadmat(
                    io.BytesIO(file.read()),
                    squeeze_me=True,
                    struct_as_record=False,
                )

        if "list" not in mat_data:
            raise ValueError("Missing MATLAB variable: list")

        data = mat_data["list"]

        # Validate required fields.
        required_fields = (
            "ALLnames",
            "ALLlabels",
            "TRNind",
            "TSTind",
        )

        for field in required_fields:
            if not hasattr(data, field):
                raise ValueError(f"Missing official field: {field}")

        names = np.asarray(data.ALLnames).reshape(-1)
        labels = np.asarray(data.ALLlabels).reshape(-1)

        if len(names) != len(labels):
            raise ValueError("ALLnames and ALLlabels sizes differ.")

        # Select the official train or test indices.
        field = "TRNind" if split == "train" else "TSTind"
        indices = np.asarray(getattr(data, field))

        if indices.ndim != 2:
            raise ValueError(f"Expected 2D official indices, got {indices.shape}")

        if not (0 <= self.OFFICIAL_SPLIT_INDEX < indices.shape[1]):
            raise ValueError("Official split index out of range.")

        indices = indices[:, self.OFFICIAL_SPLIT_INDEX]

        # MATLAB uses 0 for padding in the index arrays.
        # Remove padding before converting to Python indices.
        indices = indices[indices != 0]

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Official split indices must be integers.")

        # MATLAB: 1-based indexing
        # Python: 0-based indexing
        indices = indices.astype(np.int64) - 1

        if np.any(indices < 0) or np.any(indices >= len(names)):
            raise ValueError("Official split indices out of range.")

        if len(indices) != len(np.unique(indices)):
            raise ValueError("Duplicate indices in the official split.")

        selected = set()

        for index in indices:
            filename = self._matlab_string(names[index])
            key = self._normalized_image_key(filename)

            # Official labels are 1-based.
            official_label = int(labels[index]) - 1

            match = re.search(
                r"sample(\d{3})/",
                key,
                flags=re.IGNORECASE,
            )

            if match is None:
                raise ValueError(f"Invalid official filename: {filename}")

            folder_label = int(match.group(1)) - 1

            if not (0 <= official_label < 62):
                raise ValueError(f"Invalid official label: {official_label}")

            if official_label != folder_label:
                raise ValueError(f"Label mismatch for {filename}: official={official_label}, folder={folder_label}")

            if key in selected:
                raise ValueError(f"Duplicate official image: {filename}")

            selected.add(key)

        if not selected:
            raise ValueError(f"Official {split} split is empty.")

        return selected

    def _generate_examples(self, data_path, list_path, split):
        """Generate images using the official Chars74K split."""

        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split: {split}")

        selected = self._load_official_split(
            list_path=list_path,
            split=split,
        )

        found = set()

        with tarfile.open(data_path, "r:gz") as archive:
            members = sorted(
                (member for member in archive.getmembers() if member.isfile()),
                key=lambda member: member.name,
            )

            for member in members:
                if not member.name.lower().endswith(".png"):
                    continue

                path = member.name.removeprefix("./")

                match = self.IMAGE_PATTERN.fullmatch(path)

                if match is None:
                    raise ValueError(f"Unexpected image path: {member.name}")

                folder_id = int(match.group(1))
                filename_id = int(match.group(2))
                sample_id = int(match.group(3))

                if not (1 <= folder_id <= 62):
                    raise ValueError(f"Invalid class: {folder_id}")

                if filename_id != folder_id:
                    raise ValueError(f"Class mismatch: {member.name}")

                if not (1 <= sample_id <= 55):
                    raise ValueError(f"Invalid sample ID: {member.name}")

                key = self._normalized_image_key(path)

                # Only include images from the official split.
                if key not in selected:
                    continue

                if key in found:
                    raise ValueError(f"Duplicate image: {member.name}")

                file = archive.extractfile(member)

                if file is None:
                    raise ValueError(f"Could not read image: {member.name}")

                with file:
                    image_bytes = file.read()

                with PILImage.open(io.BytesIO(image_bytes)) as img:
                    image = img.convert("RGB")

                label = folder_id - 1

                found.add(key)

                yield (
                    member.name,
                    {
                        "image": image,
                        "label": label,
                    },
                )

        # Confirm every selected official image was found.
        missing = selected - found

        if missing:
            examples = sorted(missing)[:5]

            raise ValueError(
                f"{len(missing)} official {split} images were not found in EnglishHnd.tgz. Examples: {examples}"
            )
