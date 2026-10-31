import io
import tarfile
from urllib.request import urlretrieve

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from stable_datasets.images.chars74k import Chars74K


# --------------------------------------------------
# Test helpers
# --------------------------------------------------


def add_bytes_to_tar(archive, name, data):
    """Add a byte buffer to a TAR archive."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def make_test_tar(
    path,
    folder_ids=(1, 11, 62),
    sample_ids=(1, 2),
):
    """Create a small Chars74K-like image archive."""
    with tarfile.open(path, "w:gz") as archive:
        for folder_id in folder_ids:
            for sample_id in sample_ids:
                image = Image.new("RGB", (32, 40), color="white")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")

                name = f"English/Hnd/Img/Sample{folder_id:03d}/img{folder_id:03d}-{sample_id:03d}.png"
                add_bytes_to_tar(archive, name, buffer.getvalue())


def make_test_split_tar(path):
    """Create a synthetic MATLAB split with official fields."""

    names = np.array(
        [
            "Img/Sample001/img001-001",
            "Img/Sample001/img001-002",
            "Img/Sample011/img011-001",
            "Img/Sample062/img062-001",
        ],
        dtype=object,
    )

    labels = np.array([1, 1, 11, 62], dtype=np.uint16)

    # Two split columns with MATLAB-style 1-based indices.
    # Zeros are padding entries.
    train_indices = np.array(
        [
            [1, 2],
            [3, 4],
            [0, 0],
        ],
        dtype=np.uint16,
    )

    test_indices = np.array(
        [
            [2, 1],
            [4, 3],
            [0, 0],
        ],
        dtype=np.uint16,
    )

    mat_buffer = io.BytesIO()

    savemat(
        mat_buffer,
        {
            "list": {
                "ALLnames": names,
                "ALLlabels": labels,
                "TRNind": train_indices,
                "TSTind": test_indices,
            }
        },
    )

    with tarfile.open(path, "w:gz") as archive:
        add_bytes_to_tar(
            archive,
            "Lists/English/Hnd/lists_var_size.mat",
            mat_buffer.getvalue(),
        )


@pytest.fixture
def synthetic_archives(tmp_path):
    image_path = tmp_path / "EnglishHnd.tgz"
    list_path = tmp_path / "Lists.tgz"

    make_test_tar(image_path)
    make_test_split_tar(list_path)

    return image_path, list_path


# --------------------------------------------------
# Offline unit tests
# --------------------------------------------------


def test_label_mapping():
    assert len(Chars74K.LABELS) == 62
    assert Chars74K.LABELS[0] == "0"
    assert Chars74K.LABELS[9] == "9"
    assert Chars74K.LABELS[10] == "A"
    assert Chars74K.LABELS[35] == "Z"
    assert Chars74K.LABELS[36] == "a"
    assert Chars74K.LABELS[61] == "z"


def test_normalized_image_key():
    builder = object.__new__(Chars74K)

    official_name = "Img/Sample001/img001-001"
    archive_name = "English/Hnd/Img/Sample001/img001-001.png"

    assert builder._normalized_image_key(official_name) == builder._normalized_image_key(archive_name)


def test_official_split_indices(synthetic_archives):
    _, list_path = synthetic_archives

    builder = object.__new__(Chars74K)

    train = builder._load_official_split(list_path, "train")
    test = builder._load_official_split(list_path, "test")

    assert len(train) == 2
    assert len(test) == 2
    assert train.isdisjoint(test)

    assert "sample001/img001-001.png" in train
    assert "sample011/img011-001.png" in train

    assert "sample001/img001-002.png" in test
    assert "sample062/img062-001.png" in test


def test_train_test_splits(synthetic_archives):
    image_path, list_path = synthetic_archives

    builder = object.__new__(Chars74K)

    train = list(builder._generate_examples(image_path, list_path, "train"))

    test = list(builder._generate_examples(image_path, list_path, "test"))

    assert len(train) == 2
    assert len(test) == 2

    train_ids = {key for key, _ in train}
    test_ids = {key for key, _ in test}

    assert train_ids.isdisjoint(test_ids)

    for _, sample in train + test:
        assert 0 <= sample["label"] < 62
        assert sample["image"].mode == "RGB"


def test_invalid_split(synthetic_archives):
    image_path, list_path = synthetic_archives

    builder = object.__new__(Chars74K)

    with pytest.raises(ValueError, match="Unsupported split"):
        list(builder._generate_examples(image_path, list_path, "validation"))


def test_invalid_class_path(tmp_path, synthetic_archives):
    _, list_path = synthetic_archives

    image_path = tmp_path / "invalid.tgz"

    image = Image.new("RGB", (32, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    with tarfile.open(image_path, "w:gz") as archive:
        add_bytes_to_tar(
            archive,
            "English/Hnd/Img/Sample063/img063-001.png",
            buffer.getvalue(),
        )

    builder = object.__new__(Chars74K)

    with pytest.raises(ValueError, match="Invalid class"):
        list(builder._generate_examples(image_path, list_path, "train"))


def test_missing_official_split_file(tmp_path):
    path = tmp_path / "missing.tgz"

    with tarfile.open(path, "w:gz"):
        pass

    builder = object.__new__(Chars74K)

    with pytest.raises(ValueError, match="not found"):
        builder._load_official_split(path, "train")


# --------------------------------------------------
# Real-data integration test
# --------------------------------------------------


def test_real_chars74k_official_split(tmp_path):
    """Validate real data and the official train/test split."""

    # Download the official split file into an isolated directory.
    list_path = tmp_path / "Lists.tgz"
    split_url = Chars74K.BASE_URL + "Lists.tgz"
    urlretrieve(split_url, list_path)

    assert list_path.is_file()
    assert tarfile.is_tarfile(list_path)

    # Read the actual official MATLAB split definitions.
    builder = object.__new__(Chars74K)

    train_keys = builder._load_official_split(list_path, "train")
    test_keys = builder._load_official_split(list_path, "test")

    assert len(train_keys) == 310
    assert len(test_keys) == 1674
    assert train_keys.isdisjoint(test_keys)

    # Verify the public Dataset API.
    train = Chars74K(split="train")
    test = Chars74K(split="test")

    assert len(train) == len(train_keys)
    assert len(test) == len(test_keys)
    assert len(Chars74K.LABELS) == 62

    # Verify samples from both splits.
    for dataset in (train, test):
        sample = dataset[0]

        assert sample["image"].mode == "RGB"
        assert sample["image"].width > 0
        assert sample["image"].height > 0
        assert 0 <= sample["label"] < 62
