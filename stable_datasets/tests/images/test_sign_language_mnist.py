import csv
import io
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from stable_datasets.images.sign_language_mnist import SignLanguageMNIST


# --------------------------------------------------
# Test helpers
# --------------------------------------------------

HEADER = ["label"] + [f"pixel{i}" for i in range(1, 785)]


def make_csv(labels, header=None, pixels=None):
    """Create a small Sign Language MNIST CSV."""

    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(HEADER if header is None else header)

    for label in labels:
        row_pixels = [128] * 784 if pixels is None else pixels
        writer.writerow([label] + row_pixels)

    return text.getvalue()


def make_test_zip(path, labels, split="train"):
    """Create a ZIP using the official Kaggle CSV filename."""

    filename = SignLanguageMNIST.CSV_FILES[split]

    with ZipFile(path, "w") as archive:
        archive.writestr(filename, make_csv(labels))


def make_custom_zip(path, files):
    """Create a ZIP containing the specified CSV files."""

    with ZipFile(path, "w") as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)


# --------------------------------------------------
# Offline unit tests
# --------------------------------------------------


def test_generate_examples(tmp_path):
    zip_path = tmp_path / "sample.zip"
    make_test_zip(zip_path, [0, 10, 24])

    builder = object.__new__(SignLanguageMNIST)
    samples = list(builder._generate_examples(zip_path, "train"))

    assert len(samples) == 3
    assert [sample["label"] for _, sample in samples] == [0, 9, 23]

    for _, sample in samples:
        assert isinstance(sample["image"], Image.Image)
        assert sample["image"].size == (28, 28)
        assert sample["image"].mode == "L"


def test_all_label_mappings():
    """Check all 24 ASL classes and excluded letters."""

    mapping = SignLanguageMNIST.LABEL_MAP
    expected_original = set(range(26)) - {9, 25}

    assert set(mapping.keys()) == expected_original
    assert set(mapping.values()) == set(range(24))

    assert len(SignLanguageMNIST.LETTERS) == 24
    assert "J" not in SignLanguageMNIST.LETTERS
    assert "Z" not in SignLanguageMNIST.LETTERS

    assert mapping[0] == 0
    assert mapping[10] == 9
    assert mapping[24] == 23


def test_train_test_csv_selection(tmp_path):
    """Select the correct CSV when both splits are in one ZIP."""

    zip_path = tmp_path / "both_splits.zip"

    make_custom_zip(
        zip_path,
        {
            "sign_mnist_train.csv": make_csv([0, 10]),
            "sign_mnist_test.csv": make_csv([24]),
            "sign_mnist_train/sign_mnist_train.csv": make_csv([5]),
            "sign_mnist_test/sign_mnist_test.csv": make_csv([6]),
        },
    )

    builder = object.__new__(SignLanguageMNIST)

    train = list(builder._generate_examples(zip_path, "train"))
    test = list(builder._generate_examples(zip_path, "test"))

    assert len(train) == 2
    assert len(test) == 1

    assert [sample["label"] for _, sample in train] == [0, 9]
    assert [sample["label"] for _, sample in test] == [23]


def test_invalid_label(tmp_path):
    zip_path = tmp_path / "invalid.zip"
    make_test_zip(zip_path, [9])

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Invalid label"):
        list(builder._generate_examples(zip_path, "train"))


def test_invalid_csv_columns(tmp_path):
    """Reject CSV rows with missing pixel values."""

    zip_path = tmp_path / "bad_columns.zip"

    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(HEADER)
    writer.writerow([0] + [128] * 783)

    make_custom_zip(
        zip_path,
        {"sign_mnist_train.csv": text.getvalue()},
    )

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Invalid row"):
        list(builder._generate_examples(zip_path, "train"))


def test_invalid_csv_header(tmp_path):
    """Reject an incorrect CSV header."""

    zip_path = tmp_path / "bad_header.zip"

    wrong_header = ["class"] + [f"pixel{i}" for i in range(1, 785)]

    make_custom_zip(
        zip_path,
        {"sign_mnist_train.csv": make_csv([0], header=wrong_header)},
    )

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Unexpected CSV header"):
        list(builder._generate_examples(zip_path, "train"))


def test_missing_csv(tmp_path):
    """Reject ZIP archives without the requested split."""

    zip_path = tmp_path / "missing.zip"
    make_test_zip(zip_path, [0], split="train")

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Missing official CSV"):
        list(builder._generate_examples(zip_path, "test"))


def test_invalid_split(tmp_path):
    zip_path = tmp_path / "sample.zip"
    make_test_zip(zip_path, [0])

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match="Unsupported split"):
        list(builder._generate_examples(zip_path, "validation"))


def test_pixel_values(tmp_path):
    """Verify pixel values survive CSV-to-image conversion."""

    zip_path = tmp_path / "pixels.zip"
    pixels = [0, 127, 255] + [128] * 781

    make_custom_zip(
        zip_path,
        {"sign_mnist_train.csv": make_csv([0], pixels=pixels)},
    )

    builder = object.__new__(SignLanguageMNIST)

    samples = list(builder._generate_examples(zip_path, "train"))
    image = samples[0][1]["image"]

    assert image.size == (28, 28)
    assert image.mode == "L"
    assert np.asarray(image).flatten().tolist() == pixels


@pytest.mark.parametrize(
    ("invalid_value", "expected_error"),
    [
        (256, "Pixel out of range"),
        (-1, "Pixel out of range"),
        ("nan", "Invalid pixel value"),
        (1.5, "Non-integer pixel value"),
    ],
)
def test_invalid_pixel_value(tmp_path, invalid_value, expected_error):
    """Reject pixels outside the valid uint8 range."""

    zip_path = tmp_path / "invalid_pixels.zip"

    pixels = [invalid_value] + [128] * 783

    make_custom_zip(
        zip_path,
        {"sign_mnist_train.csv": make_csv([0], pixels=pixels)},
    )

    builder = object.__new__(SignLanguageMNIST)

    with pytest.raises(ValueError, match=expected_error):
        list(builder._generate_examples(zip_path, "train"))


# --------------------------------------------------
# Real-data integration test
# --------------------------------------------------


def test_real_sign_language_mnist(tmp_path):
    """Validate official Kaggle ZIP and the public dataset API."""

    # Download a fresh official ZIP, independently of builder cache.
    zip_path = tmp_path / "sign_language_mnist.zip"
    urlretrieve(SignLanguageMNIST.KAGGLE_URL, zip_path)

    expected_counts = {
        "train": 27455,
        "test": 7172,
    }

    expected_original_labels = set(range(26)) - {9, 25}

    with ZipFile(zip_path) as archive:
        assert archive.testzip() is None

        for split, expected_count in expected_counts.items():
            filename = SignLanguageMNIST.CSV_FILES[split]
            assert filename in archive.namelist()

            with archive.open(filename) as file:
                text = io.TextIOWrapper(file, encoding="utf-8-sig", newline="")
                reader = csv.reader(text)

                assert next(reader) == HEADER

                count = 0
                labels = set()

                for row in reader:
                    assert len(row) == 785

                    label = int(row[0])
                    assert label in expected_original_labels

                    labels.add(label)
                    count += 1

                assert count == expected_count
                assert labels == expected_original_labels

    # Verify full sample parsing from the official archive.
    builder = object.__new__(SignLanguageMNIST)

    for split, expected_count in expected_counts.items():
        count = 0
        mapped_labels = set()

        for _, sample in builder._generate_examples(zip_path, split):
            assert sample["image"].size == (28, 28)
            assert sample["image"].mode == "L"
            assert 0 <= sample["label"] < 24

            mapped_labels.add(sample["label"])
            count += 1

        assert count == expected_count
        assert mapped_labels == set(range(24))

    # Verify the public Dataset API.
    train = SignLanguageMNIST(split="train")
    test = SignLanguageMNIST(split="test")

    assert len(train) == 27455
    assert len(test) == 7172

    for dataset in (train, test):
        sample = dataset[0]

        assert sample["image"].size == (28, 28)
        assert sample["image"].mode == "L"
        assert 0 <= sample["label"] < 24
