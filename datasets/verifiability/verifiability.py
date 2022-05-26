# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A combination of two datasets on verifiability."""

import os
import csv

import datasets

_CITATION = """
"""

# You can copy an official description
_DESCRIPTION = """
"""

_HOMEPAGE = ""

_LICENSE = "UNKNOWN"

_URLS = {
    "train": "./train.csv",
    "test": "./test.csv",
    "validation": "./validation.csv",
}


class RegulationRoom(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="train",
                               version=VERSION,
                               description="Training set"),
        datasets.BuilderConfig(name="test",
                               version=VERSION,
                               description="Test set"),
        datasets.BuilderConfig(name="validation",
                               version=VERSION,
                               description="Validation set"),
    ]

    DEFAULT_CONFIG_NAME = "train"

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features({
                "sentence": datasets.Value("string"),
                "thread_id": datasets.Value("string"),
                "comment_id": datasets.Value("string"),
                "sentence_index": datasets.Value("int32"),
                "source": datasets.Value("string"),
                "verifiability": datasets.ClassLabel(
                    num_classes=3,
                    names=['verifiable', 'unverifiable', 'nonargument']),
                "experiential": datasets.ClassLabel(
                    num_classes=3,
                    names=['True', 'False', ''])
            }),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(downloaded_files["train"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(downloaded_files["test"]),
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(downloaded_files["validation"]),
                    "split": "validation"
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for key, row in enumerate(reader):
                # Yields examples as (key, example) tuples
                yield key, {
                    "sentence": row["sentence"],
                    "thread_id": row["thread_id"],
                    "comment_id": row["comment_id"],
                    "sentence_index": row["sentence_index"],
                    "source": row["source"],
                    "verifiability": row["verifiability"],
                    "experiential": row["experiential"],
                }
