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
"""A Reddit Change My View data set processed for use with HuggingFace."""

import os
import csv

import datasets

_CITATION = """
???
"""

# You can copy an official description
_DESCRIPTION = """
???
"""

_HOMEPAGE = ""

_LICENSE = "UNKNOWN"

_URLS = {
    "train": "./train_set.csv",
    "test": "./test_set.csv",
}


class ChangeMyView(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="train",
                               version=VERSION,
                               description="Author-defined training split"),
        datasets.BuilderConfig(name="test",
                               version=VERSION,
                               description="Author-defined test split"),
    ]

    DEFAULT_CONFIG_NAME = "train"

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features({
                "thread_id": datasets.Value("string"),
                "comment_id": datasets.Value("string"),
                "sentence": datasets.Value("string"),
                "comment": datasets.Value("string"),
                "id": datasets.Value("int32"),
                "verif": datasets.ClassLabel(
                    num_classes=4, names=['Verif', 'UnVerif', 'NonArg', '']),
                "personal": datasets.ClassLabel(
                    num_classes=3, names=['Pers', 'NonPers', '']),
                "difficulty": datasets.Value("string"),
                "annotator": datasets.Value("int32"),
                "annotation_id": datasets.Value("string"),
                "created_at": datasets.Value("string"),
                "updated_at": datasets.Value("string"),
                "lead_time": datasets.Value("float")
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
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for key, row in enumerate(reader):
                # Yields examples as (key, example) tuples
                yield key, {
                    "thread_id": row["thread_id"],
                    "comment_id": row["comment_id"],
                    "sentence": row["sentence"],
                    "comment": row["comment"],
                    "id": row["id"],
                    "verif": row["verif"],
                    "personal": row["personal"],
                    "difficulty": row["difficulty"],
                    "annotator": row["annotator"],
                    "annotation_id": row["annotation_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "lead_time": row["lead_time"]
                }
