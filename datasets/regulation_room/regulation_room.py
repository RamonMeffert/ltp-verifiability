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
"""The data set used by Park and Cardie (2014) processed for use with HuggingFace."""

import os
import csv

import datasets

_CITATION = """
@InProceedings{park-cardie:2014:W14-21,
  author    = {Park, Joonsuk  and  Cardie, Claire},
  title     = {Identifying Appropriate Support for Propositions in Online User Comments},
  booktitle = {Proceedings of the First Workshop on Argumentation Mining},
  month     = {June},
  year      = {2014},
  address   = {Baltimore, Maryland},
  publisher = {Association for Computational Linguistics},
  pages     = {29--38},
  url       = {http://www.aclweb.org/anthology/W/W14/W14-2105}
}
"""

# You can copy an official description
_DESCRIPTION = """
We have collected and manually annotated sentences and (independent) clauses
from user comments extracted from an eRulemaking website, Regulation Room. For
our research, we collected and manually annotated 9,476 propositions from 1,047
user comments from two recent rules: Airline Passenger Rights (serving peanuts
on the plane, tarmac delay contingency plan, oversales of tickets, baggage fees
and other airline traveller rights) and Home Mortgage Consumer Protection (loss
mitigation, accounting error resolution, etc.).
"""

_HOMEPAGE = "https://facultystaff.richmond.edu/~jpark/"

_LICENSE = "UNKNOWN"

_URLS = {
    "train": "./train.csv",
    "test": "./test.csv",
}


class RegulationRoom(datasets.GeneratorBasedBuilder):
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
                "label":
                datasets.Value("string"),
                "text":
                datasets.Value("string"),
                "rule_name":
                datasets.Value("string"),
                "comment_number":
                datasets.Value("int32"),
                "proposition_number":
                datasets.Value("int32")
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
                    "label": row["label"],
                    "text": row["text"],
                    "rule_name": row["rule_name"],
                    "comment_number": row["comment_number"],
                    "proposition_number": row["proposition_number"]
                }
