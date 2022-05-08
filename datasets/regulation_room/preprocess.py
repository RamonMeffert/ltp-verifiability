"""Transform the raw dataset files into CSV files."""

import pandas as pd
import re
import csv
from typing import Dict, Optional, Union

# Define format
pattern = re.compile("^([une]),(.*)#(.*?)\.(\d+)\.(\d+)")


def line_to_dict(line: str) -> Optional[Dict[str, Union[str, int]]]:
    result = pattern.search(line)
    if result is None:
        print("Encountered non-matching line. The following line will be skipped:\n\t" + line)
        return None
    else:
        return {
            "label": result.group(1),
            "text": result.group(2),
            "rule_name": result.group(3),
            "comment_number": int(result.group(4)),
            "proposition_number": int(result.group(5)),
        }


# Split both files and save to csv
for split in ["test", "train"]:
    with open(split + ".txt", "r") as current:
        result = filter(lambda x: x is not None, list(map(line_to_dict, current.readlines())))
        df = pd.DataFrame(result)
        df.to_csv(split + ".csv", quoting=csv.QUOTE_NONNUMERIC)
