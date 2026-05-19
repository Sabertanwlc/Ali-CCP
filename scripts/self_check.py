from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ali_ccp_format import parse_common_line, parse_skeleton_line


def check_parser() -> None:
    skeleton_line = "1,1,0,abc,2,210\x02x1\x031.0\x01211\x02x2\x030.5\n"
    skeleton = parse_skeleton_line(skeleton_line)
    assert skeleton.sample_id == "1"
    assert skeleton.click == 1
    assert skeleton.conversion == 0
    assert skeleton.common_feature_id == "abc"
    assert len(skeleton.ad_features) == 2
    assert skeleton.ad_features[0].token == "210:x1"
    assert skeleton.ad_features[1].value == 0.5

    common_line = "abc,1,101\x02u1\x031.0\n"
    common = parse_common_line(common_line)
    assert common.common_feature_id == "abc"
    assert common.common_features[0].token == "101:u1"


def main() -> None:
    check_parser()
    print("self_check: ok")


if __name__ == "__main__":
    main()
