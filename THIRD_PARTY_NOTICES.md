# Third-Party Notices

LightAIR builds on third-party research code. The original copyright and
license notices in the source files are retained. The license texts under
`LICENSES/` apply only to the corresponding third-party components and do
not, by themselves, grant a license to LightAIR-specific contributions.

No project-wide license has currently been declared for the original
LightAIR contributions. Unless the authors publish one, those contributions
remain protected under applicable copyright law.

## X2-VLM

This repository is derived in part from
[X2-VLM](https://github.com/zengyan-97/X2-VLM). Its BSD-style license is
reproduced verbatim in [`LICENSES/X2-VLM-BSD-3-Clause.txt`](LICENSES/X2-VLM-BSD-3-Clause.txt).
X2-VLM-derived code includes the vision-language backbone and supporting
configuration, data, optimization, and scheduling modules, including
`models/xvlm.py` and related integration code.

## CMP / PAB

Training, evaluation, dataset, configuration, and utility code is adapted in
part from [CMP](https://github.com/Shuyu-XJTU/CMP), Copyright (c) 2025
Shuyu-XJTU. Its MIT license is reproduced verbatim in
[`LICENSES/MIT-CMP.txt`](LICENSES/MIT-CMP.txt). Adapted paths include portions
of `configs/`, `dataset/`, `eval.py`, `models/bert.py`, `models/pose.py`,
`optim.py`, `scheduler.py`, `train.py`, and `utils.py`.

## Microsoft Vision Backbones

`models/beit2.py` and `models/swin_transformer.py` include code from Microsoft
BEiT v2 and Swin Transformer releases. Their MIT license is reproduced in
[`LICENSES/MIT-Microsoft.txt`](LICENSES/MIT-Microsoft.txt); the copyright
notices remain in the source files.

## Hugging Face and Google Model Code

`models/bert.py`, `models/xbert.py`, and `models/xroberta.py` include code
from the Google AI Language Team, Hugging Face, and NVIDIA under Apache
License 2.0. The complete license is reproduced in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt), and the attribution
headers remain in each source file.
