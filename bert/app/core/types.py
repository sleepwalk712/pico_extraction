from typing import TypedDict

import torch


class EncodingDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
