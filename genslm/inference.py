import os
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.utils import ModelOutput

import genslm

PathLike = Union[str, Path]


class GenSLM(nn.Module):

    __genslm_path = Path(genslm.__file__).parent
    __tokenizer_path = __genslm_path / "tokenizer_files"
    __architecture_path = __genslm_path / "architectures"

    MODELS: Dict[str, Dict[str, str]] = {
        "genslm_25M_patric": {
            "config": str(__architecture_path / "neox" / "neox_25,290,752.json"),
            "tokenizer": str(__tokenizer_path / "codon_wordlevel_69vocab.json"),
            "weights": "patric_25m_epoch01-val_loss_0.57_bias_removed.pt",
            "seq_length": "2048",
        },
        "genslm_250M_patric": {
            "config": str(__architecture_path / "neox" / "neox_244,464,576.json"),
            "tokenizer": str(__tokenizer_path / "codon_wordlevel_69vocab.json"),
            "weights": "patric_250m_epoch00_val_loss_0.48_attention_removed.pt",
            "seq_length": "2048",
        },
        "genslm_2.5B_patric": {
            "config": str(__architecture_path / "neox" / "neox_2,533,931,008.json"),
            "tokenizer": str(__tokenizer_path / "codon_wordlevel_69vocab.json"),
            "weights": "patric_2.5b_epoch00_val_los_0.29_bias_removed.pt",
            "seq_length": "2048",
        },
        "genslm_25B_patric": {
            "config": str(__architecture_path / "neox" / "neox_25,076,188,032.json"),
            "tokenizer": str(__tokenizer_path / "codon_wordlevel_69vocab.json"),
            "weights": "model-epoch00-val_loss0.70-v2.pt",
            "seq_length": "2048",
        },
    }

    def __init__(self, model_id: str, model_cache_dir: PathLike = ".") -> None:
        """GenSLM inference module.

        Parameters
        ----------
        model_id : str
            A model ID corresponding to a pre-trained model. (e.g., genslm_25M_patric)
        model_cache_dir : PathLike, optional
            Directory where model weights have been downloaded to (defaults to current
            working directory). If model weights are not found, then they will be
            downloaded, by default "."

        Raises
        ------
        ValueError
            If model_id is invalid.
        """
        super().__init__()
        self.model_cache_dir = Path(model_cache_dir)
        self.model_info = self.MODELS.get(model_id)
        if self.model_info is None:
            valid_model_ids = list(self.MODELS.keys())
            raise ValueError(
                f"Invalid model_id: {model_id}. Please select one of {valid_model_ids}"
            )

        self._tokenizer = self.configure_tokenizer()
        self.model = self.configure_model()

    @property
    def seq_length(self) -> int:
        assert self.model_info is not None
        return int(self.model_info["seq_length"])

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    def configure_model(self) -> AutoModelForCausalLM:
        assert self.model_info is not None
        base_config = AutoConfig.from_pretrained(self.model_info["config"])
        model = AutoModelForCausalLM.from_config(base_config)

        weight_path = self.model_cache_dir / self.model_info["weights"]
        if not weight_path.exists():
            # TODO: Implement model download
            raise NotImplementedError
        ptl_checkpoint = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ptl_checkpoint["state_dict"], strict=False)
        return model

    def configure_tokenizer(self) -> PreTrainedTokenizerFast:
        assert self.model_info is not None
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_file(self.model_info["tokenizer"])
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any
    ) -> ModelOutput:
        return self.model(
            input_ids, labels=input_ids, attention_mask=attention_mask, **kwargs
        )
