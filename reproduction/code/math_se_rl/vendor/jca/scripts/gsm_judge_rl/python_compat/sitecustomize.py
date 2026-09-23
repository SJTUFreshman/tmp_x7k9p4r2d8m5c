import os

from transformers import PreTrainedTokenizerBase


if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )


if os.environ.get("JCA_PATCH_VLLM_TRANSFORMERS_LM_HEAD") == "1":
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.transformers import TransformersModel
    from vllm.model_executor.models.utils import (is_pp_missing_parameter,
                                                   maybe_prefix)

    def _load_weights_with_root_level_heads(self, weights):
        """Preserve root-level heads in vLLM 0.8.2's Transformers fallback."""
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        base_prefix = self.model.base_model_prefix
        for name, loaded_weight in weights:
            target_name = name
            if (target_name not in params_dict
                    and not target_name.startswith(base_prefix)):
                target_name = maybe_prefix(base_prefix, target_name)
            if is_pp_missing_parameter(target_name, self):
                continue
            if target_name in params_dict:
                param = params_dict[target_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(target_name)
        return loaded_params

    TransformersModel.load_weights = _load_weights_with_root_level_heads
