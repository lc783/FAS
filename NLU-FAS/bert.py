from transformers import BertModel, PretrainedConfig, BertForMaskedLM, AutoModelForMaskedLM
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
import json
import os
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, \
    MaskedLMOutput
from transformers import AutoModelForSequenceClassification
from transformers.utils import (
    CONFIG_NAME,
    cached_file,
    copy_func,
    extract_commit_hash,
    find_adapter_config_file,
    is_peft_available,
    logging,
    requires_backends, add_start_docstrings_to_model_forward,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
import warnings
import copy
from layer import *
# from layer2 import *

from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
import warnings
import copy

class CustomBertModel(BertModel):
    def __init__(self, config, T = 0, add_pooling_layer = True):
        super(CustomBertModel, self).__init__(config, add_pooling_layer)
        self.T = T
        for i in range(12):
            self.encoder.layer[i].intermediate.intermediate_act_fn = IF()

        self.merge = MergeTemporalDim(0)


    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = T
        return

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = L
        return

    # def forward(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         token_type_ids: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         encoder_hidden_states: Optional[torch.Tensor] = None,
    #         encoder_attention_mask: Optional[torch.Tensor] = None,
    #         past_key_values: Optional[List[torch.FloatTensor]] = None,
    #         use_cache: Optional[bool] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    #     r"""
    #     encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
    #         Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
    #         the model is configured as a decoder.
    #     encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #         Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
    #         the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    #
    #         - 1 for tokens that are **not masked**,
    #         - 0 for tokens that are **masked**.
    #     past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
    #         Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
    #
    #         If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
    #         don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
    #         `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    #     use_cache (`bool`, *optional*):
    #         If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
    #         `past_key_values`).
    #     """
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     if self.config.is_decoder:
    #         use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     else:
    #         use_cache = False
    #
    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    #         input_shape = input_ids.size()
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")
    #
    #     batch_size, seq_length = input_shape
    #     device = input_ids.device if input_ids is not None else inputs_embeds.device
    #
    #     # past_key_values_length
    #     past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
    #
    #     if attention_mask is None:
    #         attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
    #
    #     if token_type_ids is None:
    #         if hasattr(self.embeddings, "token_type_ids"):
    #             buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
    #             buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
    #             token_type_ids = buffered_token_type_ids_expanded
    #         else:
    #             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    #
    #     # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    #     # ourselves in which case we just need to make it broadcastable to all heads.
    #     extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
    #
    #     # If a 2D or 3D attention mask is provided for the cross-attention
    #     # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    #     if self.config.is_decoder and encoder_hidden_states is not None:
    #         encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    #         encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    #         if encoder_attention_mask is None:
    #             encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
    #         encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    #     else:
    #         encoder_extended_attention_mask = None
    #
    #     # Prepare head mask if needed
    #     # 1.0 in head_mask indicate we keep the head
    #     # attention_probs has shape bsz x n_heads x N x N
    #     # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    #     # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    #     head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    #
    #     embedding_output = self.embeddings(
    #         input_ids=input_ids,
    #         position_ids=position_ids,
    #         token_type_ids=token_type_ids,
    #         inputs_embeds=inputs_embeds,
    #         past_key_values_length=past_key_values_length,
    #     )
    #
    #     if self.T > 0:
    #         embedding_output = add_dimention(embedding_output, self.T)    # torch.Size([32, 128, 768])
    #         embedding_output = self.merge(embedding_output)   # torch.Size([1600, 3, 32, 32])
    #
    #         extended_attention_mask = add_dimention_mask(extended_attention_mask, self.T)
    #         extended_attention_mask = self.merge(extended_attention_mask)
    #
    #     encoder_outputs = self.encoder(
    #         embedding_output,
    #         attention_mask=extended_attention_mask,
    #         head_mask=head_mask,
    #         encoder_hidden_states=encoder_hidden_states,
    #         encoder_attention_mask=encoder_extended_attention_mask,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #     sequence_output = encoder_outputs[0]
    #     pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
    #
    #     if not return_dict:
    #         return (sequence_output, pooled_output) + encoder_outputs[1:]
    #
    #     return BaseModelOutputWithPoolingAndCrossAttentions(
    #         last_hidden_state=sequence_output,
    #         pooler_output=pooled_output,
    #         past_key_values=encoder_outputs.past_key_values,
    #         hidden_states=encoder_outputs.hidden_states,
    #         attentions=encoder_outputs.attentions,
    #         cross_attentions=encoder_outputs.cross_attentions,
    #     )


from transformers import BertForSequenceClassification, BertModel, BertConfig, AutoConfig



class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super(CustomBertForMaskedLM, self).__init__(config)
        self.bert = CustomBertModel(config, T=0, add_pooling_layer=False)
        # self.cls = BertOnlyMLMHead(config)
        self.cls.predictions.transform.transform_act_fn = IF()

    def set_T(self, T):
        self.bert.set_T(T)
        self.cls.predictions.transform.transform_act_fn.T = T
        return

    def set_L(self, L):
        self.cls.predictions.transform.transform_act_fn.L = L
        self.bert.set_L(L)
        return

    # def forward(
    #         self,
    #         input_ids: Optional[torch.Tensor] = None,
    #         attention_mask: Optional[torch.Tensor] = None,
    #         token_type_ids: Optional[torch.Tensor] = None,
    #         position_ids: Optional[torch.Tensor] = None,
    #         head_mask: Optional[torch.Tensor] = None,
    #         inputs_embeds: Optional[torch.Tensor] = None,
    #         encoder_hidden_states: Optional[torch.Tensor] = None,
    #         encoder_attention_mask: Optional[torch.Tensor] = None,
    #         labels: Optional[torch.Tensor] = None,
    #         output_attentions: Optional[bool] = None,
    #         output_hidden_states: Optional[bool] = None,
    #         return_dict: Optional[bool] = None,
    # ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
    #         config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
    #         `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    #     """
    #
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    #
    #     outputs = self.bert(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         encoder_hidden_states=encoder_hidden_states,
    #         encoder_attention_mask=encoder_attention_mask,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #     sequence_output = outputs[0]
    #     prediction_scores = self.cls(sequence_output)
    #
    #     if self.T > 0:
    #         prediction_scores = self.expand(prediction_scores)
    #         prediction_scores = prediction_scores.mean(0)
    #
    #     masked_lm_loss = None
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()  # -100 index = padding token
    #         masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    #
    #     if not return_dict:
    #         output = (prediction_scores,) + outputs[2:]
    #         return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    #
    #     return MaskedLMOutput(
    #         loss=masked_lm_loss,
    #         logits=prediction_scores,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )



def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]

class SNNAutoModelForMaskedLM(AutoModelForMaskedLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)
        use_auth_token = hub_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            hub_kwargs["token"] = token

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    **hub_kwargs,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            if adapter_kwargs is None:
                adapter_kwargs = {}
                if token is not None:
                    adapter_kwargs["token"] = token

            maybe_adapter_path = find_adapter_config_file(
                pretrained_model_name_or_path, _commit_hash=commit_hash, **adapter_kwargs
            )

            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)

                    adapter_kwargs["_adapter_model_path"] = pretrained_model_name_or_path
                    pretrained_model_name_or_path = adapter_config["base_model_name_or_path"]

        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype", None) == "auto":
                _ = kwargs.pop("torch_dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config", None) is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                code_revision=code_revision,
                _commit_hash=commit_hash,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"
            if kwargs_orig.get("quantization_config", None) is not None:
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        # Set the adapter kwargs
        kwargs["adapter_kwargs"] = adapter_kwargs

        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
            )
            _ = hub_kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return CustomBertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )



