
import os
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache
from fastNLP import logger
from feedback import match_feedback_norm
from sts import SoftTokenSelector
try:
    from norm_monitor import NormMonitor
except Exception:
    NormMonitor = None


class UnifiedSoftCoT(nn.Module):

    def __init__(
        self,
        model_id,
        num_thought_tokens=4,
        tune_base_model=False,
        path_to_projection_module=None,
        device_map='auto',
        projection_type='linear',
        sts_bank_size=64,
        sts_temperature=1.0,
        sts_bank_norm_scale=1.0,
        sts_init_seed=42,
        **kwargs,
    ):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            _fast_init=False,
        )

        self.config = AutoConfig.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.num_thought_tokens = num_thought_tokens
        self.tune_base_model = tune_base_model
        self.projection_type = projection_type
        self.feedback_norm = None
        self.last_feedback_stats = []
        self.last_projection_stats = {}

        if num_thought_tokens > 0:
            self.dropout = nn.Dropout(0.0)
            if projection_type == 'linear':
                self.projections = nn.ModuleList([
                    nn.Linear(
                        self.model.config.hidden_size,
                        self.model.config.hidden_size,
                        dtype=torch.bfloat16,
                    ) for _ in range(num_thought_tokens)
                ])
                with torch.no_grad():
                    for proj in self.projections:
                        proj.weight.data.zero_()
                        proj.bias.data.zero_()
            elif projection_type == 'sts':
                embedding_weight = self.model.get_input_embeddings().weight
                self.projections = nn.ModuleList([
                    SoftTokenSelector(
                        hidden_size=self.model.config.hidden_size,
                        bank_size=sts_bank_size,
                        temperature=sts_temperature,
                        embedding_weight=embedding_weight,
                        excluded_token_ids=self.tokenizer.all_special_ids,
                        bank_norm_scale=sts_bank_norm_scale,
                        init_seed=sts_init_seed + i,
                    ) for i in range(num_thought_tokens)
                ])
            else:
                raise ValueError(f'Unknown projection_type: {projection_type}')
        else:
            self.projections = nn.ModuleList([])
            self.dropout = nn.Dropout(0.0)

        for n, p in self.model.named_parameters():
            p.requires_grad = tune_base_model

        if path_to_projection_module is not None and path_to_projection_module not in ['None']:
            state_dict = torch.load(path_to_projection_module, map_location='cpu', weights_only=True)
            self.projections.load_state_dict(state_dict)
            logger.info(f'Loaded weights from `{path_to_projection_module}`.')

        self.projections.to(self.model.device)

    @property
    def device(self):
        return self.model.device

    def scale_feedback(self, vectors):
        return match_feedback_norm(vectors, self.feedback_norm)

    def project_hidden_states(self, hidden_states, norm_monitor=None, norm_meta=None):
        projected_list = []
        stats_by_name = {
            'query_norm': [],
            'bank_norm': [],
            'output_norm': [],
            'attention_entropy': [],
            'attention_effective_tokens': [],
            'attention_top1': [],
            'attention_argmax': [],
        }
        for i in range(self.num_thought_tokens):
            token_vec = self.dropout(hidden_states[i])
            proj_vec = self.projections[i](token_vec)
            projected_list.append(proj_vec)
            if self.projection_type != 'sts':
                continue

            module = self.projections[i]
            stats = module.detached_stats()
            for key in stats_by_name:
                stats_by_name[key].append(stats[key])
            if norm_monitor is not None:
                meta = dict(norm_meta or {})
                meta['thought_position'] = i
                norm_monitor.record_tensor('sts_query', module.last_query, meta)
                norm_monitor.record_tensor('sts_soft_token_bank', module.soft_token_bank, meta)
                norm_monitor.record_tensor('sts_output', module.last_output, meta)
                norm_monitor.record_tensor(
                    'sts_attention_entropy',
                    module.last_attention.new_tensor(stats['attention_entropy']),
                    meta,
                )
                norm_monitor.record_tensor(
                    'sts_attention_top1',
                    module.last_attention.new_tensor(stats['attention_top1']),
                    meta,
                )

        self.last_projection_stats = stats_by_name if self.projection_type == 'sts' else {}
        return torch.stack(projected_list)

    def save_pretrained(self, save_model_dir_root: str, **kwargs):
        os.makedirs(save_model_dir_root, exist_ok=True)
        projection_file = os.path.join(save_model_dir_root, 'projection.bin')
        torch.save(self.projections.state_dict(), projection_file)
        logger.info(f'Saved projection module to `{projection_file}`.')

    def get_inputs_embeds_for_unified_model(
        self,
        input_ids,
        attention_mask,
        thought_index,
        print_index=False,
        norm_monitor=None,
        norm_meta=None,
    ):
        batch_size, seq_len = input_ids.size()
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        self.last_feedback_stats = []
        if norm_monitor is not None:
            norm_monitor.record_tensor('llm_input_token_embedding.raw_prompt', inputs_embeds, norm_meta)

        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]

        for b in range(batch_size):
            s_idx = thought_index[b, 0].item()
            e_idx = thought_index[b, 1].item()

            soft_token_base = inputs_embeds[b, s_idx:e_idx]
            soft_thoughts_raw = hidden_states[b, s_idx:e_idx]
            meta = dict(norm_meta or {})
            meta.update({'batch_item': b, 'thought_start': s_idx, 'thought_end': e_idx})
            if norm_monitor is not None:
                norm_monitor.record_tensor('soft_token_embedding.before_projection', soft_token_base, meta)
                norm_monitor.record_tensor('projection_input.hidden_state', soft_thoughts_raw, meta)

            if soft_thoughts_raw.size(0) != self.num_thought_tokens:
                continue

            projected_thoughts = self.project_hidden_states(
                soft_thoughts_raw, norm_monitor=norm_monitor, norm_meta=meta,
            )
            feedback = self.scale_feedback(projected_thoughts)
            if norm_monitor is not None:
                norm_monitor.record_tensor('projection_output.raw', projected_thoughts, meta)
                norm_monitor.record_tensor('projection_output.scaled_feedback', feedback, meta)
                norm_monitor.record_cosine('projection_raw_to_scaled_feedback.cosine', projected_thoughts, feedback, meta)
            inputs_embeds[b, s_idx:e_idx] = feedback
            feedback_stats = {
                'hidden_norm': soft_thoughts_raw.detach().float().norm(dim=-1).tolist(),
                'projected_norm': projected_thoughts.detach().float().norm(dim=-1).tolist(),
                'feedback_norm': feedback.detach().float().norm(dim=-1).tolist(),
                'direction_cosine': torch.nn.functional.cosine_similarity(
                    projected_thoughts.detach().float(), feedback.detach().float(), dim=-1,
                ).tolist(),
            }
            feedback_stats.update(self.last_projection_stats)
            self.last_feedback_stats.append(feedback_stats)

            if print_index:
                logger.info(f'Processed soft thoughts at index {s_idx}-{e_idx}')

        if norm_monitor is not None:
            norm_monitor.record_tensor('llm_input_token_embedding.final_prompt', inputs_embeds, norm_meta)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        thought_index: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        print_index=False,
        norm_monitor=None,
        norm_meta=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.size()

        if seq_len > 1 and self.num_thought_tokens > 0:
            inputs_embeds = self.get_inputs_embeds_for_unified_model(
                input_ids, attention_mask, thought_index, print_index,
                norm_monitor=norm_monitor, norm_meta=norm_meta,
            )
            outputs = self.model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        return outputs
