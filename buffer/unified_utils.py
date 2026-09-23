from typing import List, Dict, Any
import torch
import re
from fastNLP import logger

print("DEBUG: Loaded Unified Utils V7 (Final Llama Fix: Using Reserved Token)")

def get_token_config(tokenizer):
    model_type = tokenizer.name_or_path.lower()

    if 'llama' in model_type:
        return {
            'type': 'llama',
            'start_token': '<|reserved_special_token_0|>',
            'end_token': '<|reserved_special_token_1|>',
            'thought_token': '<|reserved_special_token_2|>',
        }

    else:
        return {
            'type': 'qwen',
            'start_token': '<|box_start|>',
            'end_token': '<|box_end|>',
            'thought_token': '<|endoftext|>',
        }

def get_soft_thoughts(num_thought_tokens, tokenizer):
    cfg = get_token_config(tokenizer)
    return f"{cfg['start_token']}{cfg['thought_token'] * num_thought_tokens}{cfg['end_token']}"

def _split_question_options(instance):
    raw_q = instance['question']
    options_str = ""
    if "Options:" in raw_q:
        parts = raw_q.split("Options:")
        clean_q = parts[0].strip()
        options_str = "Options:" + parts[1]
    else:
        clean_q = raw_q.strip()
        if 'options' in instance:
            options_str = "Options:\n" + "\n".join(instance['options'])
    return clean_q, options_str

def _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len):
    input_messages = [{'role': 'user', 'content': input_content}]

    input_ids = tokenizer.apply_chat_template(input_messages, tokenize=True)

    if max_len > 0 and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]

    attention_mask = [1] * len(input_ids)

    thought_start_idx = 0
    thought_end_idx = 0

    if num_thought_tokens > 0:
        cfg = get_token_config(tokenizer)
        start_token = cfg['start_token']
        end_token = cfg['end_token']

        start_tokens_encoded = tokenizer.encode(start_token, add_special_tokens=False)
        end_tokens_encoded = tokenizer.encode(end_token, add_special_tokens=False)

        start_id = start_tokens_encoded[-1] if start_tokens_encoded else None
        end_id = end_tokens_encoded[-1] if end_tokens_encoded else None

        input_ids_tensor = torch.tensor(input_ids)

        if start_id is not None and end_id is not None:
            start_positions = (input_ids_tensor == start_id).nonzero(as_tuple=True)[0]
            end_positions = (input_ids_tensor == end_id).nonzero(as_tuple=True)[0]

            if len(start_positions) > 0 and len(end_positions) > 0:
                thought_start_idx = start_positions[-1].item() + 1
                thought_end_idx = end_positions[-1].item()
            else:
                logger.warning(f"Soft Token markers ({start_token}, {end_token}) not found in IDs.")
        else:
            logger.warning(f"Could not encode Soft Token markers ({start_token}, {end_token}).")

    labels = [-100] * len(input_ids)

    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'thought_index': [thought_start_idx, thought_end_idx, 0, 0],
    }

    if 'answer' in instance: inputs['answer'] = instance['answer']
    elif 'correct' in instance: inputs['answer'] = instance['correct']

    if device is not None:
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device) if isinstance(v, List) else v
            for k, v in inputs.items()
        }

    return inputs

def pre_process_gsm8k_unified(instance, tokenizer, num_thought_tokens=4, device=None, split='train', max_len=-1, **kwargs):
    question = instance['question']
    input_template = (
        f'Solve the following math problem efficiently and clearly:\n'
        f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal equation.\n'
        f'- For complex problems (3 steps or more):\n'
        f'Use this step-by-step format:\n\n'
        f'## Step 1: [Brief calculations]\n'
        f'## Step 2: [Brief calculations]\n'
        f'...\n'
        f'Regardless of the approach, always conclude with:\n'
        f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
        f'Where [answer] is just the final number or expression that solves the problem.\n\n'
        f'Problem: {question}'
    )
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens, tokenizer)
        input_content = f'{input_template}{soft_thoughts}\n\n'
    else:
        input_content = f'{input_template}\n\n'
    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)

def pre_process_strategy_qa_unified(instance, tokenizer, num_thought_tokens=2, device=None, split='train', max_len=-1, **kwargs):
    question = instance['question']
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens, tokenizer)
        input_content = (
            f'You are required to answer the following question with `Yes` or `No`: {question}{soft_thoughts}\n\n'
            f'Therefore, the final answer is `Yes` or `No`?'
        )
    else:
        input_content = (
            f'You are required to answer the following question with `Yes` or `No`: {question}\n\n'
            f'Therefore, the final answer is `Yes` or `No`?'
        )
    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)

def pre_process_aqua_unified(instance, tokenizer, num_thought_tokens=2, device=None, split='train', max_len=-1, **kwargs):
    clean_q, options_str = _split_question_options(instance)
    input_template_head = (
        f'You are required to solve the following math multiple choices questions.\n'
        f'In the multiple choices question, there are five options: A, B, C, D, and E, respectively.\n'
        f'The correct answer that solves the problem is one of these options.\n'
        f'Your job is to solve the problem and find the correct option.\n'
        f'Here are the instructions for solving the problem efficiently and clearly:\n'
        f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal equation.\n'
        f'- For complex problems (3 steps or more):\n'
        f'Use this step-by-step format:\n\n'
        f'## Step 1: [Brief calculations]\n'
        f'## Step 2: [Brief calculations]\n'
        f'...\n'
        f'Regardless of the approach, always conclude with the following sentence:\n'
        f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
        f'Where [answer] is the option from A, B, C, D, and E.\n'
        f'Only one letter from A to E is accepted in the answer span.\n\n'
        f'Problem: {clean_q}'
    )
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens, tokenizer)
        input_content = f'{input_template_head}{soft_thoughts}'
    else:
        input_content = f'{input_template_head}'
    if options_str:
        input_content += '\n' + options_str
    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)

def pre_process_du_unified(instance, tokenizer, num_thought_tokens=2, device=None, split='train', max_len=-1, **kwargs):
    clean_q, options_str = _split_question_options(instance)
    input_template_head = (
        f'You are required to solve the following math multiple choices questions.\n'
        f'In the multiple choices question, there are five options: A, B, C, D, E, and F, respectively.\n'
        f'The correct answer that solves the problem is one of these options.\n'
        f'Your job is to solve the problem and find the correct option.\n'
        f'Here are the instructions for solving the problem efficiently and clearly:\n'
        f'- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal description.\n'
        f'- For complex problems (3 steps or more):\n'
        f'Use this step-by-step format:\n\n'
        f'## Step 1: [Brief reasoning step]\n'
        f'## Step 2: [Brief reasoning step]\n'
        f'...\n'
        f'Regardless of the approach, always conclude with the following sentence:\n'
        f'Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n'
        f'Where [answer] is the option from A, B, C, D, E, and F.\n'
        f'Only one letter from A to F is accepted in the answer span.\n\n'
        f'Problem: {clean_q}'
    )
    if num_thought_tokens > 0:
        soft_thoughts = get_soft_thoughts(num_thought_tokens, tokenizer)
        input_content = f'{input_template_head}{soft_thoughts}'
    else:
        input_content = f'{input_template_head}'
    if options_str:
        input_content += '\n' + options_str
    return _pack_inputs(input_content, instance, tokenizer, num_thought_tokens, device, split, max_len)
