import re
import argparse
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, GenerationConfig
from fastNLP import logger

sys.path.append(os.getcwd())

from unified_llm_model import UnifiedSoftCoT
from unified_utils import (
    pre_process_gsm8k_unified,
    pre_process_strategy_qa_unified,
    pre_process_aqua_unified,
    pre_process_du_unified
)
from data_loader import GSM8KLoader, StrategyQALoader, AugASDivLoader, AQuALoader, DULoader
from feedback import configure_feedback
from norm_monitor import NormMonitor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--params_file_name', type=str, default=None)
    parser.add_argument('--num_thought_tokens', type=int, default=3)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    parser.add_argument('--task_name', type=str, default='gsm8k', choices=['gsm8k', 'asdiv-aug', 'strategyqa', 'aqua', 'du'])
    parser.add_argument('--print_input', action='store_true', default=False)
    parser.add_argument('--print_response', action='store_true', default=False)

    parser.add_argument('--test_k', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--tune_base_model', action='store_true', default=False)

    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'dev', 'test'], help='Which split to evaluate on')
    parser.add_argument('--feedback_mode', choices=['vanilla', 'scale_match'], default=None,
                        help='Defaults to checkpoint run_config.json, or vanilla for original checkpoints')
    parser.add_argument('--feedback_norm', type=float, default=None)
    parser.add_argument('--feedback_scale', type=float, default=1.0)
    parser.add_argument('--results_file', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--norm_stats_file', type=str, default=None)
    parser.add_argument('--norm_stats_max_records', type=int, default=2000)
    parser.add_argument('--projection_type', choices=['linear', 'sts'], default='linear')
    parser.add_argument('--sts_bank_size', type=int, default=64)
    parser.add_argument('--sts_temperature', type=float, default=1.0)
    parser.add_argument('--sts_bank_norm_scale', type=float, default=1.0)
    parser.add_argument('--sts_bank_cache', type=str, default=None)
    parser.add_argument('--sts_kmeans_niter', type=int, default=20)
    parser.add_argument('--sts_kmeans_device', choices=['auto', 'cpu', 'gpu'], default='auto')
    parser.add_argument('--sts_bank_metrics_interval', type=int, default=50)

    return parser.parse_args()

def extract_answer_math(response_text):
    cleaned_str = response_text.replace(',', '').replace('%', '').replace('$', '')
    match = re.findall(r'(-?[\d,]+(?:\.\d+)?)', cleaned_str)
    if match:
        try:
            val_str = match[-1].replace(',', '')
            return round(float(val_str), 2) if '.' in val_str else int(val_str)
        except: return None
    return None

def extract_answer_boolean(response_text):
    raw_lower = response_text.lower()
    rev_text = raw_lower[::-1]
    last_yes = re.search(r'\bsey\b', rev_text)
    idx_yes = last_yes.start() if last_yes else len(rev_text)
    last_no = re.search(r'\bon\b', rev_text)
    idx_no = last_no.start() if last_no else len(rev_text)
    if idx_yes == len(rev_text) and idx_no == len(rev_text): return None
    return 'Yes' if idx_yes < idx_no else 'No'

def extract_answer_option(response_text):
    rev_text = response_text.lower()[::-1]
    match = re.search(r'\b[a-f]\b', rev_text)
    if match: return match.group(0).upper()
    return None

def main():
    args = parse_args()
    norm_monitor = NormMonitor(args.norm_stats_file, args.norm_stats_max_records) if args.norm_stats_file else None

    if not os.path.exists(args.model_id):
        raise ValueError(f"Model path does not exist: {args.model_id}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info(f"Loading model from {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = UnifiedSoftCoT(
        model_id=args.model_id,
        num_thought_tokens=args.num_thought_tokens,
        tune_base_model=args.tune_base_model,
        path_to_projection_module=args.params_file_name,
        projection_type=args.projection_type,
        sts_bank_size=args.sts_bank_size,
        sts_temperature=args.sts_temperature,
        sts_bank_norm_scale=args.sts_bank_norm_scale,
        sts_init_seed=42,
        sts_bank_cache=args.sts_bank_cache,
        sts_kmeans_niter=args.sts_kmeans_niter,
        sts_kmeans_device=args.sts_kmeans_device,
        sts_bank_metrics_interval=args.sts_bank_metrics_interval,
    )
    model.eval()
    logger.info(f"Model loaded. N={args.num_thought_tokens}, device={model.device}")

    if args.task_name == 'gsm8k':
        db = GSM8KLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
        extract_fn = extract_answer_math
    elif args.task_name == 'asdiv-aug':
        db = AugASDivLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
        extract_fn = extract_answer_math
    elif args.task_name == 'strategyqa':
        db = StrategyQALoader().load(args.data_path)
        preprocess_fn = pre_process_strategy_qa_unified
        extract_fn = extract_answer_boolean
    elif args.task_name == 'aqua':
        db = AQuALoader().load(args.data_path)
        preprocess_fn = pre_process_aqua_unified
        extract_fn = extract_answer_option
    elif args.task_name == 'du':
        db = DULoader().load(args.data_path)
        preprocess_fn = pre_process_du_unified
        extract_fn = extract_answer_option
    else:
        raise NotImplementedError

    feedback_config = configure_feedback(
        model, tokenizer, db.get_dataset('train'), preprocess_fn,
        mode=args.feedback_mode, norm=args.feedback_norm, scale=args.feedback_scale,
        checkpoint=args.params_file_name,
    )
    logger.info(f'Feedback configuration: {feedback_config}')

    if args.dataset_split not in db.datasets:
        logger.error(f"Split {args.dataset_split} not found in dataset! Available: {list(db.datasets.keys())}")
        return

    ds = db.get_dataset(args.dataset_split)
    logger.info(f"🚀 Evaluating on split: {args.dataset_split} (Size: {len(ds)})")

    if args.test_k > 0:
        ds = ds[:args.test_k]

    generation_config = GenerationConfig.from_pretrained(args.model_id)
    if 'llama' in args.model_id.lower():
        generation_config.pad_token_id = 128009
    else:
        generation_config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643

    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.top_p = 1.0
    generation_config.temperature = 1.0
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = True

    correct_count = 0
    records = []
    evaluation_start = time.perf_counter()

    for idx, ins in enumerate(tqdm(ds, desc=f'Eval {args.dataset_split}', ncols=100)):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        raw_gt = None
        if 'correct' in ins: raw_gt = ins['correct']
        elif 'answer' in ins: raw_gt = ins['answer']

        gt_val = None
        if args.task_name in ['gsm8k', 'asdiv-aug']:
            try:
                if isinstance(raw_gt, str):
                    ans_part = raw_gt.split('\n')[-1].replace(',', '').replace('####', '').strip()
                    gt_val = float(ans_part) if '.' in ans_part else int(ans_part)
            except: gt_val = None
        elif args.task_name == 'strategyqa':
            if isinstance(raw_gt, bool): gt_val = 'Yes' if raw_gt else 'No'
            else: gt_val = str(raw_gt)
        elif args.task_name in ['aqua', 'du']:
            try:
                if isinstance(raw_gt, str) and '####' in raw_gt:
                    gt_val = raw_gt.split('####')[-1].strip()
                else:
                    gt_val = raw_gt.strip() if isinstance(raw_gt, str) else None
            except: gt_val = None

        inputs = preprocess_fn(
            ins, tokenizer, num_thought_tokens=args.num_thought_tokens,
            split='test', device=model.device,
        )

        if args.print_input:
            logger.info(f'Decoded Prompt: {tokenizer.decode(inputs["input_ids"][0])}')

        if args.num_thought_tokens > 0:
            with torch.no_grad():
                inputs_embeds = model.get_inputs_embeds_for_unified_model(
                    inputs['input_ids'], inputs['attention_mask'], inputs['thought_index'], args.print_input,
                    norm_monitor=norm_monitor,
                    norm_meta={'phase': 'eval_prompt_build', 'sample_index': idx, 'seed': args.seed},
                )
        else:
            inputs_embeds = model.model.get_input_embeddings()(inputs['input_ids'])

        with torch.no_grad():
            outputs = model.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                generation_config=generation_config,
                num_return_sequences=args.num_return_sequences,
                use_cache=True
            )

        response = outputs[0]
        full_text = tokenizer.decode(response, skip_special_tokens=True)

        if args.print_response:
            logger.info(f'Full Response: {full_text}')

        pred_val = extract_fn(full_text)

        is_correct = False
        if pred_val is not None and gt_val is not None:
            if args.task_name in ['gsm8k', 'asdiv-aug']:
                if abs(pred_val - gt_val) < 1e-4: is_correct = True
            else:
                if str(pred_val).upper() == str(gt_val).upper(): is_correct = True

        if is_correct: correct_count += 1
        if args.results_file:
            records.append({
                'index': idx, 'question': ins['question'], 'ground_truth': gt_val,
                'prediction': pred_val, 'correct': is_correct,
                'output_tokens': response.numel(), 'response': full_text,
                'feedback': model.last_feedback_stats[0] if model.last_feedback_stats else {},
            })

        if not args.print_input and not args.print_response:
             logger.info(f"[{idx + 1}] GT: {gt_val} | Pred: {pred_val} | Acc: {correct_count / (idx + 1) * 100:.2f}%")

    acc = correct_count / len(ds) * 100
    logger.info(f'Final Accuracy on {args.dataset_split}: {correct_count}/{len(ds)} = {acc:.2f}%')
    if args.results_file:
        summary = {
            'correct': correct_count, 'total': len(ds), 'accuracy': acc,
            'evaluation_seconds': time.perf_counter() - evaluation_start,
            'mean_output_tokens': sum(r['output_tokens'] for r in records) / len(records),
        }
        if args.num_thought_tokens > 0:
            metric_keys = ['hidden_norm', 'projected_norm', 'feedback_norm', 'direction_cosine']
            if args.projection_type == 'sts':
                metric_keys.extend([
                    'query_norm', 'bank_norm', 'output_norm',
                    'attention_entropy', 'attention_effective_tokens', 'attention_top1',
                ])
            for key in metric_keys:
                summary[f'mean_{key}_by_position'] = torch.tensor(
                    [r['feedback'][key] for r in records], dtype=torch.float64,
                ).mean(dim=0).tolist()
            if args.projection_type == 'sts':
                summary['bank_diagnostics'] = model.get_sts_bank_diagnostics()
                top1_usage = []
                for position in range(args.num_thought_tokens):
                    counts = {}
                    for record in records:
                        selected = str(record['feedback']['attention_argmax'][position])
                        counts[selected] = counts.get(selected, 0) + 1
                    top1_usage.append(dict(sorted(counts.items(), key=lambda item: -item[1])))
                summary['attention_top1_usage_by_position'] = top1_usage
        result_path = Path(args.results_file)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps({
            'args': vars(args), 'feedback': feedback_config,
            'generation_config': generation_config.to_dict(),
            'summary': summary, 'predictions': records,
        }, indent=2, ensure_ascii=False) + '\n')
        logger.info(f'Results saved to {result_path}')
    if norm_monitor is not None:
        norm_monitor.close()
        logger.info(f'Norm stats saved to {args.norm_stats_file}')

if __name__ == '__main__':
    main()
