import os
import re
import argparse
import random
from copy import deepcopy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GenerationConfig
from tqdm import tqdm
import sys
import logging
import json
from feedback import configure_feedback
from norm_monitor import NormMonitor

sys.path.append(os.getcwd())

try:
    from data_loader import GSM8KLoader, AQuALoader, DULoader, StrategyQALoader, AugASDivLoader
    from unified_llm_model import UnifiedSoftCoT
    from unified_utils import (
        pre_process_gsm8k_unified,
        pre_process_aqua_unified,
        pre_process_du_unified,
        pre_process_strategy_qa_unified
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_buffer.log")

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def enforce_eager_backend():
    os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def seed_init(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def extract_answer_math(response_text):
    match = re.search(r'\\boxed\{([^}]+)\}', response_text)
    t = match.group(1) if match else response_text
    t = t.replace(",", "").replace("%", "").replace("$", "").replace("####", "").strip()
    m = re.findall(r"([-+]?\d+(?:\.\d+)?)", t)
    if not m: return None
    try:
        return int(m[-1]) if "." not in m[-1] else round(float(m[-1]), 2)
    except: return None

def extract_answer_boolean(response_text):
    text = response_text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    idx_yes = text.rfind('yes')
    idx_no = text.rfind('no')
    if idx_yes == -1 and idx_no == -1: return None
    return 'Yes' if idx_yes > idx_no else 'No'

def extract_answer_option(response_text):
    rev_text = response_text.lower()[::-1]
    match = re.search(r'\b[a-f]\b', rev_text)
    if match: return match.group(0).upper()
    return None

def extract_answer(text: str, task_name: str):
    text = text.strip()
    if task_name in ['gsm8k', 'asdiv-aug']: return extract_answer_math(text)
    elif task_name == 'strategyqa': return extract_answer_boolean(text)
    elif task_name in ['aqua', 'du']: return extract_answer_option(text)
    return None

def compute_reward(pred, gt, task_name):
    if pred is None or gt is None: return 0.0
    if task_name in ['gsm8k', 'asdiv-aug']:
        return 1.0 if abs(pred - gt) < 1e-4 else 0.0
    else:
        return 1.0 if str(pred).upper() == str(gt).upper() else 0.0

def build_prompt_embeddings_indep(model, input_ids, attention_mask, thought_index, use_grad=False, norm_monitor=None, norm_meta=None):
    with torch.no_grad():
        base_embeds = model.model.get_input_embeddings()(input_ids)
        if norm_monitor is not None:
            norm_monitor.record_tensor('llm_input_token_embedding.raw_prompt', base_embeds, norm_meta)
        outputs = model.model(inputs_embeds=base_embeds, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]

    inputs_embeds_new = base_embeds.clone()
    batch_size = input_ids.size(0)
    for b in range(batch_size):
        s, e = thought_index[b, 0].item(), thought_index[b, 1].item()
        if s == 0 and e == 0: continue

        soft_token_base = base_embeds[b, s:e]
        raw = hidden[b, s:e]
        meta = dict(norm_meta or {})
        meta.update({'batch_item': b, 'thought_start': s, 'thought_end': e})
        if norm_monitor is not None:
            norm_monitor.record_tensor('soft_token_embedding.before_projection', soft_token_base, meta)
            norm_monitor.record_tensor('projection_input.hidden_state', raw, meta)
        if raw.size(0) != model.num_thought_tokens: continue

        if use_grad:
            projected = model.project_hidden_states(
                raw, norm_monitor=norm_monitor, norm_meta=meta,
            )
        else:
            with torch.no_grad():
                projected = model.project_hidden_states(
                    raw, norm_monitor=norm_monitor, norm_meta=meta,
                )
        feedback = model.scale_feedback(projected)
        if norm_monitor is not None:
            norm_monitor.record_tensor('projection_output.raw', projected, meta)
            norm_monitor.record_tensor('projection_output.scaled_feedback', feedback, meta)
            norm_monitor.record_cosine('projection_raw_to_scaled_feedback.cosine', projected, feedback, meta)
        inputs_embeds_new[b, s:e] = feedback
    if norm_monitor is not None:
        norm_monitor.record_tensor('llm_input_token_embedding.final_prompt', inputs_embeds_new, norm_meta)
    return inputs_embeds_new

def get_logprobs(model, input_ids, attention_mask, thought_index, response_ids, use_grad=False, norm_monitor=None, norm_meta=None):
    p_embeds = build_prompt_embeddings_indep(model, input_ids, attention_mask, thought_index, use_grad=use_grad, norm_monitor=norm_monitor, norm_meta=norm_meta)
    r_embeds = model.model.get_input_embeddings()(response_ids)
    full_embeds = torch.cat([p_embeds, r_embeds], dim=1)

    resp_mask = torch.ones_like(response_ids)
    full_mask = torch.cat([attention_mask, resp_mask], dim=1)

    if use_grad:
        full_embeds.requires_grad_(True)
        outputs = model.model(inputs_embeds=full_embeds, attention_mask=full_mask)
    else:
        with torch.no_grad():
            outputs = model.model(inputs_embeds=full_embeds, attention_mask=full_mask)

    prompt_len = input_ids.size(1)
    logits = outputs.logits[:, prompt_len-1 : -1, :]

    return torch.gather(F.log_softmax(logits, dim=-1), -1, response_ids.unsqueeze(-1)).squeeze(-1)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task_name", type=str, required=True, choices=['gsm8k', 'asdiv-aug', 'aqua', 'du', 'strategyqa'])
    ap.add_argument("--path_to_projection_module", default="None")
    ap.add_argument("--num_thought_tokens", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=5)
    ap.add_argument("--train_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save_every", type=int, default=300)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--projection_type", choices=['linear', 'sts'], default='linear')
    ap.add_argument("--sts_bank_size", type=int, default=64)
    ap.add_argument("--sts_temperature", type=float, default=1.0)
    ap.add_argument("--sts_bank_norm_scale", type=float, default=1.0)
    ap.add_argument("--feedback_mode", choices=['vanilla', 'scale_match'], default='vanilla')
    ap.add_argument("--feedback_norm", type=float, default=None,
                    help='Reference L2 norm; defaults to the training-prompt embedding mean')
    ap.add_argument("--feedback_scale", type=float, default=1.0)

    ap.add_argument("--epsilon_left", type=float, default=3e-4, help="GSPO Left clipping range (Paper: 3e-4)")
    ap.add_argument("--epsilon_right", type=float, default=4e-4, help="GSPO Right clipping range (Paper: 4e-4)")

    ap.add_argument("--beta_kl", type=float, default=0.01, help="KL penalty coefficient (Standard RLHF default, GSPO paper omits specific value)")

    ap.add_argument("--episodes_per_round", type=int, default=16)
    ap.add_argument("--update_epochs", type=int, default=3)

    ap.add_argument("--max_data_epochs", type=int, default=0, help="Data budget in epochs (0=Unlimited, N=N*DatasetSize)")
    ap.add_argument("--norm_stats_file", type=str, default=None, help="Write tensor norm stats as JSONL when set")
    ap.add_argument("--norm_stats_max_records", type=int, default=2000)

    return ap.parse_args()

def main():
    enforce_eager_backend()
    args = parse_args()
    seed_init(args.seed)

    logger = setup_logger(args.output_dir)
    norm_monitor = NormMonitor(args.norm_stats_file, args.norm_stats_max_records) if args.norm_stats_file else None
    logger.info(f"🚀 Start Buffer Training: Task={args.task_name}, N={args.num_thought_tokens}, G={args.group_size}, LR={args.lr}, KL_Beta={args.beta_kl}")
    if norm_monitor is not None:
        logger.info(f"📏 Norm monitor enabled: {args.norm_stats_file}")

    os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if 'llama' in args.model_id.lower():
        pad_token_id = 128009
        logger.info("🔧 Detected Llama-3. Using Pad Token ID: 128009")
    else:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
        logger.info(f"🔧 Using Pad Token ID: {pad_token_id}")

    model = UnifiedSoftCoT(
        args.model_id,
        args.num_thought_tokens,
        path_to_projection_module=args.path_to_projection_module,
        projection_type=args.projection_type,
        sts_bank_size=args.sts_bank_size,
        sts_temperature=args.sts_temperature,
        sts_bank_norm_scale=args.sts_bank_norm_scale,
        sts_init_seed=args.seed,
    )
    model.eval()
    dev = model.device

    opt = torch.optim.AdamW(model.projections.parameters(), lr=args.lr)

    proj_old = deepcopy(model.projections)
    for p in proj_old.parameters(): p.requires_grad = False

    if args.task_name == 'gsm8k':
        db = GSM8KLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
    elif args.task_name == 'asdiv-aug':
        db = AugASDivLoader().load(args.data_path)
        preprocess_fn = pre_process_gsm8k_unified
    elif args.task_name == 'aqua':
        db = AQuALoader().load(args.data_path)
        preprocess_fn = pre_process_aqua_unified
    elif args.task_name == 'du':
        db = DULoader().load(args.data_path)
        preprocess_fn = pre_process_du_unified
    elif args.task_name == 'strategyqa':
        db = StrategyQALoader().load(args.data_path)
        preprocess_fn = pre_process_strategy_qa_unified
    else: raise ValueError

    train_ds = db.get_dataset("train")
    feedback_config = configure_feedback(
        model, tokenizer, train_ds, preprocess_fn, mode=args.feedback_mode,
        norm=args.feedback_norm, scale=args.feedback_scale,
    )
    logger.info(f"Feedback configuration: {feedback_config}")
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as config_file:
        json.dump({'args': vars(args), 'feedback': feedback_config}, config_file, indent=2)

    dataset_len = len(train_ds)
    if args.max_data_epochs > 0:
        max_budget = dataset_len * args.max_data_epochs
        logger.info(f"📊 Budget Control: Enabled. {args.max_data_epochs} Epochs * {dataset_len} Samples = {max_budget} Total Attempts")
    else:
        max_budget = float('inf')
        logger.info(f"📊 Budget Control: Unlimited (0). Only constrained by train_steps.")

    gen_config = GenerationConfig(
        do_sample=True, temperature=1.0, max_new_tokens=args.max_new_tokens,
        pad_token_id=pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True, top_p=1.0
    )

    def data_generator():
        while True:
            idxs = list(range(len(train_ds)))
            random.shuffle(idxs)
            for i in idxs: yield train_ds[i]
    data_iter = data_generator()

    pbar_steps = tqdm(total=args.train_steps, position=0, desc="Train Steps", ascii=True, ncols=100)

    pbar_budget = None
    if args.max_data_epochs > 0:
        pbar_budget = tqdm(total=max_budget, position=1, desc="Data Budget", ascii=True, ncols=100)

    global_step = 0
    total_attempts = 0

    try:
        while global_step < args.train_steps:
            proj_old.load_state_dict(model.projections.state_dict())
            buffer = []
            reward_stats = []

            while len(buffer) < args.episodes_per_round:
                ins = next(data_iter)

                gt = None
                if args.task_name in ['gsm8k', 'asdiv-aug']:
                    try: gt = extract_answer_math(ins['answer'].split('\n')[-1])
                    except: pass
                elif args.task_name == 'strategyqa':
                     if 'answer' in ins:
                        raw = ins['answer']
                        gt = 'Yes' if (raw is True or str(raw).lower()=='yes') else 'No'
                elif args.task_name in ['aqua', 'du']:
                    raw = None
                    if 'correct' in ins:
                        raw = ins['correct']
                    elif 'answer' in ins:
                        raw = ins['answer']

                    if raw: gt = raw.split('####')[-1].strip() if '####' in raw else raw.strip()

                processed = preprocess_fn(ins, tokenizer, args.num_thought_tokens, device=dev, split='test')

                input_ids = processed['input_ids'].repeat(args.group_size, 1)
                attention_mask = processed['attention_mask'].repeat(args.group_size, 1)
                thought_index = processed['thought_index'].repeat(args.group_size, 1)

                model.eval()
                model.projections.load_state_dict(proj_old.state_dict())

                with torch.no_grad():
                    p_embeds = build_prompt_embeddings_indep(
                        model, input_ids, attention_mask, thought_index,
                        norm_monitor=norm_monitor,
                        norm_meta={'phase': 'train_generate', 'global_step': global_step, 'total_attempts': total_attempts},
                    )
                    gen = model.model.generate(
                        inputs_embeds=p_embeds, attention_mask=attention_mask, generation_config=gen_config
                    )

                    total_attempts += 1
                    if pbar_budget:
                        pbar_budget.update(1)

                    prompt_len = input_ids.size(1)
                    if gen.size(1) >= prompt_len and torch.all(gen[:, :prompt_len].eq(input_ids)):
                        response_ids = gen[:, prompt_len:]
                    else:
                        response_ids = gen

                    if response_ids.size(1) == 0: continue

                    logp_old = get_logprobs(
                        model, input_ids, attention_mask, thought_index, response_ids,
                        norm_monitor=norm_monitor,
                        norm_meta={'phase': 'train_old_logprob', 'global_step': global_step, 'total_attempts': total_attempts},
                    )

                rewards = []
                texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                for txt in texts:
                    pred = extract_answer(txt, args.task_name)
                    rewards.append(compute_reward(pred, gt, args.task_name))

                r_tensor = torch.tensor(rewards, device=dev)
                reward_stats.append(r_tensor.mean().item())

                if r_tensor.std() < 1e-6:
                    continue

                adv = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)

                buffer.append({
                    'input_ids': input_ids.cpu(),
                    'attention_mask': attention_mask.cpu(),
                    'thought_index': thought_index.cpu(),
                    'response_ids': response_ids.cpu(),
                    'old_logprobs': logp_old.cpu(),
                    'advantages': adv.cpu()
                })

            avg_r = sum(reward_stats)/len(reward_stats) if reward_stats else 0.0

            model.train()
            model.model.gradient_checkpointing_enable()

            total_loss = 0.0
            total_kl = 0.0
            update_count = 0

            for epoch in range(args.update_epochs):
                random.shuffle(buffer)
                for batch in buffer:
                    b_input_ids = batch['input_ids'].to(dev)
                    b_attention_mask = batch['attention_mask'].to(dev)
                    b_thought_index = batch['thought_index'].to(dev)
                    b_response_ids = batch['response_ids'].to(dev)
                    b_old_logprobs = batch['old_logprobs'].to(dev)
                    b_adv = batch['advantages'].to(dev)

                    opt.zero_grad()
                    logp_new = get_logprobs(
                        model, b_input_ids, b_attention_mask, b_thought_index, b_response_ids, use_grad=True,
                        norm_monitor=norm_monitor,
                        norm_meta={'phase': 'train_update', 'global_step': global_step, 'update_count': update_count},
                    )

                    mask = (b_response_ids != pad_token_id).float()

                    log_diff = logp_new - b_old_logprobs
                    kl_val = 0.5 * ((log_diff * mask) ** 2).sum() / (mask.sum() + 1e-8)

                    ratio = torch.exp((log_diff * mask).sum(1) / (mask.sum(1)+1e-8))

                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1 - args.epsilon_left, 1 + args.epsilon_right) * b_adv

                    loss = -torch.min(surr1, surr2).mean() + args.beta_kl * kl_val

                    loss.backward()
                    opt.step()

                    total_loss += loss.item()
                    total_kl += kl_val.item()
                    update_count += 1

            global_step += 1
            avg_loss = total_loss/update_count if update_count > 0 else 0.0
            avg_kl = total_kl/update_count if update_count > 0 else 0.0

            pbar_steps.update(1)
            pbar_steps.set_description(f"Step {global_step} R={avg_r:.2f} L={avg_loss:.4f} KL={avg_kl:.4f}")

            logger.info(f"Step {global_step} | Reward: {avg_r:.4f} | Loss: {avg_loss:.6f} | KL: {avg_kl:.6f} | Total Data Attempts: {total_attempts}/{'Inf' if max_budget==float('inf') else max_budget}")

            if global_step % args.save_every == 0:
                s_path = os.path.join(args.output_dir, "ckpt", f"step{global_step}.bin")
                torch.save(model.projections.state_dict(), s_path)
                logger.info(f"💾 [SAVED] {s_path}")

            if args.max_data_epochs > 0 and total_attempts >= max_budget:
                logger.info(f"🛑 Data Budget Reached ({total_attempts} >= {max_budget}). Stopping training.")
                break

            if global_step >= args.train_steps: break

    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving...")
        torch.save(model.projections.state_dict(), os.path.join(args.output_dir, "ckpt", "interrupted.bin"))

    torch.save(model.projections.state_dict(), os.path.join(args.output_dir, "ckpt", "final.bin"))
    if norm_monitor is not None:
        norm_monitor.close()
        logger.info(f"📏 Norm monitor summary saved next to {args.norm_stats_file}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
