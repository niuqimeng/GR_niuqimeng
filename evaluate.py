import torch
import pickle
import re
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s)s - %(message)s")
logger = logging.getLogger(__name__)

EVAL_CFG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_path": "sft_final",
    "input_dir": "rqvae_output",
    "max_new_tokens": 150,
    "top_k": [5, 10]
}

def load_eval_data():
    items = pickle.load(open(os.path.join(EVAL_CFG["input_dir"], "items.pkl"), "rb"))
    asin2title = {it["item_id"]: it["title"] for it in items}
    title2idx = {t: i for i, t in enumerate(asin2title.values())}
    test_seq = pickle.load(open(os.path.join(EVAL_CFG["input_dir"], "test_user_seq.pkl"), "rb"))
    return test_seq, title2idx

def evaluate():
    test_seq, title2idx = load_eval_data()
    tokenizer = AutoTokenizer.from_pretrained(EVAL_CFG["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        EVAL_CFG["model_path"], torch_dtype=torch.bfloat16, device_map=EVAL_CFG["device"]
    )
    model.eval()

    recall = {k:0 for k in EVAL_CFG["top_k"]}
    mrr = 0.0
    total = 0
    invalid = 0

    with torch.no_grad():
        for idx, sample in enumerate(test_seq):
            uid = sample["uid"]
            hist = sample["hist_str"]
            tgt = sample["tgt_title"]

            prompt = f"用户ID：{uid}\n用户历史购买：{hist}"
            messages = [{"role":"user","content":prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(EVAL_CFG["device"])

            out = model.generate(**inputs, max_new_tokens=EVAL_CFG["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)
            pred_text = tokenizer.decode(out[0], skip_special_tokens=True)

            match = re.search(r"推荐商品：(product_\w+)", pred_text)
            if not match:
                invalid +=1
                total +=1
                continue
            pred = match.group(1)
            if pred not in title2idx:
                invalid +=1
                total +=1
                continue

            rank = title2idx[pred] + 1
            for k in EVAL_CFG["top_k"]:
                if rank <= k:
                    recall[k] +=1
            mrr += 1.0/rank
            total +=1

    final_recall = {k: recall[k]/total for k in EVAL_CFG["top_k"]}
    final_mrr = mrr/total
    logger.info("="*50)
    logger.info("Evaluation Results")
    logger.info("="*50)
    for k,v in final_recall.items():
        logger.info(f"Recall@{k}: {v:.4f}")
    logger.info(f"MRR: {final_mrr:.4f}")
    logger.info(f"Invalid rate: {invalid/len(test_seq):.4f}")

if __name__ == "__main__":
    evaluate()