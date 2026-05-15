import torch
import pickle
import json
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s)s - %(message)s")
logger = logging.getLogger(__name__)

CFG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_path": "/root/autodl-tmp/Qwen1.5-1.8B-Chat",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "v_proj"],
    "batch_size": 2,
    "epochs": 3,
    "lr": 2e-5,
    "max_length": 512,
    "input_dir": "rqvae_output",
    "output_dir": "sft_final",
    "test_size": 0.2,
    "random_seed": 42
}

tokenizer = AutoTokenizer.from_pretrained(CFG["model_path"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CFG["model_path"],
    torch_dtype=torch.bfloat16,
    device_map=CFG["device"],
    trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=CFG["lora_r"],
    lora_alpha=CFG["lora_alpha"],
    target_modules=CFG["lora_target_modules"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
logger.info(model.print_trainable_parameters())

class RecDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=tokenizer.pad_token_id
    )

def build_train_data():
    items = pickle.load(open(os.path.join(CFG["input_dir"], "items.pkl"), "rb"))
    semantic_ids = pickle.load(open(os.path.join(CFG["input_dir"], "semantic_ids.pkl"), "rb"))
    code2label = json.load(open(os.path.join(CFG["input_dir"], "code2label.json")))
    raw_df = pd.read_pickle(os.path.join(CFG["input_dir"], "raw_df.pkl"))

    asin2title = {item["item_id"]: item["title"] for item in items}
    level1_ids, level2_ids, _ = semantic_ids
    asin2semantic = {}

    for idx, item in enumerate(items):
        asin = item["item_id"]
        sem1 = code2label[0].get(str(int(level1_ids[idx])), "")
        sem2 = code2label[1].get(str(int(level2_ids[idx])), "")
        asin2semantic[asin] = sem1 if sem1 == sem2 else f"{sem1}、{sem2}"

    user_sequences = []
    for uid, group in raw_df.groupby("reviewerID"):
        asin_seq = group["asin"].tolist()
        if len(asin_seq) < 2:
            continue
        for i in range(1, len(asin_seq)):
            hist_asins = asin_seq[:i]
            target_asin = asin_seq[i]
            hist_titles = "、".join([asin2title[s] for s in hist_asins])
            target_sem = asin2semantic[target_asin]
            target_title = asin2title[target_asin]

            prompt = f"用户ID：{uid}\n用户历史购买：{hist_titles}"
            ans = f"推荐理由：用户偏好{target_sem}类产品，推荐商品：{target_title}"
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}]

            text = tokenizer.apply_chat_template(messages, tokenize=False)
            enc = tokenizer(text, truncation=True, max_length=CFG["max_length"], return_tensors="pt")
            user_sequences.append({"input_ids": enc.input_ids.squeeze()})

    train_data, test_data = train_test_split(user_sequences, test_size=CFG["test_size"], random_state=CFG["random_seed"])
    train_input_ids = [d["input_ids"] for d in train_data]

    test_user_seq = []
    for d in test_data:
        test_user_seq.append(d)
    with open(os.path.join(CFG["input_dir"], "test_data.pkl"), "wb") as f:
        pickle.dump([d["input_ids"] for d in test_data], f)
    with open(os.path.join(CFG["input_dir"], "test_user_seq.pkl"), "wb") as f:
        pickle.dump(test_user_seq, f)

    logger.info(f"Data ready: Train={len(train_input_ids)}, Test={len(test_data)}")
    return train_input_ids

def train_sft():
    train_data = build_train_data()
    dataset = RecDataset(train_data)
    loader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])

    model.train()
    for epoch in range(CFG["epochs"]):
        total_loss = 0
        for batch in loader:
            batch = batch.to(CFG["device"])
            out = model(input_ids=batch, labels=batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

    os.makedirs(CFG["output_dir"], exist_ok=True)
    model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])
    logger.info(f"Model saved to {CFG['output_dir']}")

if __name__ == "__main__":
    train_sft()