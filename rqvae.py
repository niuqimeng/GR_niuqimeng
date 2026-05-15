import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rqvae_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RQVAECfg:
    latent_dim = 128
    num_levels = 3
    vocab_sizes = [64, 128, 256]
    epochs = 20
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/root/autodl-tmp/All_Beauty_5.json"
    output_dir = "rqvae_output"
    val_ratio = 0.1

class RQVAE(nn.Module):
    def __init__(self, latent_dim, num_levels, vocab_sizes):
        super().__init__()
        self.num_levels = num_levels
        self.vocab_sizes = vocab_sizes

        self.encoder = nn.Sequential(
            nn.Linear(768, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, sum(vocab_sizes))
        )

        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(s, latent_dim)) for s in vocab_sizes
        ])

        self.decoder = nn.Linear(latent_dim * num_levels, 768)

    def encode(self, x):
        logits = self.encoder(x)
        logits_list = torch.split(logits, self.vocab_sizes, dim=-1)
        semantic_ids = [torch.argmax(l, dim=-1) for l in logits_list]
        quantized_feats = [self.codebooks[i][semantic_ids[i]] for i in range(self.num_levels)]
        return quantized_feats, semantic_ids, logits_list

    def forward(self, x):
        quant_feats, sem_ids, logits_list = self.encode(x)
        recon_feat = self.decoder(torch.cat(quant_feats, dim=-1))

        recon_loss = F.mse_loss(recon_feat, x)

        balance_loss = 0.0
        for level_logits in logits_list:
            level_probs = F.softmax(level_logits, dim=-1).mean(dim=0)
            balance_loss -= torch.sum(level_probs * torch.log(level_probs + 1e-8))

        total_loss = recon_loss + 0.1 * balance_loss
        return total_loss, sem_ids

def load_beauty_data(data_path):
    logger.info(f"Loading data: {data_path}")
    raw_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                raw_data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logger.warning(f"Bad line {line_num+1}, skipped")
                continue

    df = pd.DataFrame(raw_data)
    unique_asins = df["asin"].unique()
    item_map = {asin: {"item_id": asin, "title": f"product_{asin}"} for asin in unique_asins}
    item_feats = torch.randn(len(unique_asins), 768, device=RQVAECfg.device)

    logger.info(f"Loaded: {len(unique_asins)} items, {len(df)} reviews")
    return list(item_map.values()), item_feats, df

def train_rqvae():
    os.makedirs(RQVAECfg.output_dir, exist_ok=True)
    items, feats, raw_df = load_beauty_data(RQVAECfg.data_path)

    train_size = int(len(feats) * (1 - RQVAECfg.val_ratio))
    train_feats = feats[:train_size]
    val_feats = feats[train_size:]

    model = RQVAE(
        RQVAECfg.latent_dim,
        RQVAECfg.num_levels,
        RQVAECfg.vocab_sizes
    ).to(RQVAECfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=RQVAECfg.lr)

    best_val_loss = float("inf")
    for epoch in range(RQVAECfg.epochs):
        model.train()
        train_loss = 0.0
        loss, _ = model(train_feats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss, _ = model(val_feats)

        logger.info(f"Epoch {epoch+1}/{RQVAECfg.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(RQVAECfg.output_dir, "rqvae_best.pt"))

    torch.save(model.state_dict(), os.path.join(RQVAECfg.output_dir, "rqvae_final.pt"))
    pd.to_pickle(items, os.path.join(RQVAECfg.output_dir, "items.pkl"))
    pd.to_pickle(raw_df, os.path.join(RQVAECfg.output_dir, "raw_df.pkl"))

    model.eval()
    with torch.no_grad():
        _, semantic_ids = model(feats)
    semantic_ids = [ids.cpu().numpy() for ids in semantic_ids]
    pd.to_pickle(semantic_ids, os.path.join(RQVAECfg.output_dir, "semantic_ids.pkl"))

    logger.info(f"Training done | Total items: {len(items)}")

if __name__ == "__main__":
    train_rqvae()