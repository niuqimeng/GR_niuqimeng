import pickle
import json
import pandas as pd
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s)s - %(message)s")
logger = logging.getLogger(__name__)

def build_semantic_label_mapping():
    try:
        items = pickle.load(open("rqvae_output/items.pkl", "rb"))
        semantic_ids = pickle.load(open("rqvae_output/semantic_ids.pkl", "rb"))
        raw_df = pd.read_pickle("rqvae_output/raw_df.pkl")
    except FileNotFoundError as e:
        logger.error(f"Missing file: {e}")
        raise

    level1_ids, level2_ids, level3_ids = semantic_ids

    asin_keywords = defaultdict(list)
    label_rules = {
        "肌肤护理": ["skin"],
        "干性肌肤": ["dry"],
        "保湿补水": ["moistur", "hydrat"],
        "面霜": ["cream"],
        "乳液": ["lotion"],
        "眼部护理": ["eye"],
        "抗老抗皱": ["anti", "aging", "wrinkle"],
        "敏感肌适用": ["sensitive"],
        "头发护理": ["hair"],
        "洗发水": ["shampoo"],
        "护发素": ["conditioner"],
        "香气": ["scent"],
        "香氛": ["fragrance"],
        "薰衣草香": ["lavender"],
        "天然有机": ["natural", "organic"],
        "高品质": ["quality"]
    }

    for _, row in raw_df.iterrows():
        asin = row["asin"]
        review_text = str(row.get("reviewText", "")).lower()
        for label, keywords in label_rules.items():
            if any(kw in review_text for kw in keywords):
                asin_keywords[asin].append(label)
        if not asin_keywords[asin]:
            asin_keywords[asin].append("护肤保养")

    asin_main_label = {
        asin: Counter(kws).most_common(1)[0][0]
        for asin, kws in asin_keywords.items()
    }

    level_label_stats = [defaultdict(list), defaultdict(list), defaultdict(list)]
    for idx, item in enumerate(items):
        asin = item["item_id"]
        label = asin_main_label.get(asin, "护肤保养")
        level_label_stats[0][str(int(level1_ids[idx]))].append(label)
        level_label_stats[1][str(int(level2_ids[idx]))].append(label)
        level_label_stats[2][str(int(level3_ids[idx]))].append(label)

    code2label = []
    for stats in level_label_stats:
        level_mapping = {}
        for code, labels in stats.items():
            level_mapping[code] = Counter(labels).most_common(1)[0][0]
        code2label.append(level_mapping)

    with open("rqvae_output/code2label.json", "w", encoding="utf-8") as f:
        json.dump(code2label, f, ensure_ascii=False, indent=2)

    logger.info(f"Mapping built: L0={len(code2label[0])}, L1={len(code2label[1])}, L2={len(code2label[2])}")

if __name__ == "__main__":
    build_semantic_label_mapping()