import os
import json
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerFast

# =========================================================
# Paths
# =========================================================
ROOT = Path("/content/drive/MyDrive/Colab Notebooks/CMPE-679-Labs/GloriaMbaka_LAB2")
DATA_DIR = ROOT / "data"
TOKENIZER_DIR = ROOT / "tokenizer"
CACHE_DIR = ROOT / "tokenized_cache"

TOKENIZER_PATH = TOKENIZER_DIR / "my_trained_tokenizer.json"

TRAIN_SRC_FILE = DATA_DIR / "train.en"
TRAIN_TGT_FILE = DATA_DIR / "train.de"
VAL_SRC_FILE   = DATA_DIR / "val.en"
VAL_TGT_FILE   = DATA_DIR / "val.de"
TEST_SRC_FILE  = DATA_DIR / "test.en"
TEST_TGT_FILE  = DATA_DIR / "test.de"

# =========================================================
# Settings
# =========================================================
MAX_SRC_LEN = 128
MAX_TGT_LEN = 128
FORCE_REBUILD_CACHE = os.environ.get("FORCE_REBUILD_CACHE", "false").lower() == "true"

# =========================================================
# Sanity checks
# =========================================================
required_files = [
    TRAIN_SRC_FILE, TRAIN_TGT_FILE,
    VAL_SRC_FILE, VAL_TGT_FILE,
    TEST_SRC_FILE, TEST_TGT_FILE,
    TOKENIZER_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

print("Tokenizer path:", TOKENIZER_PATH)
print("Cache dir:", CACHE_DIR)
print("FORCE_REBUILD_CACHE:", FORCE_REBUILD_CACHE)

# =========================================================
# Load tokenizer
# =========================================================
tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
tokenizer.pad_token = "[PAD]"
tokenizer.unk_token = "[UNK]"
tokenizer.bos_token = "[BOS]"
tokenizer.eos_token = "[EOS]"

print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
print("unk_token:", tokenizer.unk_token, tokenizer.unk_token_id)
print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)
print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)

if tokenizer.pad_token_id is None:
    raise ValueError("pad_token_id is None")
if tokenizer.bos_token_id is None:
    raise ValueError("bos_token_id is None")
if tokenizer.eos_token_id is None:
    raise ValueError("eos_token_id is None")

# =========================================================
# Cache helpers
# =========================================================
def cache_looks_valid(cache_dir: Path):
    if not cache_dir.exists() or not cache_dir.is_dir():
        return False

    required = [
        cache_dir / "dataset_dict.json",
        cache_dir / "tokenized_meta.json",
    ]
    if not all(p.exists() for p in required):
        return False

    try:
        ds = load_from_disk(str(cache_dir))
        for split in ["train", "validation", "test"]:
            if split not in ds:
                return False
        _ = ds["train"][0]
        return True
    except Exception as e:
        print(f"Existing cache failed validation: {e}")
        return False

# =========================================================
# Helpers
# =========================================================
def read_parallel(src_path: Path, tgt_path: Path):
    with open(src_path, "r", encoding="utf-8") as f:
        src_raw = [line.rstrip("\n") for line in f]

    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_raw = [line.rstrip("\n") for line in f]

    print(f"\nReading:")
    print(f"  {src_path.name}: raw lines = {len(src_raw)}")
    print(f"  {tgt_path.name}: raw lines = {len(tgt_raw)}")

    if len(src_raw) != len(tgt_raw):
        raise ValueError(
            f"Raw file length mismatch before cleaning:\n"
            f"{src_path}: {len(src_raw)} lines\n"
            f"{tgt_path}: {len(tgt_raw)} lines\n\n"
            f"This means the split files themselves are already misaligned. "
            f"Please regenerate the parallel split files together."
        )

    src_lines = []
    tgt_lines = []
    dropped_both_empty = 0
    dropped_one_side_empty = 0

    for s, t in zip(src_raw, tgt_raw):
        s = s.strip()
        t = t.strip()

        if not s and not t:
            dropped_both_empty += 1
            continue

        if not s or not t:
            dropped_one_side_empty += 1
            continue

        src_lines.append(s)
        tgt_lines.append(t)

    print(f"  Dropped pairs with both sides empty: {dropped_both_empty}")
    print(f"  Dropped pairs with one side empty: {dropped_one_side_empty}")
    print(f"  Final aligned pairs kept: {len(src_lines)}")

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            f"Aligned pair construction failed:\n"
            f"{src_path}: {len(src_lines)} lines\n"
            f"{tgt_path}: {len(tgt_lines)} lines"
        )

    if len(src_lines) == 0:
        raise ValueError(f"No usable aligned pairs found in {src_path} and {tgt_path}")

    return {"src": src_lines, "tgt": tgt_lines}


def build_dataset(src_path: Path, tgt_path: Path):
    data = read_parallel(src_path, tgt_path)
    return Dataset.from_dict(data)


def add_bos_eos(ids, bos_id, eos_id, max_len):
    ids = ids[: max_len - 2]
    return [bos_id] + ids + [eos_id]


def tokenize_batch(batch):
    src_enc = tokenizer(
        batch["src"],
        truncation=True,
        max_length=MAX_SRC_LEN - 2,
        padding=False,
        add_special_tokens=False,
    )

    tgt_enc = tokenizer(
        batch["tgt"],
        truncation=True,
        max_length=MAX_TGT_LEN - 2,
        padding=False,
        add_special_tokens=False,
    )

    input_ids = [
        add_bos_eos(ids, tokenizer.bos_token_id, tokenizer.eos_token_id, MAX_SRC_LEN)
        for ids in src_enc["input_ids"]
    ]

    attention_mask = [[1] * len(ids) for ids in input_ids]

    labels = [
        add_bos_eos(ids, tokenizer.bos_token_id, tokenizer.eos_token_id, MAX_TGT_LEN)
        for ids in tgt_enc["input_ids"]
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# =========================================================
# Main control flow
# =========================================================
if cache_looks_valid(CACHE_DIR) and not FORCE_REBUILD_CACHE:
    print("\nExisting tokenized cache found. Skipping rebuild:")
    print(CACHE_DIR)

    ds = load_from_disk(str(CACHE_DIR))
    meta_path = CACHE_DIR / "tokenized_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("\nCached metadata:")
        print(json.dumps(meta, indent=2))

    sample = ds["train"][0]
    print("\nSample tokenized example from cache:")
    print("input_ids:", sample["input_ids"][:20])
    print("attention_mask:", sample["attention_mask"][:20])
    print("labels:", sample["labels"][:20])

    print("\nDecoded source sample:")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))

    print("\nDecoded target sample:")
    print(tokenizer.decode(sample["labels"], skip_special_tokens=False))

    print("\nSpecial token checks:")
    print("Source starts with BOS:", sample["input_ids"][0] == tokenizer.bos_token_id)
    print("Source ends with EOS:", sample["input_ids"][-1] == tokenizer.eos_token_id)
    print("Target starts with BOS:", sample["labels"][0] == tokenizer.bos_token_id)
    print("Target ends with EOS:", sample["labels"][-1] == tokenizer.eos_token_id)
else:
    if CACHE_DIR.exists():
        print(f"\nRemoving old cache dir: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)

    # =========================================================
    # Load raw datasets
    # =========================================================
    train_ds = build_dataset(TRAIN_SRC_FILE, TRAIN_TGT_FILE)
    val_ds   = build_dataset(VAL_SRC_FILE, VAL_TGT_FILE)
    test_ds  = build_dataset(TEST_SRC_FILE, TEST_TGT_FILE)

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    print("\nRaw dataset sizes:")
    for split, ds in dataset_dict.items():
        print(f"  {split}: {len(ds)}")

    # =========================================================
    # Tokenize
    # =========================================================
    tokenized = dataset_dict.map(
        tokenize_batch,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="Tokenizing dataset",
    )

    # =========================================================
    # Save to disk
    # =========================================================
    tokenized.save_to_disk(str(CACHE_DIR))

    print("\nTokenized dataset saved to:")
    print(CACHE_DIR)

    # =========================================================
    # Save metadata
    # =========================================================
    meta = {
        "tokenizer_path": str(TOKENIZER_PATH),
        "max_src_len": MAX_SRC_LEN,
        "max_tgt_len": MAX_TGT_LEN,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_train": len(tokenized["train"]),
        "num_validation": len(tokenized["validation"]),
        "num_test": len(tokenized["test"]),
    }

    meta_path = CACHE_DIR / "tokenized_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Metadata saved to:", meta_path)

    # =========================================================
    # Quick sample checks
    # =========================================================
    sample = tokenized["train"][0]
    print("\nSample tokenized example:")
    print("input_ids:", sample["input_ids"][:20])
    print("attention_mask:", sample["attention_mask"][:20])
    print("labels:", sample["labels"][:20])

    print("\nDecoded source sample:")
    print(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))

    print("\nDecoded target sample:")
    print(tokenizer.decode(sample["labels"], skip_special_tokens=False))

    print("\nSpecial token checks:")
    print("Source starts with BOS:", sample["input_ids"][0] == tokenizer.bos_token_id)
    print("Source ends with EOS:", sample["input_ids"][-1] == tokenizer.eos_token_id)
    print("Target starts with BOS:", sample["labels"][0] == tokenizer.bos_token_id)
    print("Target ends with EOS:", sample["labels"][-1] == tokenizer.eos_token_id)
