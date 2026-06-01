import os
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
from tokenizers.processors import TemplateProcessing

# =========================================================
# Paths
# =========================================================
ROOT = Path("/content/drive/MyDrive/Colab Notebooks/CMPE-679-Labs/GloriaMbaka_LAB2")
DATA_DIR = ROOT / "data"
TOKENIZER_DIR = ROOT / "tokenizer"
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SRC_FILE = DATA_DIR / "train.en"
TRAIN_TGT_FILE = DATA_DIR / "train.de"

TOKENIZER_PATH = TOKENIZER_DIR / "my_trained_tokenizer.json"

# =========================================================
# Settings
# =========================================================
VOCAB_SIZE = 30000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# =========================================================
# Sanity checks
# =========================================================
if not TRAIN_SRC_FILE.exists():
    raise FileNotFoundError(f"Missing training source file: {TRAIN_SRC_FILE}")

if not TRAIN_TGT_FILE.exists():
    raise FileNotFoundError(f"Missing training target file: {TRAIN_TGT_FILE}")

print("Training tokenizer from:")
print("  ", TRAIN_SRC_FILE)
print("  ", TRAIN_TGT_FILE)
print("Saving tokenizer to:")
print("  ", TOKENIZER_PATH)

# =========================================================
# Build tokenizer
# =========================================================
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents(),
])

# ByteLevel preserves whitespace properly for BPE
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# =========================================================
# Train tokenizer
# =========================================================
files = [str(TRAIN_SRC_FILE), str(TRAIN_TGT_FILE)]
tokenizer.train(files=files, trainer=trainer)

# =========================================================
# Post-processing for BOS/EOS
# =========================================================
bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")

if bos_id is None or eos_id is None:
    raise ValueError("Could not find [BOS] or [EOS] token IDs after training.")

tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    pair="[BOS] $A [EOS] $B:1 [EOS]:1",
    special_tokens=[
        ("[BOS]", bos_id),
        ("[EOS]", eos_id),
    ],
)

# ByteLevel decoder restores spaces properly
tokenizer.decoder = decoders.ByteLevel()

# =========================================================
# Save tokenizer
# =========================================================
tokenizer.save(str(TOKENIZER_PATH))

print("\nTokenizer training complete.")
print("Saved tokenizer to:", TOKENIZER_PATH)
print("Vocab size:", tokenizer.get_vocab_size())

# =========================================================
# Quick test
# =========================================================
sample = "I am going to school."
encoded = tokenizer.encode(sample)

print("\nQuick tokenizer test")
print("Input:", sample)
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)
print("Decoded:", tokenizer.decode(encoded.ids))