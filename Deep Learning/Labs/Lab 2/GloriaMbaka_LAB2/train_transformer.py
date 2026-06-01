import os
import gc
import math
import random
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast


# ============================================================
# Configuration
# ============================================================
ROOT = os.environ.get(
    "LAB2_ROOT",
    "/content/drive/MyDrive/Colab Notebooks/CMPE-679-Labs/GloriaMbaka_LAB2"
)
ROOT = str(ROOT)
Path(ROOT).mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = os.path.join(ROOT, "tokenizer", "my_trained_tokenizer.json")
TOKENIZED_DIR = os.path.join(ROOT, "tokenized_cache")

CKPT_DIR = os.path.join(ROOT, "checkpoints")
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
LATEST_CKPT = os.path.join(CKPT_DIR, "lab2_en_de_latest.pt")
BEST_CKPT = os.path.join(CKPT_DIR, "lab2_en_de_best.pt")
FINAL_MODEL_PATH = os.path.join(ROOT, "transformer_en_de_final.pth")
BEST_MODEL_PTH = os.path.join(ROOT, "transformer_en_de_best.pth")
METRICS_JSON = os.path.join(ROOT, "training_metrics.json")
METRICS_CSV = os.path.join(ROOT, "training_metrics.csv")

D_MODEL = int(os.environ.get("D_MODEL", 512))
N_HEADS = int(os.environ.get("N_HEADS", 8))
D_FF = int(os.environ.get("D_FF", 2048))
N_LAYERS = int(os.environ.get("N_LAYERS", 6))
DROPOUT = float(os.environ.get("DROPOUT", 0.1))
MAX_LEN = int(os.environ.get("MAX_LEN", 128))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS
EPOCHS = int(os.environ.get("EPOCHS", 10))
TRAIN_CHUNK_SIZE = int(os.environ.get("TRAIN_CHUNK_SIZE", 100000))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", 1.0))
SHUFFLE_CHUNKS = os.environ.get("SHUFFLE_CHUNKS", "True") == "True"
CHUNK_SHUFFLE_SEED = int(os.environ.get("CHUNK_SHUFFLE_SEED", 42))
USE_AMP = os.environ.get("USE_AMP", "false").lower() == "true"
USE_LR_SCHEDULER = os.environ.get("USE_LR_SCHEDULER", "true").lower() == "true"
BASE_LR = float(os.environ.get("LR", 5e-4))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 4000))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "true").lower() == "true"
LOG_EVERY = int(os.environ.get("LOG_EVERY", 100))
NONFINITE_TOLERANCE = int(os.environ.get("NONFINITE_TOLERANCE", 10))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("ROOT:", ROOT)
print("DEVICE:", DEVICE)
print("TOKENIZER_PATH:", TOKENIZER_PATH)
print("TOKENIZED_DIR:", TOKENIZED_DIR)
print("USE_AMP:", USE_AMP)
print("USE_LR_SCHEDULER:", USE_LR_SCHEDULER)
print("EPOCHS:", EPOCHS)
print("BATCH_SIZE:", BATCH_SIZE)
print("GRAD_ACCUM_STEPS:", GRAD_ACCUM_STEPS)
print("EFFECTIVE_BATCH_SIZE:", EFFECTIVE_BATCH_SIZE)

if BATCH_SIZE < 1:
    raise ValueError("BATCH_SIZE must be at least 1.")
if GRAD_ACCUM_STEPS < 1:
    raise ValueError("GRAD_ACCUM_STEPS must be at least 1.")
print("TRAIN_CHUNK_SIZE:", TRAIN_CHUNK_SIZE)
print("WARMUP_STEPS:", WARMUP_STEPS)
print("BASE_LR:", BASE_LR)
print("SHUFFLE_CHUNKS:", SHUFFLE_CHUNKS)
print("CHUNK_SHUFFLE_SEED:", CHUNK_SHUFFLE_SEED)


# ============================================================
# Tokenizer / dataset helpers
# ============================================================
def load_tokenizer() -> PreTrainedTokenizerFast:
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file not found at {TOKENIZER_PATH}. "
            "Run train_tokenizer.py first."
        )

    tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    tok.pad_token = "[PAD]"
    tok.unk_token = "[UNK]"
    tok.bos_token = "[BOS]"
    tok.eos_token = "[EOS]"
    return tok


def load_tokenized_datasets():
    if not os.path.exists(TOKENIZED_DIR):
        raise FileNotFoundError(
            f"Tokenized dataset directory not found at {TOKENIZED_DIR}. "
            "Run pretokenize_dataset.py first."
        )

    print("Loading tokenized dataset from disk...")
    ds = load_from_disk(TOKENIZED_DIR)

    required_splits = ["train", "validation", "test"]
    for split in required_splits:
        if split not in ds:
            raise ValueError(f"Missing split '{split}' inside {TOKENIZED_DIR}")

    return ds["train"], ds["validation"], ds["test"]


# ============================================================
# Model
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        bsz = q.size(0)

        q = self.Wq(q).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.Wk(k).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.Wv(v).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return self.Wo(out)


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x2 = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm2(x + self.dropout(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x2 = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.cross_attn(x, enc_out, enc_out, attn_mask=src_mask)
        x = self.norm2(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x


def make_pad_mask(seq, pad_id):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2).bool()


def make_causal_mask(tgt_len, device):
    return torch.tril(torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_id, bos_id, eos_id,
                 d_model=512, n_heads=8, d_ff=2048, num_layers=6, dropout=0.1, max_len=128):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, dropout, max_len)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.proj = nn.Linear(d_model, vocab_size)

    def encode(self, src):
        src_mask = make_pad_mask(src, self.pad_id)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt_in, enc_out, src_mask):
        _, tgt_len = tgt_in.shape
        pad_mask = make_pad_mask(tgt_in, self.pad_id)
        causal = make_causal_mask(tgt_len, tgt_in.device)
        tgt_mask = pad_mask & causal

        x = self.tgt_embed(tgt_in) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x

    def forward(self, src, tgt_in):
        enc_out, src_mask = self.encode(src)
        dec_out = self.decode(tgt_in, enc_out, src_mask)
        return self.proj(dec_out)


# ============================================================
# Training helpers
# ============================================================
def build_model(tokenizer: PreTrainedTokenizerFast):
    return TransformerSeq2Seq(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        num_layers=N_LAYERS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(DEVICE)


def shift_right(labels, bos_id, pad_id):
    bsz, seq_len = labels.shape
    tgt_in = labels.new_full((bsz, seq_len), pad_id)
    tgt_in[:, 0] = bos_id
    tgt_in[:, 1:] = labels[:, :-1]
    return tgt_in


def seq2seq_loss(logits, labels, pad_id):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
    )


def transformer_lr(current_step: int):
    step = max(1, current_step)
    return (D_MODEL ** -0.5) * min(step ** -0.5, step * (WARMUP_STEPS ** -1.5))


def build_optimizer_and_scheduler(model):
    if USE_LR_SCHEDULER:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: transformer_lr(step),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=BASE_LR,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = None
    return optimizer, scheduler


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def build_loader_from_dataset_slice(dataset_slice, tokenizer, shuffle):
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
    )
    return DataLoader(
        dataset_slice,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY and DEVICE.type == "cuda",
    )


def model_has_nonfinite_params(model):
    for name, p in model.named_parameters():
        if p is not None and not torch.isfinite(p).all():
            print(f"Non-finite parameter detected: {name}")
            return True
    return False


@torch.no_grad()
def smoke_test_loss(model, loader, pad_id):
    model.eval()
    try:
        batch = next(iter(loader))
    except StopIteration:
        return False

    src = batch["input_ids"].to(DEVICE, non_blocking=True)
    labels = batch["labels"].to(DEVICE, non_blocking=True)
    tgt_in = shift_right(labels, model.bos_id, pad_id)

    logits = model(src, tgt_in)
    loss = seq2seq_loss(logits, labels, pad_id)
    ok = torch.isfinite(loss).item()
    print(f"Smoke test loss: {loss.item() if torch.isfinite(loss) else 'nan'}")
    return ok


@torch.no_grad()
def evaluate_loss(model, loader, pad_id):
    model.eval()
    total = 0.0
    n = 0
    amp_enabled = (DEVICE.type == "cuda") and USE_AMP

    for batch in loader:
        src = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        tgt_in = shift_right(labels, model.bos_id, pad_id)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(src, tgt_in)
            loss = seq2seq_loss(logits, labels, pad_id)

        if not torch.isfinite(loss):
            print("Non-finite validation/test loss encountered. Returning inf.")
            return float("inf")

        total += loss.item()
        n += 1

    return total / max(1, n)


def train_one_loader(model, loader, optimizer, pad_id, scaler=None, scheduler=None):
    model.train()
    total = 0.0
    n = 0
    amp_enabled = (DEVICE.type == "cuda") and USE_AMP
    nonfinite_streak = 0
    warned_nonfinite = False
    accum_counter = 0
    num_optimizer_steps = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        src = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        tgt_in = shift_right(labels, model.bos_id, pad_id)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(src, tgt_in)
            loss = seq2seq_loss(logits, labels, pad_id)

        if not torch.isfinite(loss):
            nonfinite_streak += 1
            if not warned_nonfinite:
                print("Non-finite loss detected. Suppressing repeated NaN logs.")
                warned_nonfinite = True
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            if nonfinite_streak >= NONFINITE_TOLERANCE:
                raise RuntimeError(
                    f"Encountered {NONFINITE_TOLERANCE} consecutive non-finite losses. "
                    "Stopping this run."
                )
            continue

        nonfinite_streak = 0
        loss_for_backward = loss / GRAD_ACCUM_STEPS

        if scaler is not None and amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        accum_counter += 1

        if accum_counter == GRAD_ACCUM_STEPS:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            num_optimizer_steps += 1

        total += loss.item()
        n += 1

        if batch_idx % LOG_EVERY == 0:
            current_accum = accum_counter if accum_counter > 0 else GRAD_ACCUM_STEPS
            print(
                f"Batch {batch_idx} | LR {get_current_lr(optimizer):.6e} | "
                f"Loss {loss.item():.4f} | AccumStep {current_accum}/{GRAD_ACCUM_STEPS}"
            )

    if accum_counter > 0:
        if scaler is not None and amp_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        num_optimizer_steps += 1

    if n == 0:
        raise RuntimeError("No finite training batches were processed in this loader.")

    return total / max(1, n), num_optimizer_steps


def load_checkpoint_safely(model, optimizer, scheduler, scaler, checkpoint_path, amp_enabled):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    optimizer_state = ckpt.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = ckpt.get("scheduler_state")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = ckpt.get("scaler_state")
    if amp_enabled and scaler is not None and scaler_state:
        try:
            scaler.load_state_dict(scaler_state)
        except RuntimeError as e:
            msg = str(e)
            if "source state dict is empty" in msg or "disabled instance of GradScaler" in msg:
                print("Checkpoint scaler state is empty. Starting with a fresh AMP scaler.")
            else:
                raise

    return ckpt



def make_chunk_ranges(num_examples, chunk_size):
    ranges = []
    for start in range(0, num_examples, chunk_size):
        end = min(start + chunk_size, num_examples)
        ranges.append((start, end))
    return ranges


def chunk_key(start, end):
    return f"{start}:{end}"


def train_one_epoch_chunked(
    model,
    dataset,
    tokenizer,
    optimizer,
    pad_id,
    scaler=None,
    scheduler=None,
    checkpoint_path=None,
    epoch=None,
    best_val_loss=None,
    train_losses=None,
    val_losses=None,
    test_losses=None,
    start_offset=0,
    chunk_losses=None,
    chunk_end_offsets=None,
    completed_chunks=None,
):
    if chunk_losses is None:
        chunk_losses = []
    if chunk_end_offsets is None:
        chunk_end_offsets = []
    if completed_chunks is None:
        completed_chunks = []

    num_examples = len(dataset)
    all_chunk_ranges = make_chunk_ranges(num_examples, TRAIN_CHUNK_SIZE)
    completed_chunk_set = set(completed_chunks)

    pending_chunk_ranges = []
    partial_resume_range = None

    for start, end in all_chunk_ranges:
        key = chunk_key(start, end)
        if key in completed_chunk_set:
            continue
        if start_offset > 0 and start < start_offset < end:
            partial_resume_range = (start_offset, end, start, end)
            continue
        if end <= start_offset:
            continue
        pending_chunk_ranges.append((start, end, start, end))

    if SHUFFLE_CHUNKS:
        rng = random.Random(CHUNK_SHUFFLE_SEED + int(epoch or 0))
        rng.shuffle(pending_chunk_ranges)

    if partial_resume_range is not None:
        pending_chunk_ranges.insert(0, partial_resume_range)

    total_chunks = len(all_chunk_ranges)
    completed_before_epoch = len(completed_chunk_set)
    remaining_chunks = len(pending_chunk_ranges)

    print(
        f"Epoch chunk progress: completed {completed_before_epoch}/{total_chunks} | "
        f"remaining {remaining_chunks}"
    )

    epoch_chunk_losses = []

    if remaining_chunks == 0:
        print("All chunks in this epoch are already complete.")
        return float("nan"), num_examples, chunk_losses, chunk_end_offsets, completed_chunks

    processed_this_call = 0

    for current_start, current_end, original_start, original_end in pending_chunk_ranges:
        processed_this_call += 1
        global_chunk_idx = all_chunk_ranges.index((original_start, original_end)) + 1

        print(
            f"Training chunk {processed_this_call}/{remaining_chunks} "
            f"(global {global_chunk_idx}/{total_chunks}) | "
            f"{current_start}:{current_end} / {num_examples}"
        )

        dataset_slice = dataset.select(range(current_start, current_end))
        train_loader = build_loader_from_dataset_slice(dataset_slice, tokenizer, shuffle=True)
        chunk_loss, chunk_optimizer_steps = train_one_loader(
            model, train_loader, optimizer, pad_id, scaler=scaler, scheduler=scheduler
        )

        full_key = chunk_key(original_start, original_end)
        if full_key not in completed_chunk_set:
            completed_chunks.append(full_key)
            completed_chunk_set.add(full_key)

        chunk_losses.append(chunk_loss)
        chunk_end_offsets.append(original_end)
        epoch_chunk_losses.append(chunk_loss)

        if checkpoint_path is not None:
            ckpt = {
                "epoch": epoch,
                "resume_offset": current_end if current_start != original_start else original_end,
                "samples_trained_so_far": len(completed_chunk_set) * TRAIN_CHUNK_SIZE,
                "completed_chunks": completed_chunks,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses if train_losses is not None else [],
                "val_losses": val_losses if val_losses is not None else [],
                "test_losses": test_losses if test_losses is not None else [],
                "chunk_losses": chunk_losses,
                "chunk_end_offsets": chunk_end_offsets,
                "completed_chunks": completed_chunks,
                "lr": get_current_lr(optimizer),
                "batch_size": BATCH_SIZE,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "use_amp": USE_AMP,
                "use_lr_scheduler": USE_LR_SCHEDULER,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "shuffle_chunks": SHUFFLE_CHUNKS,
                "chunk_shuffle_seed": CHUNK_SHUFFLE_SEED,
            }
            torch.save(ckpt, checkpoint_path)

        del train_loader, dataset_slice
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"  chunk_loss = {chunk_loss:.4f} | "
            f"lr = {get_current_lr(optimizer):.6e} | "
            f"batch_size = {BATCH_SIZE} | "
            f"effective_batch_size = {EFFECTIVE_BATCH_SIZE} | "
            f"optimizer_steps = {chunk_optimizer_steps}"
        )

    if len(epoch_chunk_losses) == 0:
        raise RuntimeError("No chunk was processed in this epoch.")

    return (
        sum(epoch_chunk_losses) / len(epoch_chunk_losses),
        num_examples,
        chunk_losses,
        chunk_end_offsets,
        completed_chunks,
    )



def save_metrics_files(train_losses, val_losses, test_losses):
    metrics_rows = []
    num_epochs = max(len(train_losses), len(val_losses), len(test_losses))
    for i in range(num_epochs):
        metrics_rows.append({
            "epoch": i + 1,
            "train_loss": float(train_losses[i]) if i < len(train_losses) else None,
            "val_loss": float(val_losses[i]) if i < len(val_losses) else None,
            "test_loss": float(test_losses[i]) if i < len(test_losses) else None,
        })

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)

    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "test_loss"])
        writer.writeheader()
        writer.writerows(metrics_rows)


def save_best_model_pth(model):
    torch.save(model.state_dict(), BEST_MODEL_PTH)
    print(f"Saved best model .pth to: {BEST_MODEL_PTH}")


def save_final_model_pth_from_best_or_current(model):
    if os.path.exists(BEST_CKPT):
        best_ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
        model.load_state_dict(best_ckpt["model_state"])
        print("Loaded BEST_CKPT weights for final .pth export.")
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"Saved final model .pth to: {FINAL_MODEL_PATH}")


# ============================================================
# Main
# ============================================================
def main():
    tokenizer = load_tokenizer()
    train_tokens, val_tokens, test_tokens = load_tokenized_datasets()

    print("Train examples:", len(train_tokens))
    print("Validation examples:", len(val_tokens))
    print("Test examples:", len(test_tokens))

    val_loader = build_loader_from_dataset_slice(val_tokens, tokenizer, shuffle=False)
    test_loader = build_loader_from_dataset_slice(test_tokens, tokenizer, shuffle=False)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    model = build_model(tokenizer)
    optimizer, scheduler = build_optimizer_and_scheduler(model)

    amp_enabled = (DEVICE.type == "cuda") and USE_AMP
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    resume_offset = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    test_losses = []
    chunk_losses = []
    chunk_end_offsets = []
    completed_chunks = []
    samples_trained_so_far = 0

    if os.path.exists(LATEST_CKPT):
        ckpt = load_checkpoint_safely(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=LATEST_CKPT,
            amp_enabled=amp_enabled,
        )

        if model_has_nonfinite_params(model):
            raise RuntimeError("Latest checkpoint weights contain non-finite values.")

        start_epoch = ckpt.get("epoch", 1)
        resume_offset = ckpt.get("resume_offset", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        test_losses = ckpt.get("test_losses", [])
        chunk_losses = ckpt.get("chunk_losses", [])
        chunk_end_offsets = ckpt.get("chunk_end_offsets", [])
        completed_chunks = ckpt.get("completed_chunks", [])
        samples_trained_so_far = ckpt.get("samples_trained_so_far", resume_offset)

        print(f"Loaded full checkpoint from: {LATEST_CKPT}")
        print("Restored model / optimizer / scheduler / scaler state.")
        print(f"Epoch to resume: {start_epoch}")
        print(f"Resume offset within epoch: {resume_offset}")
        print(f"Samples trained so far: {samples_trained_so_far}")
        print(f"Current LR: {get_current_lr(optimizer):.6e}")
        print(f"Checkpoint batch_size: {ckpt.get('batch_size', 'N/A')}")
        print(f"Checkpoint effective_batch_size: {ckpt.get('effective_batch_size', 'N/A')}")
        print(f"Completed chunks in current epoch: {len(completed_chunks)}")

        smoke_slice_end = min(max(resume_offset + BATCH_SIZE * 2, BATCH_SIZE * 2), len(train_tokens))
        smoke_slice = train_tokens.select(range(resume_offset, smoke_slice_end))
        smoke_loader = build_loader_from_dataset_slice(smoke_slice, tokenizer, shuffle=False)
        smoke_ok = smoke_test_loss(model, smoke_loader, tokenizer.pad_token_id)
        del smoke_loader, smoke_slice
        gc.collect()

        if not smoke_ok:
            raise RuntimeError("Latest checkpoint weights produce non-finite smoke-test loss.")

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch > start_epoch:
            resume_offset = 0

        print(f"\n========== EN→DE Epoch {epoch}/{EPOCHS} ==========")
        if resume_offset > 0:
            print(f"Resuming from training example offset {resume_offset}")

        tr, final_offset, chunk_losses, chunk_end_offsets, completed_chunks = train_one_epoch_chunked(
            model=model,
            dataset=train_tokens,
            tokenizer=tokenizer,
            optimizer=optimizer,
            pad_id=tokenizer.pad_token_id,
            scaler=scaler,
            scheduler=scheduler,
            checkpoint_path=LATEST_CKPT,
            epoch=epoch,
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            test_losses=test_losses,
            start_offset=resume_offset,
            chunk_losses=chunk_losses,
            chunk_end_offsets=chunk_end_offsets,
            completed_chunks=completed_chunks,
        )

        samples_trained_so_far = len(train_tokens)
        completed_chunks = []
        va = evaluate_loss(model, val_loader, tokenizer.pad_token_id)
        te = evaluate_loss(model, test_loader, tokenizer.pad_token_id)

        train_losses.append(tr)
        val_losses.append(va)
        test_losses.append(te)
        save_metrics_files(train_losses, val_losses, test_losses)

        if va < best_val_loss:
            best_val_loss = va
            torch.save({
                "epoch": epoch + 1,
                "resume_offset": 0,
                "samples_trained_so_far": samples_trained_so_far,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict() if amp_enabled else None,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "test_losses": test_losses,
                "chunk_losses": chunk_losses,
                "chunk_end_offsets": chunk_end_offsets,
                "lr": get_current_lr(optimizer),
                "batch_size": BATCH_SIZE,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "use_amp": USE_AMP,
                "use_lr_scheduler": USE_LR_SCHEDULER,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
            }, BEST_CKPT)
            print(f"Saved new best checkpoint to: {BEST_CKPT}")
            save_best_model_pth(model)

        torch.save({
            "epoch": epoch + 1,
            "resume_offset": 0,
            "samples_trained_so_far": samples_trained_so_far,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_losses": test_losses,
            "chunk_losses": chunk_losses,
            "chunk_end_offsets": chunk_end_offsets,
            "completed_chunks": completed_chunks,
            "lr": get_current_lr(optimizer),
            "batch_size": BATCH_SIZE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "use_amp": USE_AMP,
            "use_lr_scheduler": USE_LR_SCHEDULER,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
        }, LATEST_CKPT)

        print(
            f"[EN→DE] Epoch {epoch}/{EPOCHS} | "
            f"train {tr:.4f} | val {va:.4f} | test {te:.4f} | "
            f"lr {get_current_lr(optimizer):.6e} | "
            f"batch_size {BATCH_SIZE} | "
            f"effective_batch_size {EFFECTIVE_BATCH_SIZE}"
        )

    save_final_model_pth_from_best_or_current(model)
    save_metrics_files(train_losses, val_losses, test_losses)
    print(f"\nLatest checkpoint: {LATEST_CKPT}")
    print(f"Best checkpoint:   {BEST_CKPT}")
    print(f"Best model .pth:   {BEST_MODEL_PTH}")
    print(f"Final model:       {FINAL_MODEL_PATH}")
    print(f"Metrics JSON:      {METRICS_JSON}")
    print(f"Metrics CSV:       {METRICS_CSV}")
    print(f"Samples trained so far: {samples_trained_so_far}")


if __name__ == "__main__":
    main()
