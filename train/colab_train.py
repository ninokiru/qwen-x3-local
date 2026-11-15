# Notebook train LoRA trên Google Colab (GPU T4). Dán từng khối "# %%" vào một cell.
# Dữ liệu và kết quả lưu trên Google Drive, vì runtime Colab sẽ bị xóa khi ngắt.
#
# Dùng bản safetensors (bf16) của model abliterated, KHÔNG dùng file GGUF.
# Kiểm tra lại tên repo, phiên bản thư viện và tham số trước khi chạy đầy đủ.

# %% Cell 1: cài đặt và kiểm tra GPU
# !pip install -q unsloth
# !nvidia-smi

# %% Cell 2: mount Drive và cấu hình đường dẫn
from google.colab import drive
drive.mount("/content/drive")

DRIVE = "/content/drive/MyDrive/qwen-x3"
MODEL = "REPO_ID_CUA_BAN"        # ví dụ: <tác giả>/Qwen2.5-3B-Instruct-abliterated (bản safetensors)
TRAIN_FILE = f"{DRIVE}/data/train.jsonl"
ADAPTER_NAME = "agent"           # đổi thành "code" khi train adapter code
MAX_SEQ = 2048

# %% Cell 3: nạp model và gắn LoRA
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

# %% Cell 4: đọc và format dữ liệu theo chat template của Qwen2.5
from datasets import load_dataset

ds = load_dataset("json", data_files=TRAIN_FILE, split="train")

def to_text(ex):
    # apply_chat_template với tools=None giữ nguyên định dạng <tool_call> đã dùng trong dữ liệu
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(to_text, remove_columns=ds.column_names)
print(ds[0]["text"][:500])

# %% Cell 5: train
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        output_dir=f"{DRIVE}/ckpt/{ADAPTER_NAME}",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        warmup_steps=20,
        fp16=True,                 # T4 không hỗ trợ bf16
        optim="adamw_8bit",
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
    ),
)
trainer.train()

# %% Cell 6: lưu LoRA (bản adapter, dùng để tiếp tục train hoặc xuất sau)
model.save_pretrained(f"{DRIVE}/lora/{ADAPTER_NAME}")
tokenizer.save_pretrained(f"{DRIVE}/lora/{ADAPTER_NAME}")

# %% Cell 7: xuất GGUF để chạy local
# Cách 1 (đơn giản): merge và lượng tử hóa thành Q4_K_M, dùng làm base hoặc chạy độc lập.
# model.save_pretrained_gguf(f"{DRIVE}/gguf/{ADAPTER_NAME}", tokenizer, quantization_method="q4_k_m")
#
# Cách 2 (hot-swap LoRA trên base Q4_K_M): chuyển adapter sang GGUF bằng
# convert_lora_to_gguf.py của llama.cpp rồi copy vào models/lora/ trên máy local.
