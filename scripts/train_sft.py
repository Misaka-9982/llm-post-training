from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from llm_post_training.data import preprocess_example, tokenize_sft_example


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = load_dataset("openai/gsm8k", "main")

    processed_train = dataset["train"].map(
        preprocess_example,
    )

    tokenized_train = processed_train.map(
        lambda example: tokenize_sft_example(example, tokenizer),
        remove_columns=processed_train.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    # 1. load model
    # 2. 配 LoRA / full fine-tuning
    # 3. TrainingArguments / SFTConfig
    # 4. SFTTrainer
    # 5. trainer.train()

if __name__ == "__main__":
    main()