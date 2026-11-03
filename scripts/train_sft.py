from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer


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

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.config.use_cache = False  # 禁用kvcache

    training_args = TrainingArguments(
        output_dir="outputs/qwen3-0.6b-gsm8k-sft-raw-lr1e-5-ep1",

        num_train_epochs=1,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=1e-5,
        weight_decay=0.01,

        bf16=True,

        logging_steps=10,
        save_steps=200,
        save_total_limit=2,

        report_to="none",

        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(
        "outputs/qwen3-0.6b-gsm8k-sft-raw-lr1e-5-ep1/final"
    )
    tokenizer.save_pretrained(
        "outputs/qwen3-0.6b-gsm8k-sft-raw-lr1e-5-ep1/final"
    )


if __name__ == "__main__":
    main()