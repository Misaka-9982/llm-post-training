from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from llm_post_training.data import preprocess_grpo_example
from llm_post_training.rewards import correctness_reward, format_reward



def main():
    model_path = "outputs/SFT/models/qwen3-0.6b-gsm8k-sft-raw/final"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    dataset = load_dataset("openai/gsm8k", "main", split="train")

    dataset = dataset.map(
        preprocess_grpo_example,
        remove_columns=dataset.column_names,
    )

    args = GRPOConfig(
        output_dir="outputs/GRPO/qwen3-0.6b-gsm8k-grpo-debug",

        # debug
        max_steps=10,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        log_completions=True,
        num_completions_to_print=8,

        # GRPO sampling
        num_generations=8,
        max_completion_length=256,
        temperature=1.0,

        # optimization
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,

        # reward
        reward_weights=[1.0, 0.2],

        # KL
        beta=0.0,

        # A100
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        reward_funcs=[
            correctness_reward,
            format_reward,
        ],
    )

    trainer.train()


if __name__ == "__main__":
    main()