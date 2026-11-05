import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_post_training.data import build_prompt, extract_dataset_final_answer, load_gsm8k
from llm_post_training.parsing import extract_model_answer, is_format_compliant

# MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_NAME = "outputs/SFT/models/qwen3-0.6b-gsm8k-sft-raw/final"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    return tokenizer, model


@torch.no_grad()
def generate_answers(model, tokenizer, questions: list[str], max_new_tokens: int = 512) -> list[str]:
    prompts = [build_prompt(question) for question in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # Match the native EOS supervised by the raw-completion SFT data.
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, eos_token_id=tokenizer.eos_token_id,
    )

    # 只留生成的，去除输入prompt（包含左侧padding）
    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return responses


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    return generate_answers(model, tokenizer, [question], max_new_tokens=max_new_tokens)[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the baseline model on GSM8K.")
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of test samples to evaluate (default: 1).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of samples per inference batch (default: 1).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/baseline_eval.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num_samples must be greater than 0")
    if args.batch_size <= 0:
        parser.error("--batch_size must be greater than 0")

    dataset = load_gsm8k()
    samples = dataset["test"]
    num_samples = min(args.num_samples, len(samples))
    tokenizer, model = load_model()

    correct = total = parse_fail = format_compliant_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file, tqdm(
        total=num_samples, desc="Evaluating", unit="sample",
    ) as progress:
        for start in range(0, num_samples, args.batch_size):
            batch = [samples[index] for index in range(start, min(start + args.batch_size, num_samples))]
            responses = generate_answers(
                model=model, tokenizer=tokenizer,
                questions=[sample["question"] for sample in batch],
            )
            for sample, response in zip(batch, responses, strict=True):
                question = sample["question"]
                gold_answer = extract_dataset_final_answer(sample["answer"])
                pred_answer = extract_model_answer(response)
                parse_success = pred_answer is not None
                is_correct = parse_success and pred_answer == gold_answer
                format_compliant = is_format_compliant(response)

                total += 1
                correct += int(is_correct)
                parse_fail += int(not parse_success)
                format_compliant_count += int(format_compliant)
                result = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_response": response,
                    "pred_answer": pred_answer,
                    "correct": is_correct,
                    "parse_success": parse_success,
                    "format_compliant": format_compliant,
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            progress.update(len(batch))

    # Parse success includes answers extracted by the final-number fallback.
    metrics = {
        "total": total,
        "correct": correct,
        "parse_success": total - parse_fail,
        "parse_fail": parse_fail,
        "format_compliant": format_compliant_count,
        "answer_accuracy": correct / total if total else 0.0,
        "parse_success_rate": (total - parse_fail) / total if total else 0.0,
        "format_compliance_rate": format_compliant_count / total if total else 0.0,
    }
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"correct: {correct} / total: {total} / "
        f"parse_fail: {parse_fail}"
    )
    for name in ("answer_accuracy", "parse_success_rate", "format_compliance_rate"):
        print(f"{name}: {metrics[name]:.4f}")
    print(f"Results saved to: {args.output}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

