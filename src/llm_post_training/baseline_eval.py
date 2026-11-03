import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_post_training.data import extract_final_answer, load_gsm8k

# MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
MODEL_NAME = "outputs/qwen3-0.6b-gsm8k-sft/final"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

    return tokenizer, model


def bulid_prompt(question: str) -> str:
    # 圆括号自动跨行拼接成一个长str
    return (
        f"Question: {question}\n"
        "Solve the problem step by step. "
        "Give the final answer after ####.\n"
        "Answer:"
    )


@torch.no_grad()
def generate_answers(model, tokenizer, questions: list[str], max_new_tokens: int = 512) -> list[str]:
    prompts = [bulid_prompt(question) for question in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # **字典解包传参  do_sample=False贪心解码
    # sft的时候使用<|im_end|>作为中止符号，不加入<|im_end|>会导致输出无法正常结束
    eos_token_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, 
                             do_sample=False, eos_token_id=eos_token_ids)

    # 只留生成的，去除输入prompt（包含左侧padding）
    generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return responses


def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    return generate_answers(model, tokenizer, [question], max_new_tokens=max_new_tokens)[0]


def normalize_number(s: str) -> str:
    return s.replace(",", "").strip()


def extract_model_answer(response: str) -> str | None:
    # 1. GSM8K-style: #### 18
    match = re.search(
        r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_number(match.group(1))

    # 2. LaTeX boxed: \boxed{18}
    matches = re.findall(
        r"\\boxed\{\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*\}",
        response,
        flags=re.IGNORECASE,
    )
    if matches:
        return normalize_number(matches[-1])

    # 3. Final Answer: 18 / Answer: **18 bolts**
    matches = re.findall(
        r"(?:final\s+answer|answer)\s*:\s*(?:\*\*)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if matches:
        return normalize_number(matches[-1])

    # 4. "the answer is 18"
    matches = re.findall(
        r"(?:the\s+)?answer\s+is\s*(?:\*\*)?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    )
    if matches:
        return normalize_number(matches[-1])

    # 5. **about 35 hours**
    bold_spans = re.findall(r"\*\*(.+?)\*\*", response)
    for span in reversed(bold_spans):
        match = re.search(
            r"\$?\s*(-?[\d,]+(?:\.\d+)?)",
            span,
        )
        if match:
            return normalize_number(match.group(1))
    
    # fallback: 只看输出末尾
    tail = response[-300:]
    numbers = re.findall(
        r"-?[\d,]+(?:\.\d+)?",
        tail,
    )
    if numbers:
        return normalize_number(numbers[-1])

    return None


def is_format_compliant(response: str) -> bool:
    # Use the same pattern as the first answer-extraction rule.
    return re.search(
        r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        response,
        flags=re.IGNORECASE,
    ) is not None


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
                gold_answer = extract_final_answer(sample["answer"])
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

