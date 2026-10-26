import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_post_training.data import extract_final_answer, load_gsm8k

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
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
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    prompt = bulid_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # **字典解包传参  do_sample=False贪心解码
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # 只留生成的，去除输入prompt
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response


def extract_model_answer(response: str) -> str | None:
    # 优先匹配要求的 GSM8K 格式：#### 18
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", response)
    if match:
        return match.group(1).replace(",", "")

    # 匹配 LaTeX boxed：\boxed{18}
    matches = re.findall(r"\\boxed\{(-?[\d,]+(?:\.\d+)?)\}", response)
    if matches:
        return matches[-1].replace(",", "")

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate the baseline model on GSM8K.")
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of test samples to evaluate (default: 1).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/baseline_eval.jsonl"),
        help="Output JSONL path.",
    )
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num_samples must be greater than 0")

    dataset = load_gsm8k()
    samples = dataset["test"]
    num_samples = min(args.num_samples, len(samples))
    tokenizer, model = load_model()

    correct = total = parse_fail = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        for index in tqdm(range(num_samples), desc="Evaluating", unit="sample"):
            sample = samples[index]
            question = sample["question"]
            gold_answer = extract_final_answer(sample["answer"])
            response = generate_answer(model=model, tokenizer=tokenizer, question=question)
            pred_answer = extract_model_answer(response)
            is_correct = pred_answer is not None and pred_answer == gold_answer

            total += 1
            correct += int(is_correct)
            parse_fail += int(pred_answer is None)
            result = {
                "question": question,
                "gold_answer": gold_answer,
                "model_response": response,
                "pred_answer": pred_answer,
                "correct": is_correct,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()

    accuracy = correct / total if total else 0.0
    print(
        f"correct: {correct} / total: {total} / "
        f"parse_fail: {parse_fail} / accuracy: {accuracy:.4f}"
    )
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

