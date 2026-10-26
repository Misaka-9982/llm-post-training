import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_post_training.data import load_gsm8k, extract_final_answer


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
    dataset = load_gsm8k()
    tokenizer, model = load_model() 

    # 加载测试数据集
    sample = dataset["test"][0]

    question = sample["question"]
    gold_answer = extract_final_answer(sample["answer"])

    response = generate_answer(model=model, tokenizer=tokenizer, question=question)

    print("Question:")
    print(question)

    print("\nGold answer:")
    print(gold_answer)

    print("\nModel response:")
    print(response)

    pred_answer = extract_model_answer(response)

    print("\nPredicted answer:")
    print(pred_answer)

    print("\nCorrect:")
    print(pred_answer == gold_answer)


if __name__ == "__main__":
    main()

