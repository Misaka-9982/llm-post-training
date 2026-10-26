import torch
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
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 256) -> str:
    prompt = bulid_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # **字典解包传参
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # 只留生成的，去除输入prompt
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response


def main():
    pass


if __name__ == "__main__":
    pass

