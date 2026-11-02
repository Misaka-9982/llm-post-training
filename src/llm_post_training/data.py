from datasets import load_dataset


def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset


def extract_final_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


# 分开返回reasoning和final answer
def parse_gsm8k_answer(answer: str) -> dict[str, str]:
    # 从右开始切分
    reasoning, final_answer = answer.rsplit("####", maxsplit=1)

    return {
        "reasoning": reasoning.strip(),
        "final_answer": final_answer.strip(),
    }


def preprocess_example(example: dict) -> dict:
    parsed = parse_gsm8k_answer(example["answer"])

    assistant_content = (
        f"{parsed['reasoning']}\n"
        f"#### {parsed['final_answer']}"
    )

    return {
        "messages": [
            {
                "role": "user",
                "content": example["question"],
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "final_answer": parsed["final_answer"],
    }


# 把预处理后的样本处理成sft chat template的格式
def format_sft_example(preprocessed_example):
    messages = preprocessed_example["messages"]

    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]

    return (
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_content}<|im_end|>\n"
    )


def tokenize_sft_example(example, tokenizer):
    text = format_sft_example(example)

    # 只到 assistant 内容开始之前
    prompt_text = (
        "<|im_start|>user\n"
        f"{example['messages'][0]['content']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    full = tokenizer(
        text,
        add_special_tokens=False,
    )

    prompt = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )

    input_ids = full["input_ids"]
    labels = input_ids.copy()

    prompt_len = len(prompt["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


if __name__ == "__main__":
    dataset = load_gsm8k()

    print(dataset)
    print(dataset["train"][0])
    print(dataset["test"][0])

    # 测试提取最终答案
    sample = dataset["train"][0]
    print("Question:")
    print(sample["question"])

    print("\nRaw answer:")
    print(sample["answer"])

    print("\nFinal answer:")
    print(extract_final_answer(sample["answer"]))

    print('\nPreprocessed preview: ')
    processed_dataset = dataset['train'].map(preprocess_example)
    for i in range(3):
        print(processed_dataset[i])


    print('\nChat Templete test')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base"
    )

    messages = processed_dataset[0]["messages"]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
    )

    print('qwen官方的chat template效果：\n'+formatted_text)

    print('自定义的chat template应用函数：\n'+format_sft_example(processed_dataset[0]))

    print('Tokenization+mask测试：\n')
    
    example = processed_dataset[0]
    tokenized = tokenize_sft_example(example, tokenizer)

    print("input length:", len(tokenized["input_ids"]))
    print("labels length:", len(tokenized["labels"]))

    print(tokenizer.decode(tokenized["input_ids"]))

    first_label_idx = next(
        i for i, x in enumerate(tokenized["labels"])
        if x != -100
    )

    print("first supervised token index:", first_label_idx)

    print(
        "supervised text:",
        tokenizer.decode(tokenized["input_ids"][first_label_idx:])
    )

    