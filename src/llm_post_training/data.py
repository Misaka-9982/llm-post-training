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
    