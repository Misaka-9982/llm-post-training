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


def preprocess_sft_example(example: dict) -> dict:
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


def preprocess_grpo_example(example: dict) -> dict:
    parsed = parse_gsm8k_answer(example["answer"])

    return {
        "prompt": [
            {
                "role": "user",
                "content": build_prompt(example["question"]),
            }
        ],
        "final_answer": parsed["final_answer"],
    }


def build_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Solve the problem step by step. "
        "Give the final answer after ####.\n"
        "Answer:"
    )


def format_sft_example(preprocessed_example, eos_token: str = "<|endoftext|>"):
    messages = preprocessed_example["messages"]
    return build_prompt(messages[0]["content"]) + messages[1]["content"] + eos_token


def tokenize_sft_example(example, tokenizer):
    # Separately encode the inference prefix so its token boundary is exact.
    prompt_ids = tokenizer(
        build_prompt(example["messages"][0]["content"]),
        add_special_tokens=False,
    )["input_ids"]
    response_ids = tokenizer(
        example["messages"][1]["content"],
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("SFT requires tokenizer.eos_token_id")
    response_ids = response_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + response_ids,
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
    processed_dataset = dataset['train'].map(preprocess_sft_example)
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

    print('qwen官方的chat template效果: \n'+formatted_text)

    print('自定义的chat template应用函数: \n'+format_sft_example(processed_dataset[0]))

    print('Tokenization+mask测试: \n')
    
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

    print('padding测试: ')
    from transformers import DataCollatorForSeq2Seq

    tokenized_train = processed_dataset.map(
    lambda example: tokenize_sft_example(example, tokenizer),
    remove_columns=processed_dataset.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    batch = data_collator(
        [tokenized_train[i] for i in range(3)]
    )

    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)
    
    print(batch["input_ids"][0])
    print(batch["attention_mask"][0])
    print(batch["labels"][0])

    print('\nEOS检查')
    example = tokenized_train[0]

    print(tokenizer.decode(example["input_ids"]))
    print(example["input_ids"][-10:])
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)
    print("last token:", example["input_ids"][-1])

    print('\nJanet最终格式化输出')
    for example in tokenized_train:
        text = tokenizer.decode(example["input_ids"], skip_special_tokens=False)

        if "Janet" in text:
            print(text+'\n')