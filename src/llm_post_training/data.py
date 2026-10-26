from datasets import load_dataset


def load_gsm8k():
    dataset = load_dataset("openai/gsm8k", "main")
    return dataset


def extract_final_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


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

