from llm_post_training.parsing import extract_reward_answer, is_format_compliant, normalize_number


def _get_completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


# GRPOTrainer调用时会传入额外关键字参数，保留**kwargs兼容
def correctness_reward(completions, final_answer, **kwargs) -> list[float]:
    rewards = []

    for completion, target in zip(completions, final_answer):
        text = _get_completion_text(completion)
        prediction = extract_reward_answer(text)
        target = normalize_number(target)

        reward = 1.0 if prediction == target else 0.0
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    return [
        1.0 if is_format_compliant(_get_completion_text(completion)) else 0.0
        for completion in completions
    ]


if __name__ == "__main__":
    print('测试reward函数')
    test_completion = [
        [{
            "role": "user",
            "content": "The answer is 90",
        }],
        [{
            "role": "user",
            "content": "The answer is 93",
        }],
        [{
            "role": "user",
            "content": "#### 90",
        }],
        [{
            "role": "user",
            "content": "#### 93",
        }]
    ]
    print(correctness_reward(test_completion, ["90", "90", "90", "90"]))
    print(format_reward(test_completion))
    