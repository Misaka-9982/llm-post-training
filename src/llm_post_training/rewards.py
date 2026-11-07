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


# 条件触发，只有答案正确才给格式奖励
def gated_format_reward(completions, final_answer, **kwargs) -> list[float]:
    rewards = []

    for completion, target in zip(completions, final_answer):
        text = _get_completion_text(completion)
        prediction = extract_reward_answer(text)
        target = normalize_number(target)

        is_correct = prediction == target
        is_format_ok = is_format_compliant(text)

        reward = 1.0 if is_correct and is_format_ok else 0.0
        rewards.append(reward)

    return rewards


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
    test_completion = ["The answer is 90", "The answer is 93", "#### 90", "#### 93"]
    print(correctness_reward(test_completion, ["90", "90", "90", "90"]))
    print(gated_format_reward(test_completion, ["90", "90", "90", "90"]))
    