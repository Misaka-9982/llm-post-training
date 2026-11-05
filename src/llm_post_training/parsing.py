import re


def normalize_number(s: str) -> str:
    return s.replace(",", "").strip()


# eval评估时用
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
    
    # fallback: 只看输出末尾最后一个数字
    tail = response[-300:]
    numbers = re.findall(
        r"-?[\d,]+(?:\.\d+)?",
        tail,
    )
    if numbers:
        return normalize_number(numbers[-1])

    return None


def is_format_compliant(response: str) -> bool:
    return re.search(
        r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)",
        response,
    ) is not None


# GRPO奖励函数用，去掉fallback和加粗规则，更严格
def extract_reward_answer(response: str) -> str | None:
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

    # 3. Final Answer: 18 / Answer: 18
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

    return None

