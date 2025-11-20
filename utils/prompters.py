def get_prompter(model, dataset, seed, reasoning=False, sampling=True):
    if model in ["InternVL3-8B", "InternVL3-14B", "InternVL3-78B"]:
        from models.internvl.main import InternVLPrompter

        return InternVLPrompter(
            model=model, dataset=dataset, seed=seed, sampling=sampling
        )
    elif model in ["Qwen2.5-VL-7B-Instruct"]:
        from models.qwen.main import Qwen2_5Prompter

        return Qwen2_5Prompter(
            model=model, dataset=dataset, seed=seed, sampling=sampling
        )
    elif model in ["Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL-30B-A3B-Thinking"]:
        from models.qwen3.main import Qwen3Prompter

        return Qwen3Prompter(model=model, dataset=dataset, seed=seed, sampling=sampling)
    elif model in ["Kimi-VL-A3B-Thinking-2506", "Kimi-VL-A3B-Instruct"]:

        from models.kimi.main import KimiPrompter

        return KimiPrompter(model=model, dataset=dataset, seed=seed, sampling=sampling)

    elif (
        model == "gpt-5-mini"
        or model == "gpt-4o"
        or model == "gpt-5"
        or model == "gpt-5-chat-latest"
    ):
        from models.gpt.main import GPTPrompter

        return GPTPrompter(model=model, dataset=dataset, seed=seed, reasoning=reasoning)
    else:
        raise ValueError(f"Model {model} not supported.")
