import os
import re

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str, **kwargs):
    from model.monkeypatch import replace_attn

    replace_attn(model_id)

    config = AutoConfig.from_pretrained(model_id)
    # **本地新增开关 `VARIKV_NO_YARN=1`（默认不设 ⇒ 行为与上游逐字节相同）。**
    # 动机：上游对 `"Qwen3-" in id and "Instruct" not in id` **无条件**开 YaRN×4，
    # 但我们的 Qwen3 实验上下文只有 24,470，**远在原生 40,960 之内** ——
    # YaRN 被无谓开启，它重标定全部位置的 RoPE、已知会轻微损伤短上下文，
    # 于是成为跨 backbone 对照里一个**未受控变量**。这个开关用来把它控住。
    # **⚠ 只有在 clen < 原生 max_position_embeddings 时才可以关**，否则会越界；
    # 下面把两个数都打进日志，判读时必须核对。
    _no_yarn = os.environ.get("VARIKV_NO_YARN") == "1"
    if "Qwen3-" in model_id and "Instruct" not in model_id:
        _native = int(getattr(config, "max_position_embeddings", 0))
        if _no_yarn:
            print(f"[rope] **VARIKV_NO_YARN=1 ⇒ 不开 YaRN**；"
                  f"native max_position_embeddings={_native} rope_scaling={config.rope_scaling}",
                  flush=True)
        else:
            config.rope_scaling = {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
            config.max_position_embeddings = 131072
            print(f"[rope] YaRN x4 已开（上游默认）；native was {_native} "
                  f"⇒ max_position_embeddings=131072", flush=True)
            print("Max context length extended")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if "llama" in model_id.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004

    if "gemma-3" in model_id.lower():
        model = model.language_model

    model.eval()
    model.name = model_id.split("/")[-1].lower()
    model.name_or_path = model_id
    print(f"\nLoad {model_id} with {model.dtype}")
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="llama3-8b")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.name)
    print(model)

    messages = [
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        }
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(input_text)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_new_tokens=30)
    print(tokenizer.decode(outputs[0]))
