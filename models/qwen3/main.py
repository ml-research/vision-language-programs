from datetime import datetime
import re
from typing import Dict, Tuple
from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import os
import json
import torch
import gc
from PIL import Image
from io import BytesIO


class Qwen3Prompter:
    def __init__(
        self,
        model="Qwen/Qwen3-VL-30B-A3B-Instruct",
        dataset=None,
        seed=0,
        sampling=False,
    ):

        # set seed
        # print("set seed ", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed

        if model == "Qwen3-VL-30B-A3B-Instruct":
            self.model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        elif model == "Qwen3-VL-30B-A3B-Thinking":
            self.model_name = "Qwen/Qwen3-VL-30B-A3B-Thinking"
        else:
            raise ValueError(
                f"Model {model} not supported. Use 'Qwen3-VL-30B-A3B-Instruct'."
            )

        self.device = "auto"
        self.model_loaded = False
        self.memory = {}  # Dictionary to store (prompt, image_path) -> response
        self.generated_response = {}
        self.produced_tokens = 0

        if "bongard-op" in dataset:
            dataset = "bongard-op"

        self.sampling = sampling

        if not self.sampling:
            print("Using greedy decoding (no sampling) for VLM.")
            model_name = model + "_no_sampling"
        else:
            model_name = model + "_sampling"

        self.memory_file = (
            f"models/qwen/memory/{dataset}/vlm_memory_{model_name}_{seed}.json"
        )
        self._load_memory()

    def _load_model(self):
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            self.model_name, dtype="auto", device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model_loaded = True

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                # Convert string keys back to tuples
                self.memory = {eval(k): v for k, v in data.items()}
        else:
            # create the folder if it does not exist
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

    def _save_memory(self):
        memory_dict = {str(k): v for k, v in self.memory.items()}
        memory_as_json = json.dumps(memory_dict, indent=4)

        with open(self.memory_file, "w") as f:
            f.write(memory_as_json)

    def remove_from_gpu(self):
        if self.model_loaded:
            print("Removing model from GPU...")
            del self.model
            del self.processor
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            self.model_loaded = False

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    def _preprocess_images(self, paths, max_size=224):

        images = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            img.thumbnail((max_size, max_size))
            images.append(img)
        return images

    def prompt_with_text(self, prompt_text, use_memory=True, max_new_tokens=128):

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if use_memory:
            key = (prompt_text, "")
            if key in self.memory:
                return self.memory[key]

        if not self.model_loaded:
            self._load_model()

        # print(f"Prompting model...")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=self.sampling
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if use_memory:
            self.memory[key] = output_text[0] if output_text else ""
            self._save_memory()

        return output_text[0] if output_text else ""

    def prompt_with_images(
        self,
        prompt_text,
        paths,
        url=False,
        use_memory=True,
        max_new_tokens=128,
        overwrite_memory=False,
        sampling=False,
        thinking=False,
    ):

        # print("Use sampling:", self.sampling)
        # overwrite_memory = True

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if use_memory:
            path_string = ",".join(paths)
            key = (prompt_text, path_string)
            if not overwrite_memory:
                if key in self.memory:
                    if type(self.memory[key]) == dict:
                        if key not in self.generated_response:
                            self.generated_response[key] = True
                            self.produced_tokens += self.memory[key]["num_tokens"]
                        # parse only the answer from the full response (after </think>)
                        answer_text = re.sub(
                            r".*?</think>",
                            "",
                            self.memory[key]["response"],
                            flags=re.DOTALL,
                        ).strip()

                        # print("PARSED ANSWER FROM MEMORY: ", answer_text)
                        return answer_text

                    elif not thinking and type(self.memory[key]) == str:
                        print("Returning stored response from memory...")
                        return self.memory[key]

                    else:
                        print("Regenerating, no tokens stored in memory...")
                        # return self.memory[key]
            else:
                print("Overwriting memory for this prompt and images...")

        if not self.model_loaded:
            self._load_model()

        print(f"Prompting model...")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        if len(paths) > 1:
            resized_images = self._preprocess_images(paths, max_size=224)
            image_messages = [{"type": "image", "image": img} for img in resized_images]
        else:
            image_messages = [{"type": "image", "image": path} for path in paths]

        messages[0]["content"] += image_messages

        # [{"type": "image", "image": image_paths[i]} for i in range(len(image_paths))] +

        # Prompt the model
        with torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")

            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.sampling,
                return_dict_in_generate=True,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated.sequences)
            ]

            # Decode — keep special tokens so <think> shows up
            decoded_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=False,  # <-- change this
                clean_up_tokenization_spaces=False,
            )

        # Extract reasoning and final output
        output_tokens = 0
        for full_text in decoded_texts:
            think_match = re.search(r"(.*?)</think>", full_text, re.DOTALL)
            thinking_text = think_match.group(1).strip() if think_match else ""
            final_answer = re.sub(
                r".*?</think>", "", full_text, flags=re.DOTALL
            ).strip()

            # if not thinking_text:
            #     print("⚠️ No thinking section found for this output.")

            thinking_tokens = (
                len(
                    self.processor.tokenizer.encode(
                        thinking_text, add_special_tokens=False
                    )
                )
                if thinking_text
                else 0
            )
            answer_tokens = (
                len(
                    self.processor.tokenizer.encode(
                        final_answer, add_special_tokens=False
                    )
                )
                if final_answer
                else 0
            )

            output_tokens += thinking_tokens + answer_tokens
            # print("OUTPUT TOKENS:", output_tokens)

        self.produced_tokens += output_tokens

        # Free up GPU memory
        del inputs, generated, generated_ids_trimmed
        torch.cuda.empty_cache()

        if use_memory:
            self.memory[key] = {"response": final_answer, "num_tokens": output_tokens}
            self._save_memory()

        return final_answer


if __name__ == "__main__":
    prompter = Qwen3Prompter(
        model="Qwen3-VL-30B-A3B-Instruct", dataset="clevr", seed=42, sampling=False
    )
    prompt = "Describe the image in detail."
    paths = [
        "data/clevr/all_cubes_10/CLEVR_Hans_classid_0_000000.png",
        # "data/clevr/all_cubes_10/CLEVR_Hans_classid_0_000001.png",
    ]
    response = prompter.prompt_with_images(prompt, paths, url=False, max_new_tokens=500)
    print(response)
