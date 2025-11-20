from typing import Dict, Tuple
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import os
import json
import torch
import gc


class Qwen2_5Prompter:
    def __init__(
        self, model="Qwen/Qwen2.5-VL-72B-Instruct", dataset=None, seed=0, sampling=True
    ):

        # set seed
        print("set seed ", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.sampling = sampling

        if model == "Qwen2.5-VL-72B-Instruct":
            self.model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
        elif model == "Qwen2.5-VL-32B-Instruct":
            self.model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
        elif model == "Qwen2.5-VL-7B-Instruct":
            self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            raise ValueError(
                f"Model {model} not supported. Use 'Qwen2.5-VL-72B-Instruct'."
            )

        self.model_loaded = False
        self.memory = {}  # Dictionary to store (prompt, image_path) -> response

        if "bongard-op" in dataset:
            dataset = "bongard-op"

        if not self.sampling:
            print("Using greedy decoding (no sampling) for VLM.")
            model_name = model + "_no_sampling"
        else:
            model_name = model

        self.memory_file = (
            f"models/qwen/memory/{dataset}/vlm_memory_{model_name}_{seed}.json"
        )
        self._load_memory()

    def _load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
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
        # Not implemented for Qwen model
        return 0

    def reset_produced_tokens(self):
        # Not implemented for Qwen model
        pass

    def prompt_with_text(
        self,
        prompt_text,
        paths,
        url=False,
        use_memory=True,
        max_new_tokens=5000,
        overwrite_memory=False,
    ):

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
    ):

        # print("Prompt model with sampling:", self.sampling)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if use_memory:
            path_string = ",".join(paths)
            key = (prompt_text, path_string)
            if not overwrite_memory:
                if key in self.memory:
                    return self.memory[key]

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
        if url:
            messages[0]["content"].append(
                {
                    "type": "image",
                    "image": paths,
                }
            )
        else:
            if len(paths) > 1:
                image_messages = [
                    {"type": "image", "image": path, "max_pixels": 224 * 224}
                    for path in paths
                ]
            else:
                image_messages = [{"type": "image", "image": path} for path in paths]

            messages[0]["content"] += image_messages

            # [{"type": "image", "image": image_paths[i]} for i in range(len(image_paths))] +

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
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
