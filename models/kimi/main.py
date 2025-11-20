import torch
from typing import Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)
from qwen_vl_utils import process_vision_info
import os
import json
import requests
from PIL import Image


from transformers import AutoProcessor


class KimiPrompter:
    def __init__(
        self,
        model="Kimi-VL-A3B-Thinking-2506",
        dataset="bongard_hoi",
        seed=0,
        sampling=False,
    ):

        # set seed
        # print("set seed ", seed)
        self.seed = seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        print(f"LOADING KIMI PROMPTER: {model}...")

        # set the model name
        self.model_name = model
        self.sampling = sampling
        self.produced_tokens = 0
        self.generated_response = {}

        self.memory: Dict[Tuple[str, str], str] = (
            {}
        )  # Dictionary to store (prompt, image_path) -> response
        self.token_memory = {}

        if not self.sampling:
            print("Using greedy decoding (no sampling) for VLM.")
            model_name = model + "_no_sampling"
        else:
            model_name = model

        if "bongard-op" in dataset:
            dataset = "bongard-op"

        self.memory_file = (
            f"models/kimi/memory/{dataset}/vlm_memory_{model_name}_{seed}.json"
        )
        self.token_file = (
            f"models/kimi/memory/{dataset}/vlm_tokens_{model_name}_{seed}.json"
        )
        self._load_memory()
        self._load_token_memory()

        self.model_name = model
        self.model_loaded = False
        # set the device
        self.device = "auto"
        self.execution_counter = 0

    def _load_model(self):
        # self.device = "cuda:0"
        if self.model_name == "Kimi-VL-A3B-Thinking-2506":
            self.model_path = "moonshotai/Kimi-VL-A3B-Thinking-2506"
        elif self.model_name == "Kimi-VL-A3B-Instruct":
            self.model_path = "moonshotai/Kimi-VL-A3B-Instruct"
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self.model_loaded = True

    def _load_memory(self):
        """Loads memory from a file if it exists, converting keys back to tuples."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                # Convert string keys back to tuples
                self.memory = {eval(k): v for k, v in data.items()}

    def _load_token_memory(self):
        """Loads token memory from a file if it exists."""
        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                data = json.load(f)
                self.token_memory = {eval(k): v for k, v in data.items()}

    def _save_memory(self):
        """Saves memory to a file, converting tuple keys to strings for JSON storage."""
        memory_dict = {str(k): v for k, v in self.memory.items()}
        memory_as_json = json.dumps(memory_dict, indent=4)

        # create directories if not exist
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        with open(self.memory_file, "w") as f:
            f.write(memory_as_json)

    def _save_token_memory(self):
        """Saves token memory to a file, converting tuple keys to strings for JSON storage."""
        token_dict = {str(k): v for k, v in self.token_memory.items()}
        token_as_json = json.dumps(token_dict, indent=4)

        # create directories if not exist
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)

        with open(self.token_file, "w") as f:
            f.write(token_as_json)

    def _extract_thinking_and_summary(
        self, text: str, bot: str = "◁think▷", eot: str = "◁/think▷"
    ) -> str:
        if bot in text and eot not in text:
            return "", ""
        if eot in text:
            return (
                text[text.index(bot) + len(bot) : text.index(eot)].strip(),
                text[text.index(eot) + len(eot) :].strip(),
            )
        return "", text

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    def prompt_with_text(
        self,
        prompt_text: str,
        system_prompt=None,
        seed=None,
        use_memory=True,
        max_new_tokens=96,
        do_sample=True,
        overwrite_memory=False,
        thinking=False,
    ):

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if use_memory:
            key = (prompt_text, "no_image")
            if not overwrite_memory:
                # check if the prompt and image path are already in memory
                if key in self.memory:
                    # print("Retrieving response from memory.")
                    return self.memory[key]

        if not self.model_loaded:
            self._load_model()

        OUTPUT_FORMAT = (
            "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = self.processor(
            images=None, text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        if "Thinking" in self.model_name or thinking:
            max_new_tokens = 32768

        if self.sampling:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8 if "Thinking" in self.model_name else 0.2,
                    # use_cache=False,
                    do_sample=True,
                )
        else:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        # print("Generated ids.")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # print("Batch decode...")
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # print(response)

        # get number of tokens
        output_tokens = generated_ids_trimmed[0].size(0)
        # print("Output tokens:", output_tokens)

        if "Thinking" in self.model_name or thinking:

            thinking, summary = self._extract_thinking_and_summary(response)
            print(OUTPUT_FORMAT.format(thinking=thinking, summary=summary))

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        if use_memory:
            self.memory[key] = response
            self._save_memory()

        return response

    def prompt_with_images(
        self,
        prompt_text,
        paths,
        url=False,
        use_memory=True,
        max_new_tokens=5000,
        overwrite_memory=False,
        thinking=False,
    ):

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
                        # check if num_tokens is in memory
                        # print(self.memory[key])
                        if (
                            "num_tokens" in self.memory[key]
                            and self.memory[key]["num_tokens"] is not None
                        ):
                            self.produced_tokens += self.memory[key]["num_tokens"]

                            return self.memory[key]["response"]

                    elif not thinking and type(self.memory[key]) == str:
                        return self.memory[key]

                    else:
                        print("no token info in memory, regenerating...")
                        # return self.memory[key]
                else:
                    # print("No stored response in memory, generating...")
                    pass

        if self.model_loaded is False:
            self._load_model()

        # print(f"Prompting model...")

        OUTPUT_FORMAT = (
            "--------Thinking--------\n{thinking}\n\n--------Summary--------\n{summary}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        images = [Image.open(path) for path in paths]

        if len(paths) > 1:
            # resize images
            images = [img.resize((224, 224)) for img in images]

            image_messages = [
                {"type": "image", "image": path, "max_pixels": 224 * 224}
                for path in paths
            ]
        else:
            image_messages = [{"type": "image", "image": path} for path in paths]

        messages[0]["content"] += image_messages

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = self.processor(
            images=images, text=text, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        # print("Start generating...")

        if "Thinking" in self.model_name or thinking:
            max_new_tokens = 32768

        if self.sampling:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8 if "Thinking" in self.model_name else 0.2,
                    # use_cache=False,
                    do_sample=True,
                )
        else:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

        # print("Generated ids.")
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # print("Batch decode...")
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # print(response)

        # get number of tokens
        output_tokens = generated_ids_trimmed[0].size(0)
        # print("Output tokens:", output_tokens)

        if "Thinking" in self.model_name or thinking:

            thinking, summary = self._extract_thinking_and_summary(response)
            print(OUTPUT_FORMAT.format(thinking=thinking, summary=summary))

        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

        if use_memory:
            self.memory[key] = {"response": response, "num_tokens": output_tokens}
            self._save_memory()

        return response


if __name__ == "__main__":
    prompter = KimiPrompter(
        model="Kimi-VL-A3B-Thinking-2506", dataset="bongard_hoi", seed=0
    )
    response = prompter.prompt_with_images(
        prompt_text="Describe the images.",
        paths=[
            "./data/bongard-op/images/0000/neg__0__2023-03-18-12-28-26__https:__secure.static.meredith.com__crt__store__covers__magazines__nmo__3308_l.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__0__2023-03-18-12-28-26__https:__secure.static.meredith.com__crt__store__covers__magazines__nmo__3308_l.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
            "./data/bongard-op/images/0000/neg__1__2023-03-18-12-28-17__https:__www.usmagazine.com__wp-content__uploads__2022__12__About-Last-Night-See-the-Best-Dressed-Stars-on-the-Red-Carpet-and-Beyond-4.jpg?w=1200&quality=86&strip=all.jpg",
        ],
        url=False,
        max_new_tokens=5000,
        use_memory=True,
    )
    print("Response:", response)
