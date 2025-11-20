import base64
import os
import json
from pathlib import Path
from openai import OpenAI
from datetime import datetime
from typing import Dict, Tuple
from PIL import Image
from io import BytesIO


class GPTPrompter:
    def __init__(self, model="gpt-4o", dataset="default", seed=42, reasoning=True):
        # load the API key from "open-ai-key"
        with open("models/gpt/open-ai-key-vlp", "r") as file:
            api_key = file.read().strip()

        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        if model == "gpt-4o":
            model = "gpt-4o-2024-08-06"
        if model == "o1":
            model = "o1-2024-12-17"
        if model == "gpt-5-mini":
            model = "gpt-5-mini-2025-08-07"
        else:
            model = model

        self.model = model
        self.seed = seed
        self.system_fingerprint = None
        self.execution_counter = 0
        self.produced_tokens = 0
        self.generated_response = {}
        self.reasoning = reasoning

        print(f"USING MODEL: {model}")

        self.memory: Dict[Tuple[str, str], str] = (
            {}
        )  # Dictionary to store (prompt, image_path) -> response
        self.memory_file = f"models/gpt/memory/{dataset}/vlm_memory_{model}_{seed}.json"
        self._load_memory()

        self.token_file = f"models/gpt/tokens/{dataset}/vlm_tokens_{model}_{seed}.json"
        self._load_tokens()

    def _load_memory(self):
        """Loads memory from a file if it exists, converting keys back to tuples."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                # Convert string keys back to tuples
                self.memory = {eval(k): v for k, v in data.items()}

    def _load_tokens(self):
        """Loads token usage from a file if it exists."""
        # get current timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                self.token_usage = json.load(f)
            # Add current timestamp
            self.token_usage[self.timestamp] = 0

        else:
            # create directories if not exist
            os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
            self.token_usage = {self.timestamp: 0}

    def _save_memory(self):
        """Saves memory to a file, converting tuple keys to strings for JSON storage."""
        memory_dict = {str(k): v for k, v in self.memory.items()}
        memory_as_json = json.dumps(memory_dict, indent=4)

        # create directories if not exist
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        with open(self.memory_file, "w") as f:
            f.write(memory_as_json)

    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    def prompt(self, prompt_text, system_prompt=None, seed=None, temp=None):
        """Generate a response to a prompt using the OpenAI API."""

        if seed is None:
            seed = self.seed

        # if system_prompt is None:
        #     system_prompt = "You are a helpful assistant that can describe images provided by the user in extreme detail. You are able to recognize abstract concepts in images like humans do. You are helping a scientist discover relevant patterns in images."

        # Call the completion endpoint
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                        }
                    ],
                },
                {"role": "user", "content": prompt_text},
            ],
            model=self.model,
            seed=seed,
            max_tokens=2000,
            temperature=temp,
        )
        # Get system fingerprint
        self.system_fingerprint = response.system_fingerprint
        self.model = response.model

        response_text = response.choices[0].message.content.strip()

        # self._log_prompt(prompt_text, response_text)

        return response_text

    def _encode_image(self, image_path, resize=False):

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {image_path}")

        # Always go through PIL to ensure consistent format
        with Image.open(p) as img:
            # # Convert to RGB for JPEG to avoid errors with PNG/PNG+alpha
            # if format.upper() == "JPEG" and img.mode != "RGB":
            #     img = img.convert("RGB")

            if resize:
                size = (224, 224)
                # High quality downscale
                img = img.resize(size, Image.Resampling.LANCZOS)

            buf = BytesIO()
            save_kwargs = {}
            # if format.upper() == "JPEG":
            #     # Reasonable defaults for JPEG
            #     save_kwargs.update({"quality": 90, "optimize": True})
            img.save(buf, format="PNG", **save_kwargs)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return b64

    def prompt_with_text(
        self,
        prompt_text: str,
        system_prompt=None,
        seed=None,
        use_memory=True,
        max_new_tokens=2000,
        do_sample=True,
        overwrite_memory=False,
    ):

        if use_memory:
            key = (prompt_text, None)
            # check if the prompt is already in memory
            if key in self.memory:
                if not overwrite_memory:
                    # print("Retrieving response from memory.")
                    return self.memory[key]
                else:
                    print("Overwriting memory for this prompt...")

        if seed is None:
            seed = self.seed

        # Call the completion endpoint
        response = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt_text},
            ],
            model=self.model,
            seed=seed,
            max_tokens=max_new_tokens,
        )

        response_text = response.choices[0].message.content.strip()

        if use_memory:
            # Store in memory
            self.memory[key] = response_text

        return response_text

    def prompt_with_images(
        self,
        prompt_text: str,
        paths: [str],
        system_prompt=None,
        seed=None,
        url=False,
        use_memory=True,
        max_new_tokens=2000,
        overwrite_memory=False,
        thinking=False,
    ):

        # # overwrite_memory = True
        if "gpt-5" in self.model and max_new_tokens < 4096:
            max_new_tokens = 4096

        if use_memory:
            path_string = ",".join(paths)
            key = (prompt_text, path_string)
            # check if the prompt and image path are already in memory
            if key in self.memory and not overwrite_memory:
                # check that response is not empty
                if type(self.memory[key]) == dict:
                    if self.memory[key]["response"] != "":
                        if key not in self.generated_response:
                            self.generated_response[key] = True
                            self.produced_tokens += self.memory[key]["num_tokens"]

                        return self.memory[key]["response"]

                    else:
                        print("Memorized response is empty, re-generating.")

        if seed is None:
            seed = self.seed

        print(f"Processing prompt with {len(paths)} images.")

        # encode images
        resize = len(paths) > 1
        encoded_images = [self._encode_image(path, resize=resize) for path in paths]

        # build input content
        input_content = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt_text}],
            }
        ]

        for image in encoded_images:
            input_content[-1]["content"].append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image}",
                }
            )

        try:

            if (
                "gpt-5" in self.model
                and not self.model == "gpt-5-chat-latest"
                and self.reasoning
            ):
                print("Using reasoning...")
                # create a response using the new unified endpoint with reasoning
                response = self.client.responses.create(
                    model=self.model,
                    input=input_content,
                    reasoning={"effort": "high"},
                    max_output_tokens=max_new_tokens,
                )
            elif (
                "gpt-5" in self.model
                and not self.model == "gpt-5-chat-latest"
                and not self.reasoning
            ):
                print("Not using reasoning...")
                # create a response using the new unified endpoint without reasoning
                response = self.client.responses.create(
                    model=self.model,
                    input=input_content,
                    reasoning={"effort": "low"},
                    max_output_tokens=max_new_tokens,
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=input_content,
                    max_output_tokens=max_new_tokens,
                    temperature=0.2,
                )

            # print(response)
            if response == "":
                print("EMPTY RESPONSE!")

            # extract model output
            response_text = getattr(response, "output_text", "")
            output_tokens = response.usage.output_tokens

            response_text = response_text.strip()

            print("Output tokens details:", output_tokens)
            self.token_usage[self.timestamp] += output_tokens
            # save token usage to file
            with open(self.token_file, "w") as f:
                json.dump(self.token_usage, f, indent=4)

            all_tokens = output_tokens

        except Exception as e:
            print("Error:", e)
            response_text = ""
            all_tokens = 0

        self.execution_counter += 1

        if use_memory:
            # Store in memory
            self.memory[key] = {"response": response_text, "num_tokens": all_tokens}
            self._save_memory()

        self.produced_tokens += all_tokens

        return response_text


if __name__ == "__main__":
    prompter = GPTPrompter(model="gpt-5-mini", dataset="cocologic", seed=42)
    prompt = "Describe the image in detail."
    image_path = "data/CLEVR-Hans3/train/images/CLEVR_Hans_classid_0_000000.png"

    response = prompter.prompt_with_images(
        prompt,
        [image_path, "data/CLEVR-Hans3/train/images/CLEVR_Hans_classid_0_000001.png"],
    )
    print("Response:", response)
