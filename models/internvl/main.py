from transformers import AutoTokenizer, AutoModel, AutoConfig
from PIL import Image
import requests
import torch
import json
import os
from typing import Dict, Tuple
import torchvision.transforms as T
import math
import gc

# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from rtpt import RTPT


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLPrompter:
    def __init__(
        self, model="InternVL3-8B", dataset="bongard_hoi", seed=0, sampling=False
    ):

        # set seed
        print("set seed ", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.seed = seed

        print(f"LOADING INTERNVL PROMPTER: {model}...")
        print("Use sampling:", sampling)

        # set the model name
        self.model_name = model

        self.sampling = sampling
        self.produced_tokens = 0

        self.memory: Dict[Tuple[str, str], str] = (
            {}
        )  # Dictionary to store (prompt, image_path) -> response
        if "bongard-op" in dataset:
            dataset = "bongard-op"

        if not self.sampling:
            print("Using greedy decoding (no sampling) for VLM.")
            model_name = model + "_no_sampling"
        else:
            model_name = model

        self.memory_file = (
            f"models/internvl/memory/{dataset}/vlm_memory_{model_name}_{seed}.json"
        )
        self._load_memory()
        self.model_name = model
        self.model_loaded = False
        # set the device
        self.device = "auto"
        self.execution_counter = 0

    def _load_model(self):
        # self.device = "cuda:0"
        if self.model_name == "InternVL3-8B":
            path = "OpenGVLab/InternVL3-8B"
        elif self.model_name == "InternVL3-14B":
            path = "OpenGVLab/InternVL3-14B"
        elif "InternVL3" in self.model_name:
            path = f"OpenGVLab/{self.model_name}"
        else:
            raise ValueError(f"Model {self.model_name} not supported.")

        device_map = "auto"  # self._split_model("InternVL3-8B")  # use model path?
        if self.model_name == "InternVL3-14B":
            device_map = self._split_model(path)
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )

        self.model_loaded = True

    def _split_model(self, model_path):
        device_map = {}
        world_size = torch.cuda.device_count()
        print(f"Using {world_size} GPUs for model parallelism.")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f"language_model.model.layers.{layer_cnt}"] = i
                layer_cnt += 1
        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

        return device_map

    def _load_memory(self):
        """Loads memory from a file if it exists, converting keys back to tuples."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r") as f:
                data = json.load(f)
                # Convert string keys back to tuples
                self.memory = {eval(k): v for k, v in data.items()}

    def _save_memory(self):
        """Saves memory to a file, converting tuple keys to strings for JSON storage."""
        memory_dict = {str(k): v for k, v in self.memory.items()}
        memory_as_json = json.dumps(memory_dict, indent=4)

        # create directories if not exist
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)

        with open(self.memory_file, "w") as f:
            f.write(memory_as_json)

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def remove_from_gpu(self):
        if self.model_loaded:
            print("Removing model from GPU...")
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            self.model_loaded = False

    def _load_image(self, image_file, input_size=448, max_num=12, resize=False):
        image = Image.open(image_file).convert("RGB")
        # print(image.size)

        if resize:
            image = image.resize((224, 224))
            # input_size = 224

        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

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

        # print("Prompting model...")
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=self.sampling,
            eos_token_id=151645,
            pad_token_id=151645,
        )

        question = f"{prompt_text}"
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer, None, question, generation_config
            )
        torch.cuda.empty_cache()

        self.execution_counter += 1

        if use_memory:
            # Store in memory
            self.memory[key] = response
            # if self.execution_counter % 1 == 0:
            self._save_memory()
            # print("Answer saved to memory.")

        return response

    def prompt_with_images(
        self,
        prompt_text: str,
        paths: [str],
        system_prompt=None,
        seed=None,
        url=False,
        use_memory=True,
        max_new_tokens=96,
        overwrite_memory=False,
    ):

        # print("Prompt model with sampling:", self.sampling)

        # print(f"Prompting model: {self.model_name} with images...")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if use_memory:
            path_string = ",".join(paths)
            key = (prompt_text, path_string)
            # append all paths to one string

            if not overwrite_memory:
                # check if the prompt and image path are already in memory
                if key in self.memory:
                    # print("Retrieving response from memory.")
                    return self.memory[key]

        if url:
            raise NotImplementedError("URL support is not implemented yet.")

        else:

            if not self.model_loaded:
                self._load_model()

            # print(f"Prompting model with seed {self.seed}...")
            # load the images
            # set the max number of tiles in `max_num`
            resize = True if len(paths) > 1 else False
            images = [
                self._load_image(path, max_num=12, resize=resize)
                .to(torch.bfloat16)
                .cuda()
                for path in paths
            ]
            if len(images) > 1:
                # cat all images together
                pixel_values = torch.cat(images, dim=0)
            else:
                pixel_values = images[0]

            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=self.sampling,
                eos_token_id=151645,
                pad_token_id=151645,
            )

            # single/multi-image single-round conversation
            question = f"<image>\n{prompt_text}"
            with torch.no_grad():
                response = self.model.chat(
                    self.tokenizer, pixel_values, question, generation_config
                )
            torch.cuda.empty_cache()

            self.execution_counter += 1

        if use_memory:
            # Store in memory
            self.memory[key] = response
            # if self.execution_counter % 1 == 0:
            self._save_memory()
            # print("Answer saved to memory.")

        return response


if __name__ == "__main__":

    # Example usage
    prompter = InternVLPrompter(model="InternVL3-8B")
    prompt_text = "Count the objects in the image. How many objects are there?"
    paths = [
        "data/clevr/all_cubes_10/CLEVR_Hans_classid_0_000000.png",
        # "data/clevr/all_cubes_10/CLEVR_Hans_classid_0_000001.png",
    ]
    response = prompter.prompt_with_images(prompt_text, paths, url=False)

    print(response)
