# Vision-Language Programs

Vision-Language Programs (VLP) is a framwork for inducing executable programs that explain Bongard-like visual reasoning tasks. The system combines vision-language models (VLMs) with a symbolic DSL, and searches for structured programs that satisfy a dataset’s positive/negative image constraints. This repository contains the full training, inference, and evaluation stack used in our experiments.

## Repository Layout

| Path | Description |
| --- | --- |
| `main.py`,| End-to-end pipeline that orchestrate symbol grounding, DSL construction, and program search. |
| `method/` | DSL definitions, program search algorithms, grammar utilities, and type system. |
| `models/` | Prompter implementations for each supported VLM, including caching logic in `models/*/memory/`. |
| `prompts/` | Prompt templates for variable discovery, baselines, judgment etc.. |
| `utils/` | Argument parsing, dataset helpers, prompter factory, and GPU reservation utilities. |
| `scripts/` | Convenience launchers for running sweeps across datasets, models, and seeds. |


All experiment outputs are written to `results/<dataset>/...` and include discovered programs, cached image representations, and token usage summaries.

## Installation

1. **Clone and create an environment**
   ```bash
   git clone <repo-url>
   cd vision-language-programs
   python -m venv .venv && source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Qwen3 VL models depend on the bleeding-edge Transformers package:
   ```bash
   pip install --upgrade pip
   pip install git+https://github.com/huggingface/transformers
   ```

3. **Configure model credentials**
   - Hugging Face models (InternVL, Ovis, Molmo, Qwen) require cached checkpoints or access tokens.
   - GPT backend expects API key
   - `models/*/memory` will cache question/answer pairs per dataset, so make sure the folders are writable.

4. **Hardware**
   - Experiments assume at least one CUDA GPU. `utils/util.py::reserve_gpus()` pins a tensor on every detected device so LLM loading does not preempt another job.
   - Memory requirements grow with the chosen model (InternVL3-14B and Qwen3 30B need model-parallel setups, see the device mapping in `models/internvl/main.py`).

## Dataset Preparation

Datasets should be placed under `data/<dataset-name>/...` following the paths assumed in `utils/dataset_utils.py`. The loader handles train/test splits and enforces that test images are disjoint from train.

To obtain the datasets you can use the following commands. 

### Bongard-HOI
```bash
wget https://zenodo.org/record/7079175/files/bongard_hoi_images.tar?download=1 -O bongard_hoi_images.tar
tar -xvf bongard_hoi_images.tar -C data/
```

### Bongard-RWR
```bash
mkdir -p data
cd data
git clone https://github.com/pavonism/Bongard-RWR.git
cd ..
```

### Bongard-OpenWorld (bongard-op)
- Images are expected under `data/bongard-op/`.
- Metadata is loaded via `datasets.load_dataset("rujiewu/Bongard-OpenWorld")`; make sure you have a local Hugging Face cache.

### CLEVR-Hans 3
```bash
wget https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2611/CLEVR-Hans3.zip
unzip CLEVR-Hans3.zip -d data/
```

### COCOLogic
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip -d data/cocologic/coco
unzip val2017.zip -d data/cocologic/coco
unzip annotations_trainval2017.zip -d data/cocologic/coco
python data/cocologic/cocologic.py
```

If you add a new dataset, register it in `utils/args.py`, implement a loader in `utils/dataset_utils.py`, and optionally define a custom DSL in `method/DSL/`.

## Running Program Synthesis

The main entry point is `main.py`, which loops over every task in a dataset, discovers variables, builds a DSL, and searches for programs:

```bash
python main.py \
  --dataset bongard-op \
  --model InternVL3-8B \
  --search_timeout 10 \
  --n_objects 10 \
  --n_properties 10 \
  --n_actions 3 \
  --max_program_depth 4 \
  --max_imgs 6 \
  --variable_distribution naive_weighted \
  --seed 0
```

Important flags (see `utils/args.py` for the full list):

- `--dataset`: one of the loaders implemented in `utils/dataset_utils.py`.
- `--model`: any backend supported by `utils/prompters.get_prompter()`.
- `--max_program_depth` / `--search_timeout`: trade off search completeness and runtime.
- `--variable_distribution`: choose `uniform`, `naive_frequency`, `naive_weighted`, or `positive_ratio` PCFG weighting.
- `--no_sampling`, `--xil_remove_confounders`, `--xil_add_functions`, `--xil_add_properties`: toggles for ablations.

Results are written to `results/<dataset>/.../discovered_programs_<args>.json` along with cached image representations and a TXT file containing total VLM token usage.

### Single-task debugging
Use `run_single_task.py` to focus on one Bongard puzzle, inspect the discovered variables, and return the best program/D SL pair:
```bash
python run_single_task.py --dataset bongard-op --model Qwen2.5-VL-7B-Instruct --seed 4
```

### Baselines
`baseline.py`, `baseline_structure.py`, and `baseline_structure.py` implement direct prompting approaches that predict rules without program synthesis. Each baseline uses `prompts/baseline_prompt.txt` (or alternatives) and logs qualitative predictions to `results/qualitative`.

### Explainable Intervention Learning (XIL)
`xil_experiment.py` mirrors `main.py` but augments the DSL with extra functions/properties or removes confounders. Use the flags above or run the script directly for fine-grained control.

### Batch scripts
`scripts/*.sh` encode common experiment grids (datasets × models × seeds). They assume a UNIX shell with CUDA visibility (e.g., `CUDA_VISIBLE_DEVICES=<id> bash scripts/run_experiment.sh`). Adapt these scripts to your cluster scheduler if needed.

## Evaluation and Analysis

- `eval.py` contains helpers to aggregate accuracies from the JSON result files (see `n_tasks_per_dataset` and `params_per_dataset` for recommended settings). You can import `eval.eval(path, n_tasks)` in a notebook or a small driver script.
- `eval_variable_discovery.py` and `eval_qualitative.py` benchmark the discovery prompts and produce qualitative grids of predictions.
- `plots/`, `motivating_example.ipynb`, and `qualitative_examples.ipynb` show example visualizations of discovered programs and rules.

## Customization Tips

- **Prompt editing:** Modify files in `prompts/` to adjust how variables are queried, how baselines describe rules, or how judges verify predictions.
- **DSL extensions:** Add new primitives under `method/DSL/` and update the semantics/types so they appear in the CFG (see `method/DSL/dsl_with_img_repr.py` for reference).
- **Additional VLMs:** Implement a new prompter in `models/<name>/main.py` that exposes `prompt_with_images`, register it in `utils/prompters.get_prompter`, and add any required dependencies to `requirements.txt`.
- **Caching:** Every prompter writes JSON files under `models/<model>/memory/<dataset>/` to avoid repeated API calls.

