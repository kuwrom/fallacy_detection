# Logical Fallacy Dataset and Model Fine-tuning

This repository contains a collection of logical fallacy datasets, tools for generating synthetic data, and resources for fine-tuning language models on logical fallacy detection and generation.

## Datasets

### Organic Datasets

The organic datasets are sourced from various projects:

- [causalNLP/logical-fallacy](https://github.com/causalNLP/logical-fallacy)
- [Argotario](https://github.com/UKPLab/argotario/blob/master/data/arguments-en-2018-01-15.tsv)
- [Logic (edu_train, edu_dev, edu_test)](https://github.com/causalNLP/logical-fallacy/tree/main/data)
- [Propaganda](https://propaganda.qcri.org/nlp4if-shared-task/data/datasets-v2.tgz)

### Synthetic Datasets

Synthetic datasets are generated from the organic datasets to expand the number of examples for each fallacy category. The `generate_synthetic_data.py` script is used for this purpose.

### Training Datasets

The training datasets consist of articles generated from the synthetic sentences. Due to computational constraints, only 3 categories of articles have been fully generated. The training data is stored in JSONL format in the `data/training/` directory.

## Scripts

- `generate_synthetic_data.py`: Main script for generating synthetic fallacy sentences and articles.
- `validate_dataset.py`: Script to validate the generated datasets.
- `check_status.py`: Script to check the status of fine-tuning jobs, create files, and test the model.

## Model Fine-tuning

The generated datasets are used to fine-tune a LLaMA 2 or 3 model. The fine-tuning process was performed using [Any Scale](https://www.anyscale.com/).

## Usage

1. Clone the repository: git clone https://github.com/kuwrom/fallacy_detection.git
2. Install the required dependencies: pip install -r requirements.txt
3. Generate synthetic data: python generate_synthetic_data.py
4. Validate the generated dataset: python validate_dataset.py
5. 5. Use the `check_status.py` script for various operations:
   - Create files for fine-tuning:
   - Start a fine-tuning job:
   - List fine-tuning jobs:
   - Retrieve file content:
   - Test the fine-tuned model:
     
## Results

The results of the model fine-tuning can be found in the `result.txt` file.

## Contributing

Contributions to expand the dataset or improve the data generation process are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## Acknowledgements

This project builds upon the work of several open-source projects and datasets. We thank the authors and contributors of the original datasets for making their work available.
