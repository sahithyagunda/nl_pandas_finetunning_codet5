# CodeT5 Fine-Tuning for Natural Language to Pandas Code Translation
## Overview

This project fine-tunes the Salesforce CodeT5-small model to translate natural language instructions into pandas code snippets. The objective is to enable automatic generation of pandas commands from simple English queries, facilitating data analysis and manipulation tasks.

## Project Goals

Fine-tune a transformer model (CodeT5) on a custom dataset of natural language and corresponding pandas code.

Handle data preprocessing, tokenization, and batching for effective training.

Evaluate model performance using BLEU score to measure translation quality.

Deploy a pipeline for inference to test and demonstrate model predictions on new inputs.

## Setup and Installation

Environment Requirements

Python 3.8 or higher

GPU support strongly recommended (NVIDIA CUDA-enabled GPU)

## Required Libraries

transformers

datasets

evaluate

accelerate

torch

json (built-in)

## Installation
Install the necessary Python packages using pip:

pip install transformers datasets evaluate accelerate torch


Ensure your GPU drivers and CUDA toolkit are properly installed for GPU acceleration.

## Data Preparation

The dataset consists of JSON Lines (.jsonl) formatted records.

Each record contains a list of messages representing a conversation:

System prompt defining the assistant role.

User message with a natural language instruction.

Assistant message with the corresponding pandas code.

This format is designed to fit conversational input style compatible with CodeT5.

The dataset is split into training and evaluation subsets with a fixed random seed to ensure reproducibility.

## Tokenization and Preprocessing

The CodeT5 tokenizer is used for encoding inputs and targets.

Inputs are prepended with a task-specific prefix: "translate to pandas: " to help the model understand the task context.

Both inputs and outputs are tokenized with fixed maximum lengths and padded to ensure consistent batch sizes.

Padding tokens in labels are replaced with -100 so the loss function ignores them during training.

## Model and Training

Salesforce’s codet5-small model is loaded as the base model.

Fine-tuning is performed using Hugging Face’s Seq2SeqTrainer and Seq2SeqTrainingArguments.

## Key training parameters:

Number of epochs: 6

Batch size: 8 (with gradient accumulation to simulate larger batch size)

Learning rate: 5e-5

Mixed precision training is disabled for compatibility.

Evaluation and saving are performed at the end of each epoch.

The DataCollatorForSeq2Seq is used to dynamically pad batches and handle label padding correctly.

## Evaluation Metrics

Model output quality is evaluated using the BLEU score metric, a common choice for translation tasks.

Predictions and references are decoded from token IDs before evaluation.

Special care is taken to filter out invalid token IDs that could cause decoding errors.

## Inference Pipeline

After training, the model and tokenizer are saved for later use.

A text-to-text generation pipeline is created to perform inference on new natural language instructions.

## Example inputs:

"translate to pandas: Select columns age and salary and remove rows with missing salary"

"translate to pandas: Create a new column called tax which is 10% of salary"

The pipeline generates the corresponding pandas code snippet with beam search decoding.

## Challenges and Solutions

Encountered decoding errors due to invalid token IDs and padding token misalignments.

Handled OverflowError issues by ensuring token IDs fit within the expected integer range.

Managed device placement errors by aligning tensor devices during metric computations.

Adjusted generation parameters to avoid conflicts between max_length and max_new_tokens.

Added explicit task prefixes to improve model understanding and training effectiveness.

## Results and Usage

The fine-tuned model generates accurate and syntactically correct pandas code for a range of natural language queries.

This project demonstrates an effective workflow for customizing transformer models to domain-specific code generation tasks.

The resulting model can be integrated into applications for automating data manipulation instructions.

## Future Work

Expand the dataset size and diversity to cover more pandas functions and edge cases.

Explore larger models or different architectures for improved performance.

Implement a user-friendly interface or API for broader accessibility.

Incorporate additional evaluation metrics and human evaluation for quality assessment.

##  References

Salesforce CodeT5 Model

Hugging Face Transformers Documentation

Hugging Face Datasets

BLEU Metric
