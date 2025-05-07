import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import json

# --- Configuration ---
MODEL_NAME = "tasksource/deberta-small-long-nli"
# Replace with the actual paths to your JSON files
MAIN_DATA_JSON_PATH = "data/train-claims.json"  # e.g., "data/train_claims.json"
EVIDENCE_JSON_PATH = "data/evidence.json"  # e.g., "data/evidences.json"
DEV_DATA_JSON_PATH = "data/dev-claims.json"  # Path for the development/validation set
TEXT_COLUMN_CLAIM = "claim_text"
TEXT_COLUMN_EVIDENCE = "evidence_text"
LABEL_COLUMN = "claim_label"

# Define your labels (these can be overridden by discovery if the discovery block is active)
LABELS = ["DISPUTED", "REFUTES", "SUPPORTS", "NOT_ENOUGH_INFO"]
label2id = {}
id2label = {}
NUM_LABELS = 0  # Will be set after label discovery

MAX_TOKEN_LENGTH = 512

# Training Hyperparameters
OUTPUT_DIR = "./results_claim_classifier"
LOGGING_DIR = "./logs_claim_classifier"
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
LOGGING_STEPS = 100
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"
TEST_SIZE_SPLIT = 0.1  # Fallback if dev set is not provided or fails to load


# --- New function to load data from JSON files ---
def load_data_from_json(main_data_path, evidence_json_path, data_type_name="data"):
    """
    Loads main data and evidence data from JSON files, combines them,
    and returns a Pandas DataFrame.
    `data_type_name` is for logging purposes (e.g., "training", "development").
    """
    print(f"Loading {data_type_name} from: {main_data_path}")
    print(f"Using evidence from: {evidence_json_path}")
    try:
        with open(main_data_path, "r") as f:
            main_data = json.load(f)
        with open(evidence_json_path, "r") as f:
            evidence_lookup = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {data_type_name} file not found. {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON for {data_type_name}. {e}")
        return None

    processed_data = {TEXT_COLUMN_CLAIM: [], LABEL_COLUMN: [], TEXT_COLUMN_EVIDENCE: []}

    for claim_id, datum in main_data.items():
        if not all(key in datum for key in ["claim_text", "claim_label", "evidences"]):
            print(
                f"Warning: Skipping claim_id '{claim_id}' in {data_type_name} due to missing keys ('claim_text', 'claim_label', or 'evidences')."
            )
            continue

        processed_data[TEXT_COLUMN_CLAIM].append(datum["claim_text"])
        processed_data[LABEL_COLUMN].append(datum["claim_label"])

        current_evidence_texts = []
        for evidence_id in datum["evidences"]:
            if evidence_id in evidence_lookup:
                current_evidence_texts.append(str(evidence_lookup[evidence_id]))
            else:
                print(
                    f"Warning: Evidence ID '{evidence_id}' for claim '{claim_id}' in {data_type_name} not found in evidence file. Skipping this evidence."
                )

        evidence_concat = " ".join(current_evidence_texts)
        processed_data[TEXT_COLUMN_EVIDENCE].append(evidence_concat)

    if not processed_data[TEXT_COLUMN_CLAIM]:
        print(
            f"Error: No {data_type_name} was processed. Check the structure of your JSON files and logs for warnings."
        )
        return None

    return pd.DataFrame(processed_data)


# --- 1. Load and Preprocess Data (Modified) ---
def load_and_preprocess_data(
    train_dataframe,
    eval_dataframe,  # Can be None
    tokenizer,
    label2id_map,
    test_size_for_split=0.1,  # Used if eval_dataframe is None
    random_state=42,
):
    """Processes train and eval DataFrames, or splits train_dataframe if eval_dataframe is None."""

    processed_datasets = {}

    # --- Process Training Data ---
    train_df_processed = train_dataframe.copy()
    if not all(
        col in train_df_processed.columns
        for col in [TEXT_COLUMN_CLAIM, TEXT_COLUMN_EVIDENCE, LABEL_COLUMN]
    ):
        print(
            f"Error: Training DataFrame must contain columns: '{TEXT_COLUMN_CLAIM}', '{TEXT_COLUMN_EVIDENCE}', and '{LABEL_COLUMN}'."
        )
        return None

    train_df_processed["label_id"] = train_df_processed[LABEL_COLUMN].map(label2id_map)
    if train_df_processed["label_id"].isnull().any():
        unmapped_train_labels = train_df_processed[
            train_df_processed["label_id"].isnull()
        ][LABEL_COLUMN].unique()
        print(
            f"Warning (Train): Labels not in label2id_map: {unmapped_train_labels}. These rows will be dropped."
        )
        train_df_processed.dropna(subset=["label_id"], inplace=True)
        if train_df_processed.empty:
            print(
                "Error (Train): No valid training data remains after attempting to map labels."
            )
            return None
    train_df_processed["label_id"] = train_df_processed["label_id"].astype(int)

    # --- Process Evaluation Data ---
    if eval_dataframe is not None:
        eval_df_processed = eval_dataframe.copy()
        if not all(
            col in eval_df_processed.columns
            for col in [TEXT_COLUMN_CLAIM, TEXT_COLUMN_EVIDENCE, LABEL_COLUMN]
        ):
            print(
                f"Error: Evaluation DataFrame must contain columns: '{TEXT_COLUMN_CLAIM}', '{TEXT_COLUMN_EVIDENCE}', and '{LABEL_COLUMN}'."
            )
            # Optionally, one might decide to proceed without eval data or raise an error.
            # For now, we'll set eval_df_processed to None and proceed with training only if this happens.
            print(
                "Warning: Evaluation data is invalid. Proceeding without it or splitting train data if fallback enabled."
            )
            eval_df_processed = None
        else:
            eval_df_processed["label_id"] = eval_df_processed[LABEL_COLUMN].map(
                label2id_map
            )
            if eval_df_processed["label_id"].isnull().any():
                unmapped_eval_labels = eval_df_processed[
                    eval_df_processed["label_id"].isnull()
                ][LABEL_COLUMN].unique()
                print(
                    f"Warning (Eval): Labels not in label2id_map: {unmapped_eval_labels}. These rows will be dropped from eval set."
                )
                eval_df_processed.dropna(subset=["label_id"], inplace=True)
                if eval_df_processed.empty:
                    print(
                        "Warning (Eval): No valid evaluation data remains after label mapping. Evaluation set will be empty."
                    )
                    # eval_df_processed will remain empty, leading to an empty eval dataset.
            if not eval_df_processed.empty:
                eval_df_processed["label_id"] = eval_df_processed["label_id"].astype(
                    int
                )
            else:  # if it became empty after dropping NaNs
                eval_df_processed = None  # Ensure it's None if empty, to potentially trigger fallback later or handle empty eval

    else:  # eval_dataframe was None initially, so split train_df_processed
        print(
            f"No evaluation DataFrame provided. Splitting training data (split size: {test_size_for_split})."
        )
        if (
            len(train_df_processed) < 2
        ):  # Cannot stratify with less than 2 samples or less than 2 samples per class
            print(
                "Warning: Training data is too small to split for evaluation. Proceeding without evaluation."
            )
            eval_df_processed = None
        elif (
            train_df_processed["label_id"].nunique() < 2
            and len(train_df_processed) * test_size_for_split >= 1
        ):
            print(
                "Warning: Only one class in training data. Splitting without stratification for evaluation."
            )
            train_df_processed, eval_df_processed = train_test_split(
                train_df_processed,
                test_size=test_size_for_split,
                random_state=random_state,
            )
        else:
            try:
                train_df_processed, eval_df_processed = train_test_split(
                    train_df_processed,
                    test_size=test_size_for_split,
                    random_state=random_state,
                    stratify=train_df_processed["label_id"],
                )
            except ValueError as e:
                print(
                    f"Warning: Could not stratify split for evaluation ({e}). Splitting without stratification."
                )
                train_df_processed, eval_df_processed = train_test_split(
                    train_df_processed,
                    test_size=test_size_for_split,
                    random_state=random_state,
                )

    # --- Tokenize ---
    def tokenize_function(examples):
        return tokenizer(
            examples[TEXT_COLUMN_CLAIM],
            examples[TEXT_COLUMN_EVIDENCE],
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKEN_LENGTH,
        )

    train_dataset = Dataset.from_pandas(train_df_processed)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.rename_column(
        "label_id", "labels"
    )
    tokenized_train_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    processed_datasets["train"] = tokenized_train_dataset

    if eval_df_processed is not None and not eval_df_processed.empty:
        eval_dataset = Dataset.from_pandas(eval_df_processed)
        tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        tokenized_eval_dataset = tokenized_eval_dataset.rename_column(
            "label_id", "labels"
        )
        tokenized_eval_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )
        processed_datasets["eval"] = tokenized_eval_dataset
    else:
        print("Note: Evaluation dataset is empty or not available.")
        processed_datasets["eval"] = None  # Explicitly set to None if no eval data

    return DatasetDict(processed_datasets)


# --- Load initial data from JSONs ---
print("Loading training data...")
train_raw_df = load_data_from_json(
    MAIN_DATA_JSON_PATH, EVIDENCE_JSON_PATH, data_type_name="training data"
)
if train_raw_df is None or train_raw_df.empty:
    print("Failed to load training data or data is empty. Exiting.")
    exit()
print(f"Successfully loaded training data. DataFrame shape: {train_raw_df.shape}")

print("\\nLoading development (validation) data...")
dev_raw_df = None
if DEV_DATA_JSON_PATH:
    dev_raw_df = load_data_from_json(
        DEV_DATA_JSON_PATH, EVIDENCE_JSON_PATH, data_type_name="development data"
    )
    if dev_raw_df is None or dev_raw_df.empty:
        print(
            "Warning: Failed to load development data or it is empty. Will attempt to split training data for evaluation."
        )
        # dev_raw_df remains None, fallback logic in load_and_preprocess_data will be used
    else:
        print(
            f"Successfully loaded development data. DataFrame shape: {dev_raw_df.shape}"
        )
else:
    print(
        "No development data path provided. Will attempt to split training data for evaluation."
    )

# --- 2. Discover Labels and Mappings (from training data ONLY) ---
print("\\nDiscovering labels from the training DataFrame...")
try:
    if LABEL_COLUMN not in train_raw_df.columns:
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' not found in the training DataFrame."
        )

    unique_labels = sorted(train_raw_df[LABEL_COLUMN].astype(str).unique().tolist())
    if not unique_labels:
        raise ValueError(
            f"No labels found in column '{LABEL_COLUMN}' of the training DataFrame."
        )

    NUM_LABELS = len(unique_labels)
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    print(f"Using {NUM_LABELS} labels discovered from training data: {unique_labels}")
    print(f"Label to ID mapping: {label2id}")

except ValueError as e:
    print(f"Error processing labels from the training DataFrame: {e}")
    print("Please check your training data and LABEL_COLUMN configuration.")
    exit()

# --- 3. Load Tokenizer and Model ---
print("\\nLoading tokenizer and model...")
if NUM_LABELS == 0 or not label2id or not id2label:
    print(
        "Error: Label information (NUM_LABELS, label2id, id2label) not properly set. Exiting."
    )
    exit()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

# --- 4. Prepare Datasets ---
print("\\nPreparing datasets...")
datasets = load_and_preprocess_data(
    train_raw_df,
    dev_raw_df,  # Pass the loaded dev_raw_df (can be None)
    tokenizer,
    label2id,  # Use the mapping derived from training data
    test_size_for_split=TEST_SIZE_SPLIT,
)

if datasets is None or "train" not in datasets or datasets["train"] is None:
    print("Failed to prepare datasets or training dataset is missing. Exiting.")
    exit()

print(f"Train dataset size: {len(datasets['train'])}")
if datasets.get("eval"):
    print(f"Evaluation dataset size: {len(datasets['eval'])}")
else:
    print("Evaluation dataset is not available.")


# --- 5. Define Metrics ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(
        labels, predictions, average="weighted"
    )  # Use "weighted" for multiclass

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


# --- 6. Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eval_strategy=(
        EVALUATION_STRATEGY if datasets.get("eval") else "no"
    ),  # Only eval if eval set exists
    save_strategy=SAVE_STRATEGY,
    logging_dir=LOGGING_DIR,
    logging_steps=LOGGING_STEPS,
    load_best_model_at_end=(
        LOAD_BEST_MODEL_AT_END if datasets.get("eval") else False
    ),  # Only if eval set exists
    metric_for_best_model=(
        METRIC_FOR_BEST_MODEL if datasets.get("eval") else None
    ),  # Only if eval set exists
    report_to="tensorboard",
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
)

# --- 7. Initialize Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets.get("eval"),  # Pass eval dataset, can be None
    tokenizer=tokenizer,
    compute_metrics=(
        compute_metrics if datasets.get("eval") else None
    ),  # Only if eval set exists
)

# --- 8. Train and Evaluate ---
if __name__ == "__main__":
    print("\\nStarting training...")
    trainer.train()

    print("\\nTraining complete.")

    if datasets.get("eval") and trainer.state.best_model_checkpoint:
        print(f"Best model saved to: {trainer.state.best_model_checkpoint}")
        print("\\nEvaluating the best model on the development set...")
        eval_results = trainer.evaluate()
        print("\\nDevelopment Set Evaluation results:")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    elif datasets.get("eval"):
        print(
            "Training finished, but no best model checkpoint found. Evaluating current model state."
        )
        eval_results = trainer.evaluate()
        print("\\nDevelopment Set Evaluation results (current model state):")
        for key, value in eval_results.items():
            print(f"  {key}: {value}")
    else:
        print(
            "Training finished. No evaluation was performed as no development set was available."
        )

    # --- 9. Save the final model and tokenizer ---
    final_model_path = f"{OUTPUT_DIR}/final_model_after_training"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\\nFinal model and tokenizer saved to {final_model_path}")

    # Example of how to make a prediction (after loading a trained model)
    # from transformers import pipeline
    # best_model_to_load = trainer.state.best_model_checkpoint if trainer.state.best_model_checkpoint else final_model_path
    # pipe = pipeline("text-classification", model=best_model_to_load, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    # sample_claim = "The sky is blue."
    # sample_evidence = "Observations show the sky appears blue due to Rayleigh scattering."
    # result = pipe(dict(text=sample_claim, text_pair=sample_evidence))
    # print(f"\\nExample prediction for claim '{sample_claim}' with evidence: {result}")

    # To use the saved model later for inference:
    # tokenizer = AutoTokenizer.from_pretrained(final_model_path)
    # model = AutoModelForSequenceClassification.from_pretrained(final_model_path)
    # ... (rest of the inference example code)
