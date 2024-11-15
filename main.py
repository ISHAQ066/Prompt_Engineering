import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from typing import Dict, List
from huggingface_hub import login

# Replace 'YOUR_TOKEN_HERE' with your actual Hugging Face token
login(token="YOUR_TOKEN_HERE")


def load_data(file_path: str, num_samples: int = 100) -> List[Dict]:
    """
  Load a subset of the JSON data from a file.

  Args:
  file_path (str): Path to the JSON file.
  num_samples (int): Number of samples to load.

  Returns:
  List[Dict]: Loaded JSON data as a list of dictionaries.
  """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[:num_samples]  # Limit to num_samples for testing


def evaluate_model(data: List[Dict], predictions: Dict) -> float:
    """
  Evaluate the model's performance on a dataset.

  Args:
  data (List[Dict]): The dataset to evaluate on (can be validation or test set).
  predictions (Dict): The model's predictions for the dataset.

  Returns:
  float: Accuracy of the model.
  """
    correct = 0
    total = len(data)

    for idx, question_data in enumerate(data):
        if idx in predictions:
            # Normalize answers by stripping whitespace and converting to lowercase
            predicted_answer = predictions[idx].strip().lower()
            true_answer = question_data['answer'].strip().lower()
            if predicted_answer == true_answer:
                correct += 1

    accuracy = correct / total
    return accuracy


def load_model(model_name: str):
    """
  Load the tokenizer and model with 8-bit quantization using bitsandbytes.

  Args:
  model_name (str): Name or path of the pre-trained model.

  Returns:
  tokenizer, model: Loaded tokenizer and model.
  """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Load the model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,  # Enable 8-bit loading
            device_map="auto",  # Automatically map to available devices
        )
        model.eval()
        print("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print("Error loading model or tokenizer:", e)
        return None, None


def inference_naive(data: List[Dict], tokenizer, model) -> Dict:
    """
  Generate predictions using the Llama 3.2 - 3B instruct model with 8-bit quantization.

  Args:
  data (List[Dict]): The dataset containing questions.
  tokenizer: The tokenizer for the model.
  model: The loaded model.

  Returns:
  Dict: A dictionary mapping question IDs to model-generated answers.
  """
    predictions = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for idx, item in enumerate(tqdm(data, desc="Processing questions")):
        question = item.get('question', '').strip()
        if not question:
            predictions[idx] = ""
            continue

        # Craft the prompt for the model
        prompt = f"Answer the following telecom-related question:\n\nQuestion: {question}\nAnswer:"

        try:
            # Tokenize the input prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate the answer
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.2,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode the generated tokens to string
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer part after the prompt
            answer = answer.replace(prompt, "").strip()

            # Store the prediction
            predictions[idx] = answer

        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM when processing question {idx}. Trying to clear cache and retry.")
            torch.cuda.empty_cache()
            predictions[idx] = ""
        except Exception as e:
            print(f"Error generating answer for question {idx}: {e}")
            predictions[idx] = ""

    return predictions


def main():
    # Load a subset of validation data (e.g., 100 samples)
    data_path = "./data/val_data.json"
    data = load_data(data_path, num_samples=5)

    print("Loading model...")
    model_name = "meta-llama/Llama-3.2-3B-instruct"  # Replace with actual model name/path
    tokenizer, model = load_model(model_name)

    if tokenizer and model:
        print("Starting inference...")
        predictions = inference_naive(data, tokenizer, model)

        pred_responses = list(predictions.values())

        for response in pred_responses:
            print(response)

        # for pred in predictions[:1]:
        #     print(pred)

        # Evaluate the predictions
        accuracy = evaluate_model(data, predictions)
        print(f"Model accuracy on {len(data)} samples: {accuracy * 100:.2f}%")


# Example usage
if __name__ == "__main__":
    main()
