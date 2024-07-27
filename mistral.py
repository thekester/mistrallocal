import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import os
from tqdm import tqdm
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# Define the current directory and model directory
current_dir = os.getcwd()
model_dir = os.path.join(current_dir)  # Do not change this

# Check if the model directory exists and contains the necessary files
required_files = ["params.json", "consolidated.safetensors"]
for file in required_files:
    if not os.path.exists(os.path.join(model_dir, file)):
        raise FileNotFoundError(f"Required file {file} not found in the model directory {model_dir}.")

# Load the tokenizer and model
tokenizer = MistralTokenizer.from_file("tokenizer.model.v3")  # Do not change this
model = Transformer.from_folder(model_dir)  # Load the model from the extracted directory

# Function for generating tokens with sequential batching
def generate_with_sequential_batching(token_batches, model, max_tokens, temperature, eos_id):
    out_batches = []  # List to store the generated sequences
    for tokens in token_batches:
        out_tokens = tokens  # Initialize the output tokens with the input tokens
        # Use tqdm to display a progress bar for each sequence
        for _ in tqdm(range(max_tokens), desc="Generating tokens"):
            # Generate a new token for the current sequence
            new_tokens, _ = generate([out_tokens], model, max_tokens=1, temperature=temperature, eos_id=eos_id)
            # Add the newly generated tokens to the output sequence
            out_tokens.extend(new_tokens[0])
            # Stop generation if the end of sequence token is generated
            if eos_id in new_tokens[0]:
                break
        # Add the complete sequence to the list of generated sequences
        out_batches.append(out_tokens)
    return out_batches

# Define the generation parameters
max_tokens = 25  # Slightly increase the maximum number of generated tokens
temperature = 0.3  # Adjust the temperature for more diversity
eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id

# Welcome message
print("Welcome! This script uses the Mistral model to generate responses based on your prompts.")
print("You will start with a single prompt. After seeing the result, you can decide to add more prompts.")

# Ask the user for the first prompt
prompt = input("Please enter your prompt: ")

# Create a chat completion request for the prompt
completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

# Encode the prompt into tokens
token_batches = [tokenizer.encode_chat_completion(completion_request).tokens]

# Generate the responses with sequential batching
out_batches = generate_with_sequential_batching(token_batches, model, max_tokens, temperature, eos_id)

# Decode the generated results
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_batches[0])

# Print the generated result
print(f"Prompt: {prompt}")
print(f"Generated Response: {result}\n")

# Ask the user if they want to add more prompts
while True:
    add_prompt = input("Would you like to add another prompt? (yes/no): ").strip().lower()
    if add_prompt == 'yes':
        prompt = input("Please enter your prompt: ")
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        token_batches = [tokenizer.encode_chat_completion(completion_request).tokens]
        out_batches = generate_with_sequential_batching(token_batches, model, max_tokens, temperature, eos_id)
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_batches[0])
        print(f"Prompt: {prompt}")
        print(f"Generated Response: {result}\n")
    elif add_prompt == 'no':
        print("Thank you for using the script. Goodbye!")
        break
    else:
        print("Invalid response. Please enter 'yes' or 'no'.")
