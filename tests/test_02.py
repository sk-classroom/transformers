import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re

def validate_json(json_file_path):
    """
    Validate the JSON file according to specified criteria:
    1. Check if model is gemma 27B
    2. Check if last message contains at least 100 normally distributed random variables

    Args:
        json_file_path: Path to the JSON file to validate
    """
    try:
        # Read and parse the JSON file
        with open(json_file_path, 'r') as f:
            data = json.load(f)

        # Check if the JSON structure is valid
        print("✅ JSON is valid")

        # Check if model is gemma 27B
        model_check = check_model(data)

        # Check if the last message has 100+ normally distributed numbers
        numbers_check = check_last_message_numbers(data)

        return model_check and numbers_check

    except json.JSONDecodeError:
        print("❌ Invalid JSON format")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_model(data):
    """Check if the model is gemma 27B"""
    try:
        # Extract the model information
        first_character_id = next(iter(data["characters"]))
        model = data["characters"][first_character_id]["model"]
        model_name = data["characters"][first_character_id]["modelInfo"]["name"]

        # Check if it's Gemma 27B
        is_gemma_27b = "gemma-3-27b" in model.lower() or "gemma 3 27b" in model_name.lower()

        if is_gemma_27b:
            print(f"✅ Model is correctly Gemma 27B: {model}")
            return True
        else:
            print(f"❌ Model is not Gemma 27B. Found: {model}")
            return False

    except KeyError:
        print("❌ Model information not found in expected structure")
        return False

def check_last_message_numbers(data):
    """Check if the last message contains 100+ normally distributed random variables"""
    try:
        # Find the last message from the model (not from USER)
        sorted_messages = sorted(
            [msg for msg_id, msg in data["messages"].items() if msg["characterId"] != "USER"],
            key=lambda x: x["updatedAt"],
            reverse=True
        )

        if not sorted_messages:
            print("❌ No messages from the model found")
            return False

        last_message = sorted_messages[0]
        content = last_message["content"]

        # Extract numbers from the message content
        # Extract numbers from the message content
        # Try to handle both space-separated and comma-separated formats
        numbers_str = re.findall(r'-?\d+\.?\d*', content.replace('\n', ','))
        numbers = [float(num) for num in numbers_str]

        # Check if we have at least 100 numbers
        if len(numbers) < 100:
            print(f"❌ Not enough numbers found in the last message: {len(numbers)} (needed at least 100)")
            return False

        print(f"✅ Found {len(numbers)} numbers in the last message")

        # Perform KS test to check normal distribution
        pval = stats.kstest(numbers, stats.norm(loc=0.0, scale=1.0).cdf)[1]
        if pval > 0.20:
            print(f"✅ The numbers are normally distributed (p-value = {pval:.2f})")
        else:
            print(f"❌ The numbers are not normally distributed (p-value = {pval:.2f})")

        # Plot histogram to visualize distribution
        plt.figure(figsize=(10, 6))
        plt.hist(numbers, bins=30, density=True, alpha=0.7)

        # Plot normal distribution curve for comparison
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, 0, 1)
        plt.plot(x, p, 'k', linewidth=2)

        plt.title('Distribution of Random Numbers in Last Message')
        plt.savefig('distribution_plot.png')
        print("✅ Generated histogram plot saved as distribution_plot.png")

        return True

    except Exception as e:
        print(f"❌ Error analyzing numbers: {str(e)}")
        return False

import sys

json_file_path = "assignment/chat.json"
success = validate_json(json_file_path)

if success:
    print("\n✅ JSON validation successful")
else:
    print("\n❌ JSON validation failed")
    sys.exit(1)