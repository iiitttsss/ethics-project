import itertools

import openai
import math

from api_keys import OPENAI_API_KEY
import re
import pandas as pd
import matplotlib.pyplot as plt
import csv


def generate_all_prompts():
    # load prompt template from "ethics_final_prompt.txt"
    with open("ethics_final_prompt.txt", "r") as file:
        prompt_template = file.read().strip()
        prompt_template += " "

    # the template has place marked with {option1\option2\...}
    # for each combination of options, we will generate a prompt

    # extract all the strings that start with { and end with }
    options_raw = re.findall(r'\{([^}]+)\}', prompt_template)
    # split the options by \ and remove whitespace
    options_split = [option.split("\\") for option in options_raw]

    possible_prompts = []
    for combination in itertools.product(*options_split):
        # replace the placeholders in the prompt template with the combination
        prompt = prompt_template
        for option, value in zip(options_raw, combination):
            prompt = prompt.replace(f"{{{option}}}", value.strip())
        possible_prompts.append((prompt, combination))

    return possible_prompts


def generate_probabilities():
    possible_prompts = generate_all_prompts()

    # Initialize OpenAI API
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Prepare CSV file
    with open("ethics_final_dataset.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write header
        header = ["id", "prompt", "question", "marriage status", "city", "religion", "age", "sexual orientation", "yes probability",
                  "no probability", "other probability"]
        writer.writerow(header)
        print(*header, sep="\t")

        for prompt_index, (prompt, combination) in enumerate(possible_prompts):
            temperature = 1.0  # Adjust for randomness
            response = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=1,
                logprobs=20,
            )

            positive_prob = 0
            negative_prob = 0
            other_prob = 0

            if response.choices:
                logprobs = response.choices[0].logprobs.top_logprobs[0]

                # Convert natural log probabilities to standard probabilities
                probs = {token: math.exp(logprobs[token] / temperature) for token in logprobs}
                total_prob = sum(probs.values())
                normalized_probs = {token: prob / total_prob for token, prob in probs.items()}

                sorted_probs = sorted(normalized_probs.items(), key=lambda x: x[1], reverse=True)

                for token, prob in sorted_probs:
                    token = token.strip().lower()
                    if "yes" in token:
                        positive_prob += prob
                    elif "no" in token:
                        negative_prob += prob
                    else:
                        other_prob += prob

            row = [prompt_index, prompt, *combination, positive_prob, negative_prob, other_prob]
            print(*row, sep="\t")
            writer.writerow(row)


def analyze_probabilities():
    df = pd.read_csv("ethics_final_dataset.csv", header=0)

    # group by the following attributes
    attributes_to_group_by = ["question", "marriage status", "city", "religion", "age"]
    attribute_to_evaluate = "sexual orientation"

    # find all unique combinations of the values for the attributes in attributes_to_group_by
    unique_combinations = df[attributes_to_group_by].drop_duplicates()

    z_scores = {}

    for _, unique_combination in unique_combinations.iterrows():
        # filter the dataframe by the unique combination
        filtered_df = df
        for attr in attributes_to_group_by:
            filtered_df = filtered_df[filtered_df[attr] == unique_combination[attr]]

        # for each row print the attribute_to_evaluate and the yes probability
        # print(f"Analysis for combination: {unique_combination.to_dict()}")
        for key, value in unique_combination.to_dict().items():
            print(f"{key}: {value}")
        mean_yes_prob = filtered_df["yes probability"].mean()
        std_yes_prob = filtered_df["yes probability"].std()
        delta_result = {}
        for _, row in filtered_df.iterrows():
            z_score = (row['yes probability'] - mean_yes_prob) / std_yes_prob if std_yes_prob > 0 else 0
            print(f"{row[attribute_to_evaluate]}: {z_score:.4f}")
            delta_result[row[attribute_to_evaluate]] = z_score
            if row[attribute_to_evaluate] not in z_scores:
                z_scores[row[attribute_to_evaluate]] = []
            z_scores[row[attribute_to_evaluate]].append(z_score)
        print()

    # display the z-scores as an histogram
    plt.figure(figsize=(10, 6))
    for orientation, scores in z_scores.items():
        plt.hist(scores, bins=30, alpha=0.5, label=orientation)
        print(f"{orientation} mean z-score: {sum(scores) / len(scores):.4f}")
    plt.title("Z-Scores Distribution by Sexual Orientation")
    plt.xlabel("Z-Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    # make the x axis by in 0.25 jumps
    plt.xticks([i * 0.2 for i in range(-8, 8)])
    plt.show()


if __name__ == '__main__':
    generate_probabilities()
    analyze_probabilities()