Ethics Project: Bias in Language Models
Group 5 - Bias in Language Models

This project investigates potential biases in OpenAI's gpt-3.5-turbo-instruct language model. Specifically, we analyze how different demographic attributes in prompts may influence the model’s responses, using probability distributions derived from model completions.

Project Overview
The core objective is to identify if and how bias manifests when altering attributes such as:
Sexual Orientation
Religion
Age
Marital Status
City of Residence

We generate prompts with controlled variations and evaluate the model’s probabilistic output for response options like "yes", "no", or "other".


Prompt Generation
Prompts are created from a template file (ethics_final_prompt.txt) using placeholders like {option1\option2}. The script generates all possible combinations using itertools.product.

Model Evaluation
Each prompt is submitted to the gpt-3.5-turbo-instruct model via the OpenAI API. The script records the log probabilities of different token completions and aggregates them into three categories:

Yes Probability
No Probability
Other Probability

Data Collection
All prompts and their corresponding results are stored in ethics_final_dataset.csv for further analysis.

Statistical Analysis
Using pandas and matplotlib, we compute the difference in "yes" probabilities from the average (for the same profile) across different demographic groupings to detect outliers or bias patterns. Results are visualized using histograms grouped by sexual orientation.

API
The API key is not provided for obvius reason. The API key can be added to the file api_keys.py in order to execute the code.
