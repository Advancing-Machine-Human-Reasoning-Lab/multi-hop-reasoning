
import os
import pandas as pd
import torch
import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from tqdm import tqdm
from datasets import load_dataset

from transformers import BitsAndBytesConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

argparser = argparse.ArgumentParser()

argparser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Model name. Can be any of the huggingface models.",
)

argparser.add_argument(
    "--shot",
    type=int,
    default=5,
    help="Number of shots parameter in the dataset. Can be any one of: 5, 10, ...",
)

argparser.add_argument(
    "--hop",
    type=int,
    default=3,
    help="Hop parameter in the dataset. Can be any one of: 2, 3, ...",
)

argparser.add_argument(
    "--shot_strategy",
    type=str,
    default="tfidf",
    help="Shot strategy.",
)

args = argparser.parse_args()
print(args)

model_filename = args.model_name
if "/" or ":" in args.model_name:
    model_filename = args.model_name.replace("/","_")
    model_filename = model_filename.replace(":","_")


def rank_tfidf(query, documents):

    all_texts = [query] + documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    return similarities


def create_shots(candidates, best_N, query):

    shot = ""
    cand_indices = candidates[:best_N].index
    for ind in cand_indices:
        sample_story = candidates['Text'].iloc[ind]
        sample_fact = candidates['Facts'].iloc[ind]

        sample = f"Scenario: {sample_story}\n{query}\nFacts:\n{sample_fact}\n\n###\n\n"
        shot = shot + sample
        
    return shot


model_name = args.model_name
quan_model_name = f"{model_filename}_quantized"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)

model, tokenizer = setup_chat_format(model, tokenizer)

model.save_pretrained(quan_model_name)
tokenizer.save_pretrained(quan_model_name)



# DATASET

# References for similarity calculation:
reference_dataset = 'train_facts.tsv'
references = pd.read_csv(reference_dataset, delimiter='\t')
ref_story = references['Text'].tolist()
ref_fact = references['Facts'].tolist()


# Test dataset:
test_size = 100
test_dataset = load_dataset("michaelszx/StepGame", split="test")
indices = [i for i, example in enumerate(test_dataset) if example['k_hop'] == args.hop]
test_data = test_dataset.select(indices)
selected_data = test_data.select(range(test_size))


columns = ['text', 'question', 'label', 'output']
results_df = pd.DataFrame(columns=columns)


save_path = f"results/stepgame_{quan_model_name}_shot_{args.shot}_hop_{args.hop}_{args.shot_strategy}_test.tsv"
if os.path.exists(save_path):
    results_df = pd.read_csv(save_path, sep="\t", index_col=None)
print("Save path: ", save_path)


pbar = tqdm(total=len(selected_data))
pbar.update(len(results_df))

for i in range(len(results_df), len(selected_data)):

    print(f"-------- {i} --------")

    data = selected_data[i]
    story = " ".join(data['story'])
    question = data['question']
    label = data['label']

    print(story)

    if args.shot_strategy == 'tfidf':
        similarities = rank_tfidf(story, ref_story)
        sim_df = pd.DataFrame({
            'Text': ref_story,
            'Facts': ref_fact,
            'Similarity': similarities,
        })   
        sim_df = sim_df.sort_values(by='Similarity', ascending=False)


    system_prompt = "Your task is to find the spatial relationships between the two letters in the scenario. If a sentence is describing clock-wise information, then 12 denotes above, 1 and 2 denote upper-right, 3 denotes right, 4 and 5 denote lower-right, 6 denotes below, 7 and 8 denote lower-left, 9 denotes left, 10 and 11 denote upper-left. If a sentence describes directions, then north denotes above, east denotes right, south denotes below, west denotes left, northeast denotes upper-right, southeast denotes lower-right, southwest denotes lower-left, and northwest denotes upper-left."
    query = "Convert the spatial relationship of the letters in the scenario to the following facts: left, right, above, below, lower-left, lower-right, upper-left, and upper-right."

    shots = create_shots(sim_df, args.shot, query)
    query_cot = f"{query} Let's think step by step."

    user_prompt = f"{shots}Scenario: {story}\n{query_cot}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                    truncation=True).to(device)

    outputs = model.generate(**inputs,
                            max_length=2048,
                            num_return_sequences=1,
                            temperature=0.1,
                            do_sample=True,
                            )
    

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Input token count:", len(inputs['input_ids'][0]), "Output token count:", len(outputs[0]))


    out1 = text.split("assistant")[1]
    
    query2 = "So, respond to the last scenario as in the previous answers. Write only the facts. Write nothing else."

    user_prompt2 = f"{query2}\nFacts:\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": out1},
        {"role": "user", "content": user_prompt2},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, 
                    truncation=True).to(device)


    outputs = model.generate(**inputs, 
                            num_return_sequences=1,
                            max_new_tokens=128,
                            temperature=0.1,
                            do_sample=True,
                            )


    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("Input token count:", len(inputs['input_ids'][0]), "Output token count:", len(outputs[0]))

    out2 = text.split("assistant")[2]

    output = out2
    
    print(output)

    row_result = pd.DataFrame({"text": story, "question": question, 
                               "label": label, "output": output},index=[0])
    results_df = pd.concat([results_df, row_result], axis=0, ignore_index=True)

    results_df.to_csv(save_path, sep="\t", index=False)
    pbar.update(1)

