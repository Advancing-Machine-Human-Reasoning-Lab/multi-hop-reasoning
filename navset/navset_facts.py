import os
import re
import argparse
import pandas as pd

from navset_library import (port2starboard, 
                          apply_rule1, 
                          apply_rule2, 
                          apply_rule3,
                          facts2preds,
                          get_statistics,
                          )

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
    "--shot_strategy",
    type=str,
    default="tfidf",
    help="Shot strategy.",
)

argparser.add_argument(
    "--split",
    type=str,
    default="test",
    help="Split in the dataset.",
)


args = argparser.parse_args()
print(args)

model_shortname = args.model_name
if "/" in args.model_name:
    model_shortname = args.model_name.split("/")[1]
if ":" in model_shortname:
    model_shortname = model_shortname.replace(":","_")


load_path = f"results/navset_{model_shortname}_shot_{args.shot}_{args.shot_strategy}_{args.split}.tsv"
print(load_path)
if os.path.exists(load_path):
    results_df = pd.read_csv(load_path, sep="\t", index_col=None)



def extract_row(row_list):
    match = re.search(r'\((.*?)\)', row_list[0])
    if match:
        extracted_letter = match.group(1)
        return extracted_letter



dataset = pd.read_csv('navset.tsv',sep='\t')
test_labels = dataset[dataset["split"] == args.split]


correct = 0
all_corr_labels, all_false_labels, all_false_preds = 0, 0, 0
for ind in range(len(results_df)):

    data = results_df.iloc[ind]

    scenario = data["text"]
    output = data["output"]
    label = data["label"]

    fact_labels = test_labels['facts'].iloc[ind]
    fact_labels = fact_labels.split("\n")
    
    output = output.replace("Facts:", "").strip()
    
    pred_facts = []
    true_facts = []
    facts = output.split("\n")


    pred_facts = facts2preds(facts)
    true_facts = facts2preds(fact_labels)


    fact_acc, corr_labels, false_labels, false_preds = get_statistics(pred_facts, true_facts)


    new_facts1 = port2starboard(pred_facts)
    pred_facts.extend(new_facts1)
    
    rows = []

    row1 = apply_rule1(pred_facts)
    row2 = apply_rule2(pred_facts)
    row3 = apply_rule3(pred_facts)

    rows.extend(row1)
    rows.extend(row2)
    rows.extend(row3)
    rows = list(set(rows))

    if len(rows) in [0,2]:
        predicted_row = ""
    else:
        predicted_row = extract_row(rows)

    if predicted_row == label:
        correct = correct + 1

    all_corr_labels = all_corr_labels + corr_labels
    all_false_labels = all_false_labels + false_labels
    all_false_preds = all_false_preds + false_preds

tot_precision = all_corr_labels / (all_corr_labels+all_false_preds)
tot_recall = all_corr_labels / (all_corr_labels+all_false_labels)
print("==============================")
print(f"Total Precision: {tot_precision:.3f}")
print(f"Total Recall: {tot_recall:.3f}")

acc = correct/len(results_df)

print(f"Accuracy: {acc}")
print(load_path)