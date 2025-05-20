import os
import argparse
import pandas as pd

from comparison_library import (facts2preds,
                            get_statistics,
                            question2query,
                            get_shortest,
                            get_tallest,
                            get_candidates,
                            derive_shorter,
                            derive_taller,
                            extend_facts1,
                            extend_facts2,
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


load_path = f"results/comparison_{model_shortname}_shot_{args.shot}_{args.shot_strategy}_{args.split}.tsv"
print(load_path)
if os.path.exists(load_path):
    results_df = pd.read_csv(load_path, sep="\t", index_col=None)


dataset = pd.read_csv('comparison.tsv',sep='\t')
test_labels = dataset[100:]


wrong_list = []
correct_count = 0
all_corr_labels, all_false_labels, all_false_preds = 0, 0, 0
for ind in range(len(results_df)):

    data = results_df.iloc[ind]

    scenario = data["text"]
    output = data["output"] 


    question = test_labels['question'].iloc[ind]
    label = test_labels['label'].iloc[ind]
    fact_labels = test_labels['facts'].iloc[ind]
    fact_labels = fact_labels.split("\n")


    output = output.replace("Facts:", "").strip()
        
    facts = output.split("\n")

    pred_facts = []
    true_facts = []
    
    pred_facts = facts2preds(facts)
    true_facts = facts2preds(fact_labels)

    fact_acc, corr_labels, false_labels, false_preds = get_statistics(pred_facts, true_facts)
    facts = pred_facts

    q_fact = question2query(question)

    cand = get_candidates(pred_facts)
    shortest = get_shortest(pred_facts, cand.copy())
    tallest = get_tallest(pred_facts, cand.copy())

  
    if (shortest not in pred_facts) and (shortest != []):
        pred_facts.append(shortest)
    if (tallest not in pred_facts) and (tallest != []):
        pred_facts.append(tallest)
    pred_facts = extend_facts1(pred_facts)
    pred_facts = extend_facts2(pred_facts)

    new_facts = derive_shorter(pred_facts)
    pred_facts.extend(new_facts)
    new_facts = derive_taller(pred_facts)
    pred_facts.extend(new_facts)

    facts_set = list(map(list, set(map(tuple, pred_facts))))

    if len(q_fact)==2:
        collect = []
        for fa in facts_set:
            if (q_fact[0]==fa[0]) and (len(fa)==2):
                collect.append(fa)


    elif len(q_fact)==3:
        collect = []
        for fa in facts_set:
            if len(fa)==3:
                if (q_fact[1]==fa[1]) and (q_fact[2]==fa[2]):
                    collect.append(fa)

    correct = False
    for co in collect:
        if q_fact[0]==co[0]:
            correct=True
        elif co[0].startswith("not_"):
            pass
        else:
            correct=False
            break

    if correct == True:
        correct_count = correct_count+1
    else:
        wrong_list.append(ind)

    all_corr_labels = all_corr_labels + corr_labels
    all_false_labels = all_false_labels + false_labels
    all_false_preds = all_false_preds + false_preds


tot_precision = all_corr_labels / (all_corr_labels+all_false_preds)
tot_recall = all_corr_labels / (all_corr_labels+all_false_labels)

print("Correct count: ",correct_count)
print("==============================")
print(f"Total Precision: {tot_precision:.3f}")
print(f"Total Recall: {tot_recall:.3f}")
acc = correct_count/len(results_df)
print(f"Accuracy: {acc:.3f}")
print(load_path)

