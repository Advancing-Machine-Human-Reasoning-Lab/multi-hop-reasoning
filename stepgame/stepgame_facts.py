import os
import argparse
import pandas as pd

from stepgame_library import *

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


save_path = f"results/stepgame_{model_shortname}_shot_{args.shot}_hop_{args.hop}_{args.shot_strategy}_{args.split}.tsv"


print(save_path)
if os.path.exists(save_path):
    results_df = pd.read_csv(save_path, sep="\t", index_col=None)



def test_query(label_query, fact_preds):

    candidates = ['left', 'right', 'above', 'below', 'upper-left', 'upper-right', 'lower-left', 'lower-right', 'overlap']
    output = False
    for cand in candidates:
        to_test = [cand, label_query[1], label_query[2]]
        if to_test in fact_preds and to_test[0] == label_query[0]:
            output = True

        elif to_test in fact_preds and to_test[0] != label_query[0]:
            output = False
            return output

    return output



def get_statistics(y_pred, y_true):

    corr_labels = 0
    for fact in y_true:

        

        sym_fact = get_symmetrics(fact)

        if (fact in y_pred) or (sym_fact in y_pred):
            corr_labels = corr_labels + 1

    false_labels = len(y_true) - corr_labels
    false_preds = len(y_pred) - corr_labels
    acc = corr_labels/len(y_true)

    return acc, corr_labels, false_labels, false_preds


test_labels = pd.read_csv(f"test_facts_{args.hop}_hops.tsv", delimiter='\t')


correct_count = 0
wrong_list = []
all_corr_labels, all_false_labels, all_false_preds = 0, 0, 0
for i in range(len(results_df)):


    question = results_df['question'].iloc[i]
    label = results_df['label'].iloc[i]
    output = results_df['output'].iloc[i]
    true_facts = test_labels.iloc[i]['facts']
    true_facts = true_facts.strip().split('\n')
    true_facts_list = facts2preds(true_facts)
    true_facts_list = remove_symmetrics(true_facts_list)


    output_facts = remove_facts_text(output)

    facts_list = split_facts_text(output_facts)


    label_query = question2label(question, label)

    fact_preds = facts2preds(facts_list)
    # For calculation of precision, recall metrics:
    pred_facts_list = remove_symmetrics(fact_preds)

    fact_preds = [sublist for sublist in fact_preds if len(sublist) == 3]
    fact_preds = facts_lower(fact_preds)


    try:
        new_facts, to_remove = apply_rules0(fact_preds)
        fact_preds.extend([element for element in new_facts if element not in fact_preds])
        fact_preds = [element for element in fact_preds if element not in to_remove]

    except ValueError:
        pass


    progress = True
    try:
        while progress:


            len_before_loop = len(fact_preds)


            new_facts = get_new_facts(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])


            new_facts = apply_rules1(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])

            
            new_facts = apply_rules4(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])

            len_after_loop = len(fact_preds)
            progress = len_before_loop != len_after_loop


    except ValueError:
        pass


    new_facts3 = apply_rules2(fact_preds)
    unique_list = list(dict.fromkeys(tuple(x) for x in new_facts3))
    new_facts3 = [list(x) for x in unique_list]


    new_facts4 = get_overlaps(fact_preds)
    unique_list = list(dict.fromkeys(tuple(x) for x in new_facts4))
    new_facts4 = [list(x) for x in unique_list]

    fact_preds.extend([element for element in new_facts3 if element not in fact_preds])
    fact_preds.extend([element for element in new_facts4 if element not in fact_preds])


    progress3 = True
    try:
        while progress3:


            len_before_loop = len(fact_preds)


            new_facts = get_new_facts(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])


            new_facts = apply_rules1(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])

            
            new_facts = apply_rules4(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])


            len_after_loop = len(fact_preds)
            progress3 = len_before_loop != len_after_loop


    except ValueError:
        pass


    progress4 = True
    try:
        while progress4:


            len_before_loop = len(fact_preds)

            # NO RETURN FACTS
            new_facts = get_same_relations(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])


            new_facts = get_new_facts(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])

            # NO RETURN FACTS
            new_facts = apply_rules3(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])


            new_facts = get_new_facts(fact_preds)
            unique_list = list(dict.fromkeys(tuple(x) for x in new_facts))
            new_facts = [list(x) for x in unique_list]
            fact_preds.extend([element for element in new_facts if element not in fact_preds])

            len_after_loop = len(fact_preds)
            progress4 = len_before_loop != len_after_loop


    except ValueError:
        pass


    out = test_query(label_query, fact_preds)
    if out == True:
        correct_count = correct_count + 1
    else:
        wrong_list.append(i)
    
    fact_acc, corr_labels, false_labels, false_preds = get_statistics(pred_facts_list, true_facts_list)

    all_corr_labels = all_corr_labels + corr_labels
    all_false_labels = all_false_labels + false_labels
    all_false_preds = all_false_preds + false_preds


tot_precision = all_corr_labels / (all_corr_labels+all_false_preds)
tot_recall = all_corr_labels / (all_corr_labels+all_false_labels)
print("==============================")
print(f"Total Precision: {tot_precision:.3f}")
print(f"Total Recall: {tot_recall:.3f}")


acc = correct_count/len(results_df)
print("Correct count: ",correct_count, "Accuracy: ", acc)
print(save_path)





