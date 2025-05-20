
import re



def get_overlaps(relations):
    """
    Input:
    relations [list]: [['upper-left', 'A', 'B'], ['upper-left', 'C', 'B'], ['left', 'A', 'D'], ['left', 'E', 'D'], ['right', 'A', 'E']] 
    Output:
    overlaps [list]: [['overlap', 'A', 'C'], ['overlap', 'A', 'E']]
    """
    overlaps = []
    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            # Check if both relations have the same position and the same letter in position 2, but different letters in position 1
            if relations[i][0] == relations[j][0] and relations[i][2] == relations[j][2] and relations[i][1] != relations[j][1]:
                A = relations[i][1]
                B = relations[j][1]
                common_letter = relations[i][2]
                
                # Check if there's no other direct relationship involving both A and B
                has_other_relation = False
                for relation in relations:
                    if relation != relations[i] and relation != relations[j]:
                        if (A in relation and B in relation) or (B in relation and A in relation):
                            has_other_relation = True
                            break
                
                # Only add to overlaps if no other relation exists between A and B
                if not has_other_relation:
                    overlaps.append(['overlap', A, B])

    return overlaps


def get_same_relations(relations):
    """
    Input:
    relations [list]: [['upper-left', 'A', 'B'], ['upper-left', 'C', 'A'], ['left', 'A', 'D'], ['left', 'E', 'A'], ['right', 'A', 'E']]
    Output:
    same_relations [list]: [['upper-left', 'C', 'B'], ['left', 'E', 'D']]
    """
    same_relations = []
    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            if relations[i][0] == relations[j][0] and relations[i][1] == relations[j][2] and relations[j][1]!=relations[i][2]:
                same_relations.append([relations[i][0], relations[j][1], relations[i][2]])

    return same_relations


def facts2preds(facts_list):
    """
    Input:
    facts_list [list]: ["upper-left(A,B)", "upper-left(C,B)", "left(A,D)", "left(E,D)", "right(A,E)"]
    Output:
    preds [list]: [['upper-left', 'A', 'B'], ['upper-left', 'C', 'B'], ['left', 'A', 'D'], ['left', 'E', 'D'], ['right', 'A', 'E']] 
    """
    preds = []
    for fact in facts_list:
        parsed_str = [s.strip() for s in fact.replace("(", ",").replace(")", "").split(",")]
        preds.append(parsed_str)

    return preds


def get_new_facts(facts_list):
    """
    Input:
    facts_list [list]: [['left', 'A', 'B'], ['upper-left', 'C', 'B'], ['overlap', 'C', 'E']]
    Output:
    new_facts [list]: [['right', 'B', 'A'], ['lower-right', 'C', 'B'], ['overlap', 'E', 'C']]
    """
    locs_map = {'left':'right', 'above':'below', 'upper-left':'lower-right', 'upper-right':'lower-left', 'overlap':'overlap'}
    reversed_locs_map = {value: key for key, value in locs_map.items()}
    locs_map.update(reversed_locs_map)

    new_facts = []
    for item in facts_list:
        key, value1, value2 = item
        if key in locs_map:
            new_key = locs_map[key]
            new_fact = [new_key, value2, value1]
            new_facts.append(new_fact)

    return new_facts


def apply_rules0(facts_list):
    """
    Input:
    facts[list]: ["above(A,C)","left(A,C)","below(D,F)","right(D,F)","left(E,B)","below(D,G)"]
    Output:
    conclusions [list]: [['upper-left', 'A', 'C'], ['lower-right', 'D', 'F']]
    to_remove [list]: [['above', 'A', 'C'], ['left', 'A', 'C'], ['below', 'D', 'F'], ['right', 'D', 'F']]
    """
    conclusions = []  
    to_remove = []  
    # Define the rules
    rules = {
        ("above", "left"): "upper-left",
        ("above", "right"): "upper-right",
        ("below", "left"): "lower-left",
        ("below", "right"): "lower-right",
    }

    for key1, value1a, value1b in facts_list:
        for key2, value2a, value2b in facts_list:
            if key1 in ["above", "below"] and key2 in ["left", "right"]:
                if value1a == value2a and value1b == value2b:
                    combined_key = (key1, key2)
                    if combined_key in rules:
                        conclusion = [rules[combined_key], value1a, value1b]
                        conclusions.append(conclusion)
                        to_remove.append([key1, value1a, value1b])
                        to_remove.append([key2, value2a, value2b])

    return conclusions, to_remove


def apply_rules1(facts_list):
    """
    Input:
    facts[list]: ["above(A,C)","left(C,B)","below(D,F)","right(F,E)","left(E,B)","below(D,G)"]
    Output:
    conclusions [list]: ["upper-left(A,B)", "lower-right(D,E)"]
    """
    conclusions = []    
    # Define the rules
    rules = {
        ("above", "left"): "upper-left",
        ("above", "right"): "upper-right",
        ("below", "left"): "lower-left",
        ("below", "right"): "lower-right",
    }

    for key1, value1a, value1b in facts_list:
        for key2, value2a, value2b in facts_list:
            if key1 in ["above", "below"] and key2 in ["left", "right"]:
                if value1b == value2a:
                    combined_key = (key1, key2)
                    if combined_key in rules and value1a!=value2b:
                        conclusion = [rules[combined_key], value1a, value2b]
                        conclusions.append(conclusion)
    
    return conclusions



def apply_rules2(facts_list):
    """
    NON-DETERMINISTIC RULE
    Input:
    facts[list]: ["upper-left(A,C)","left(B,C)"]
    Output:
    conclusions [list]: ["above(A,B)"]
    """
    conclusions = []    

    existing_relations = set((value1a, value1b) for key1, value1a, value1b in facts_list)

    # Define the rules
    rules = {
        ("upper-left", "left"): "above",
        ("upper-right", "right"): "above",
        ("upper-left", "above"): "left",
        ("upper-right", "above"): "right",

        ("lower-left", "left"): "below",
        ("lower-right", "right"): "below",
        ("lower-left", "below"): "left",
        ("lower-right", "below"): "right",

        ("lower-left", "upper-left"): "below",
        ("lower-right", "upper-right"): "below",
        ("lower-left", "lower-right"): "left",
        ("upper-left", "upper-right"): "left",
    }

    for key1, value1a, value1b in facts_list:
        for key2, value2a, value2b in facts_list:
            if key1 in ["upper-left", "upper-right", "lower-left", "lower-right"] and key2 in ["left", "right", "above", "below", "upper-left", "upper-right", "lower-right"]:
                if value1b == value2b:
                    combined_key = (key1, key2)
                    if combined_key in rules and value1a!=value2a and (value1a, value2a) not in existing_relations:
                        conclusion = [rules[combined_key], value1a, value2a]
                        conclusions.append(conclusion)
    
    return conclusions


def apply_rules3(facts_list):
    """
    Input:
    facts[list]: [""upper-left"(A,C)","left(C,B)","lower-right(D,F)","right(F,E)","left(E,B)","below(D,G)"]
    Output:
    conclusions [list]: ["upper-left(A,B)", "lower-right(D,E)"]
    """
    conclusions = []    
    # Define the rules
    rules = {
        ("upper-left", "left"): "upper-left",
        ("upper-right", "right"): "upper-right",
        ("upper-left", "above"): "upper-left",
        ("upper-right", "above"): "upper-right",

        ("lower-left", "left"): "lower-left",
        ("lower-right", "right"): "lower-right",
        ("lower-left", "below"): "lower-left",
        ("lower-right", "below"): "lower-right",  
    }

    for key1, value1a, value1b in facts_list:
        for key2, value2a, value2b in facts_list:
            if key1 in ["upper-left", "upper-right", "lower-left", "lower-right"] and key2 in ["left", "right", "above", "below", "upper-left", "upper-right", "lower-left", "lower-right"]:
                if value1a == value2b:
                    combined_key = (key1, key2)
                    if combined_key in rules and value2a!=value1b:
                        conclusion = [rules[combined_key], value2a, value1b]
                        conclusions.append(conclusion)
    
    return conclusions



def apply_rules4(facts_list):
    """
    Input:
    facts[list]: ["overlap(A,C)","left(C,B)","below(D,F)","overlap(F,E)","left(E,B)","below(D,G)"]
    Output:
    conclusions [list]: ["left(A,B)", "below(D,E)"]
    """    
    conclusions = []    
    # Define the rules
    rules = {
        ("overlap", "left"): "left",
        ("overlap", "right"): "right",
        ("overlap", "above"): "above",
        ("overlap", "below"): "below",

        ("overlap", "upper-left"): "upper-left",
        ("overlap", "upper-right"): "upper-right",
        ("overlap", "lower-left"): "lower-left",
        ("overlap", "lower-right"): "lower-right",
    }

    for key1, value1a, value1b in facts_list:
        for key2, value2a, value2b in facts_list:
            if key1 in ["overlap"] and key2 in ["left", "right", "above", "below", "upper-left", "upper-right", "lower-left", "lower-right"]:
                if value1b == value2a:
                    combined_key = (key1, key2)
                    if combined_key in rules and value1a!=value2b:
                        conclusion = [rules[combined_key], value1a, value2b]
                        conclusions.append(conclusion)
    
    return conclusions



def remove_facts_text(text):
    """
    Input:
    text [str]: Fact:\nbelow(F,U)\nleft(X,U) 
    Output:
    new_text [str]: below(F,U)\nleft(X,U) 
    """
    if text.endswith('\n'):
        text = text.replace('\n', '')

    patterns = [
        r'Facts:\n?',  # Matches "Facts:", "Facts:\n", "Facts\n"
        r'Fact:\n?',   # Matches "Fact:", "Fact:\n"
    ]
    # Join patterns into a single regular expression pattern
    pattern = '|'.join(patterns)
    match = re.search(pattern, text)
    if match:
        split_text = re.split(pattern, text, flags=re.IGNORECASE)
        new_text = split_text[1] if len(split_text) > 1 else ''

        return new_text
    else:
        return text



def split_facts_text(facts):
    """
    Input:
    facts [str]: below(F,U)\nleft(X,U)
    Output:
    facts_list [list]: ['below(F,U)', 'left(X,U)']
    facts_list [list]: [['below', 'F', 'U'], ['left', 'X', 'U']]
    """
    facts = facts.replace(")",")\n")

    facts = facts.replace("`", "")
    if facts.startswith("-"):
        facts = facts.replace("- ", "")

    facts_list = facts.split("\n")
    facts_list = [facts.strip() for facts in facts_list if facts != '']

    return facts_list



def get_symmetrics(fact):
    """
    Get the symmetric fact.
    Input:
    facts [list]: ['left', 'A', 'B']
    Output:
    sym_facts [list]: ['right', 'B', 'A']
    """
    locs_map = {'left':'right', 'above':'below', 'upper-left':'lower-right', 'upper-right':'lower-left', 'overlap':'overlap'}
    reversed_locs_map = {value: key for key, value in locs_map.items()}
    locs_map.update(reversed_locs_map)

    if len(fact)==3:
        key, value1, value2 = fact
        if key in locs_map:
            new_key = locs_map[key]
            sym_fact = [new_key, value2, value1] 
        else:
            sym_fact = []
    else:
        sym_fact = []
    return sym_fact


def remove_symmetrics(facts_list):
    """
    Remove symmetric facts from facts_list.
    Input:
    facts_list [list]: [['left', 'A', 'B'], ['lower-left', 'A', 'E'], ['above', 'D', 'S'], ['below', 'S', 'D'], 
    ['upper-right', 'D', 'F'], ['right', 'A', 'S']]
    Output:
    facts_list [list]: [['left', 'A', 'B'], ['lower-left', 'A', 'E'], ['above', 'D', 'S'], ['upper-right', 'D', 'F'], 
    ['right', 'A', 'S']]
    """
    locs_map = {'left':'right', 'above':'below', 'upper-left':'lower-right', 'upper-right':'lower-left', 'overlap':'overlap'}
    reversed_locs_map = {value: key for key, value in locs_map.items()}
    locs_map.update(reversed_locs_map)

    for fact in facts_list:
        if len(fact)==3:
            sym_fact = get_symmetrics(fact)
            if sym_fact in facts_list:
                facts_list.remove(sym_fact)

    return facts_list


def facts_lower(fact_preds):   

    for sub_list in fact_preds:
        sub_list[0] = sub_list[0].lower()
    
    return fact_preds


def question2label(question, label):

    pattern = r'agent\s*([A-Z])\s*to\s*the\s*agent\s*([A-Z])'
    matches = re.findall(pattern, question)
    label_query = [label,matches[0][0],matches[0][1]]

    return label_query