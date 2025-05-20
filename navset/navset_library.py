
def port2starboard(facts_list):
    """
    Given a list of facts, if the fact ["port", "D", "E"] is present,
    generate the opposite fact ["starboard", "E", "D"].

    Input:
    facts[list]: [["port","D","E"],["engine","D"],["engine","E"]]
    
    Output:
    new_facts[list]: [["starboard","E","D"]]
    """
    new_facts = []
    for fact in facts_list:
        if fact[0] == "port":
            if len(fact)==3:
                _, A, B = fact
                new_fact = ["starboard", B, A]
                new_facts.append(new_fact)
    
    return new_facts


def apply_rule1(facts_list):
    """
    Input:
    facts_list[list]: [["starboard","A","B"],["engine","A"],["engine","B"],["starboard","D","E"],["engine","D"],["engine","E"],["head-on","F","G"]]
    Output:
    conclusions [list]: ["RoW(A)","RoW(D)","RoW(0)"]
    """
    conclusions = []

    # Parse facts
    starboard_pairs = []
    head_on_pairs = []
    engines = set()
    sails = set()
    restricteds = set()

    for fact in facts_list:
        predicate, *args = fact
        if predicate == 'starboard':
            if len(args)==2:
                subj1, subj2 = args
                starboard_pairs.append((subj1, subj2))
        elif predicate == 'engine':
            engines.add(args[0])
        elif predicate == 'sail':
            sails.add(args[0])
        elif predicate == 'restricted':
            restricteds.add(args[0])
        elif predicate == 'head-on':
            try:
                subj1, subj2 = args
                head_on_pairs.append((subj1, subj2))
            except:
                pass

    # Apply the rule: starboard(A,B) & engine(A) & engine(B) --> RoW(A)
    for (A, B) in starboard_pairs:
        if A in engines and B in engines and B not in sails and B not in restricteds:
            conclusion = f"RoW({A})"
            conclusions.append(conclusion)
    
    # Apply the new rule: head-on(A,B) & engine(A) & engine(B) --> RoW(0)
    for (A, B) in head_on_pairs:
        if A in engines and B in engines and B not in sails and B not in restricteds:
            conclusions.append("RoW(0)")
    
    return conclusions




def apply_rule2(facts_list):
    """
    Input:
    facts[list]: [["starboard","A","B"],["engine","A"],["sail","B"],["starboard","D","E"],["sail","D"],["engine","E"],["head-on","F","G"],["engine","F"],["sail","G"]]
    Output:
    conclusions [list]: ["RoW(B)","RoW(A)","RoW(E)","RoW(D)","RoW(G)"]
    """
    conclusions = []

    # Parse facts
    starboard_pairs = []
    head_on_pairs = []
    engines = set() 
    sails = set()
    restricteds = set()

    for fact in facts_list:
        predicate, *args = fact
        if predicate == 'starboard':
            if len(args)==2:
                subj1, subj2 = args
                starboard_pairs.append((subj1, subj2))
        elif predicate == 'engine':
            engines.add(args[0])
        elif predicate == 'sail':
            sails.add(args[0])
        elif predicate == 'restricted':
            restricteds.add(args[0])
        elif predicate == 'head-on':
            try:
                subj1, subj2 = args
                head_on_pairs.append((subj1, subj2))
            except:
                pass

    # Apply the rule: starboard(A,B) & engine(A) & sail(B) --> RoW(B)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in engines and B in sails and A not in sails and A not in restricteds:
            conclusion = f"RoW({B})"
            conclusions.append(conclusion)
    
    # Apply the rule: starboard(A,B) & sail(A) & engine(B) --> RoW(A)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in sails and B in engines and B not in sails and B not in restricteds:
            conclusion = f"RoW({A})"
            conclusions.append(conclusion)

    return conclusions




def apply_rule3(facts_list):
    """
    Input:
    facts[list]: [["starboard","A","B"],["engine","A"],["restricted","B"],["starboard","D","E"],["sail","D"],["restricted","E"],["starboard","F","G"],["restricted","F"],["engine","G"],["starboard","H","I"],["restricted","H"],["sail","I"]]
    Output:
    conclusions [list]: ["RoW(B)","RoW(E)","RoW(A)","RoW(A)"]
    """
    conclusions = []

    # Parse facts
    starboard_pairs = []
    head_on_pairs = []
    restricteds = set()
    sails = set()
    engines = set()

    for fact in facts_list:
        predicate, *args = fact
        if predicate == 'starboard':
            if len(args)==2:
                subj1, subj2 = args
                starboard_pairs.append((subj1, subj2))
        elif predicate == 'head-on':
            try:
                subj1, subj2 = args
                head_on_pairs.append((subj1, subj2))
            except:
                pass
            
        elif predicate == 'restricted':
            restricteds.add(args[0])
        elif predicate == 'sail':
            sails.add(args[0])
        elif predicate == 'engine':
            engines.add(args[0])
    
    # Apply the rules:
    # Rule 1: starboard(A,B) & engine(A) & restricted(B) --> RoW(B)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in engines and B in restricteds and A not in restricteds:
            conclusion = f"RoW({B})"
            conclusions.append(conclusion)
    
    # Rule 2: starboard(A,B) & sail(A) & restricted(B) --> RoW(B)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in sails and B in restricteds and A not in restricteds:
            conclusion = f"RoW({B})"
            conclusions.append(conclusion)

    # Rule 3: starboard(A,B) & restricted(A) & engine(B) --> RoW(A)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in restricteds and B in engines and B not in restricteds:
            conclusion = f"RoW({A})"
            conclusions.append(conclusion)

    # Rule 4: starboard(A,B) & restricted(A) & sail(B) --> RoW(A)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in restricteds and B in sails and B not in restricteds:
            conclusion = f"RoW({A})"
            conclusions.append(conclusion)

    # Rule 5: starboard(A,B) & restricted(A) & restricted(B) --> RoW(0)
    for (A, B) in starboard_pairs or head_on_pairs:
        if A in restricteds and B in restricteds:
            conclusion = "RoW(0)"
            conclusions.append(conclusion)

    return conclusions



def facts2preds(facts):

    preds = []
    for fact in facts:
        if fact.strip():
            # Check if the fact contains multiple arguments
            if fact.count('(') > 1:
                continue
            try:
                preds.append(fact.split("(")[0].strip().split() + fact.split("(")[1].replace(")", "").split(","))
            except (IndexError, AttributeError):
                pass
    
    return preds


def get_symmetrics(fact):
    """
    Get the symmetric fact.
    Input:
    facts [list]: ['port', 'A', 'B']
    Output:
    sym_facts [list]: ['starboard', 'B', 'A']
    """
    locs_map = {'port':'starboard', 'starboard':'port', 'head-on':'head-on'}
    reversed_locs_map = {value: key for key, value in locs_map.items()}
    locs_map.update(reversed_locs_map)

    key, value1, value2 = fact
    if key in locs_map:
        new_key = locs_map[key]
        sym_fact = [new_key, value2, value1] 
    else:
        sym_fact = []
    return sym_fact


def get_statistics(y_pred, y_true):

    corr_labels = 0
    for fact in y_true:

        if len(fact) == 3:
        # No need to remove symmetrics from navset 
            sym_fact = get_symmetrics(fact)
        else:
            sym_fact = fact


        if (fact in y_pred) or (sym_fact in y_pred):
            corr_labels = corr_labels + 1

    false_labels = len(y_true) - corr_labels
    false_preds = len(y_pred) - corr_labels
    acc = corr_labels/len(y_true)

    return acc, corr_labels, false_labels, false_preds
