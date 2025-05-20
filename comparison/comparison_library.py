


def facts2preds(facts):

    preds = []
    for fact in facts:

        if fact.strip():
            # Check if the fact contains multiple arguments (indicated by multiple '(' and ')' pairs)
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
    facts [list]: ['shorter', 'A', 'B']
    Output:
    sym_facts [list]: ['taller', 'B', 'A']
    """
    locs_map = {'taller':'shorter', 'shorter':'taller'}
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


def get_statistics(y_pred, y_true):

    corr_labels = 0
    for fact in y_true:

        if len(fact) == 3:
            sym_fact = get_symmetrics(fact)
        else:
            sym_fact = fact


        if (fact in y_pred) or (sym_fact in y_pred):
            corr_labels = corr_labels + 1

    false_labels = len(y_true) - corr_labels
    false_preds = len(y_pred) - corr_labels
    acc = corr_labels/len(y_true)

    return acc, corr_labels, false_labels, false_preds



def parse_first_word(question):
    """
    Parse the first word after 'Is' in the given question.
    """
    words = question.split()
    if len(words) > 1 and words[0].lower() == "is":
        return [words[3], words[1]]
    return None


def parse_comparison(sentence):
    """
    Parse the comparative statement into [comparison, entity1, entity2].
    For example, "Is E taller than A?" -> [taller, E, A].
    """
    words = sentence.strip(" ?").split()
    if len(words) >= 4 and words[0].lower() == "is" and words[3].lower() == "than":
        return [words[2], words[1], words[4]]
    return None


def question2query(question):


    if ('shortest' in question) or ('tallest' in question):
        q_fact = parse_first_word(question)

    if ('shorter' in question) or ('taller' in question):
        q_fact = parse_comparison(question)

    return q_fact


def get_candidates(facts):

    cand = set()
    for fact in facts:
        if len(fact)==2:
            cand.add(fact[1])
        if len(fact)==3:
            cand.add(fact[1])
            cand.add(fact[2])

    return cand


def get_shortest(facts, candidates):

    for fact in facts:

        if (fact[0]=='shorter') and (len(fact)==3):
            if fact[2] in candidates:
                candidates.remove(fact[2])
        elif (fact[0]=='taller')  and (len(fact)==3):
            if fact[1] in candidates:
                candidates.remove(fact[1])
        elif (fact[0]=='not_shorter')  and (len(fact)==3):
            if fact[1] in candidates:
                candidates.remove(fact[1])
        elif (fact[0]=='not_taller') and (len(fact)==3):
            if fact[2] in candidates:
                candidates.remove(fact[2])
        elif (fact[0]=='tallest') and (len(fact)==2):
            if fact[1] in candidates:
                candidates.remove(fact[1])
        elif (fact[0]=='not_shortest') and (len(fact)==2):
            if fact[1] in candidates:
                candidates.remove(fact[1])

    if len(candidates) == 1:        
        remain = ['shortest', list(candidates)[0]]
    else: 
        remain = []

    return remain


def get_tallest(facts, candidates):

    for fact in facts:
        if (fact[0]=='shorter') and (len(fact)==3):
            if fact[1] in candidates:
                candidates.remove(fact[1])
        elif (fact[0]=='taller') and (len(fact)==3):
            if fact[2] in candidates:
                candidates.remove(fact[2])
        elif (fact[0]=='not_shorter') and (len(fact)==3):
            if fact[2] in candidates:
                candidates.remove(fact[2])
        elif (fact[0]=='not_taller') and (len(fact)==3):
            if fact[1] in candidates:
                candidates.remove(fact[1])
        elif fact[0]=='shortest' and (len(fact)==2):
            if fact[1] in candidates:
                candidates.remove(fact[1])
        elif fact[0]=='not_tallest' and (len(fact)==2):
            if fact[1] in candidates:
                candidates.remove(fact[1])

    if len(candidates) == 1:        
        remain = ['tallest', list(candidates)[0]]
    else: 
        remain = []

    return remain


def derive_shorter(facts):
    """
    Derive ['shorter', Y, Z] from the given facts using the transitive property.
    """
    result = []
    for fact1 in facts:
        for fact2 in facts:
            if (fact1[0] == 'taller') and (fact2[0] == 'shorter') and (fact1[1] == fact2[1]) and (len(fact1)==3) and (len(fact2)==3):
                # Match: ['taller', X, Y] and ['shorter', X, Z]
                result.append(['shorter', fact1[2], fact2[2]])
    return result


def derive_taller(facts):
    """
    Derive ['taller', Y, Z] from the given facts.
    """
    result = []
    for fact1 in facts:
        for fact2 in facts:
            if (fact1[0] == 'shorter') and (fact2[0] == 'taller') and (fact1[1] == fact2[1]) and (fact1[1] == fact2[1]) and (len(fact1)==3) and (len(fact2)==3):
                # Match: ['shorter', X, Y] and ['taller', X, Z]
                result.append(['taller', fact1[2], fact2[2]])
    return result


def not2fact(fact):
    """
    Convert ['not_taller', X, Y] into ['shorter', X, Y].
    """
    if fact[0] == 'not_taller' and (len(fact)==3):
        return ['taller', fact[2], fact[1]]
    elif fact[0] == 'not_shorter'and (len(fact)==3):
        return ['shorter', fact[2], fact[1]]
    else:
        return []


def extend_facts1(facts):


    new_facts = []
    for fact in facts:
        new_fact = not2fact(fact)
        if new_fact != []:
            new_facts.append(new_fact)
    
    facts.extend(new_facts)

    return facts


def extend_facts2(facts):


    new_facts = []
    for fact in facts:
        new_fact = get_symmetrics(fact)
        if new_fact != []:
            new_facts.append(new_fact)
    
    facts.extend(new_facts)

    return facts
