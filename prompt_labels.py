
def amazon_pattern(label, pattern_id=0):
    if pattern_id==0:
        return label
    elif pattern_id==1:
        return f"All in all, the product is {label}."

def ag_news_pattern(label, pattern_id=0):
    if pattern_id==0:
        return label
    elif pattern_id==1:
        return f"This news is about {label}."

def xnli_pattern(label, pattern_id=0):
    if pattern_id==0:
        return label
    elif pattern_id==1:
        return f"The relationship between the two sentences is {label}."

def pawsx_pattern(label, pattern_id=0):
    if pattern_id==0:
        return label
    elif pattern_id==1:
        return f"The two sentences are {label} to each other."
    
def yahoo_pattern(label, pattern_id=0):
    if pattern_id==0:
        return label
    elif pattern_id==1:
        return f"The topic of the question and answer is {label}."

ID2LABELS = {
    'amazon_polarity': {0: 'bad',
                        1: 'great'},
    'amazon_star': {0: 'terrible',
                    1: 'bad',
                    2: 'ok',
                    3: 'good',
                    4: 'great'},
    'ag_news': {0: "world",
                1: "sports",
                2: "business",
                3: "tech"},
    'xnli': {0: 'entailment',
             1: 'neutral',
             2: 'contradiction'},
    'pawsx': {0: 'non-paraphrase',
              1: 'paraphrase'},
    'yahoo': {0: 'society',
        1: 'science',
        2: 'health',
        3: 'education',
        4: 'computer',
        5: 'sports',
        6: 'business',
        7: 'entertainment',      
        8: 'family',
        9: 'politics',} 
}

PATTERNS = {
    'amazon_polarity': amazon_pattern,
    'amazon_star': amazon_pattern,
    'ag_news': ag_news_pattern,
    'xnli': xnli_pattern,
    'pawsx': pawsx_pattern,
    'yahoo': yahoo_pattern
}

