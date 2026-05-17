import numpy as np

def force_incoherence_to_null(df, rules=None):
    
    '''Apply rules to set incoherent values to null.'''
    if rules is None:
        rules = {
            "DONOR_AGE": lambda s: (s < 0) | (s > 100),
            "CHILDREN": lambda s: s < 0,
            "TARGET_B": lambda s: ~s.isin([0, 1]),
            "RECENT_RESPONSE_PROP": lambda s: (s < 0) | (s > 1),
            "RECENT_CARD_RESPONSE_PROP": lambda s: (s < 0) | (s > 1),
            "RECENT_AVG_CARD_GIFT_AMT": lambda s: s < 0,
            "MEDIAN_HOUSEHOLD_INCOME": lambda s: s < 0,
            "PER_CAPITA_INCOME": lambda s: s < 0,
        }

    for col, condition in rules.items():
        if col not in df.columns:
            continue
        mask = condition(df[col])
        mask = mask.fillna(False)
        df.loc[mask, col] = np.nan
    return df
