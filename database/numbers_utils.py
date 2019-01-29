import numpy as np
import pandas as pd

def parse_num(text):
    words = text.split()
    first_word = words[0]
    second_word = ''
    if len(words) > 1:
        second_word = words[1]
    try:
        num = float(first_word.strip())
        if second_word.strip().strip('.') == 'bis':
            num += 0.5
        return num
    except:
        return np.nan

def complete_seq(seq):
    if len(seq) == 0:
        return seq
    final_seq = seq.copy()
    valid_seqs = get_valid_seqs(seq[:,1])
    if valid_seqs.size == 0:
        return final_seq
    padded_valid_seqs = np.array([[]] + valid_seqs.tolist() + [[]])
    for seq_l, seq_r in zip(padded_valid_seqs[:-1], padded_valid_seqs[1:]):
        
        if len(seq_r) > 0:
            cut_r = seq_r[0]
            th_r = get_theoritical_seq(seq, seq_r)
        else:
            cut_r = len(seq)
            th_r = get_theoritical_seq(seq, seq_l)
            
        if len(seq_l) > 0:
            cut_l = seq_l[1]+1
            th_l = get_theoritical_seq(seq, seq_l)
        else:
            cut_l = 0
            th_l = get_theoritical_seq(seq, seq_r)
            cut_l = len(get_theoritical_seq(seq, seq_r)) - len(th_l[th_l > 0])

        th_l = th_l[cut_l:cut_r]
        th_r = th_r[cut_l:cut_r]
        if (th_l == th_r).all() and len(th_l) > 0:
            final_seq[cut_l:cut_r] = th_l
    return final_seq

def get_theoritical_seq(seq, valid_seq):
    if len(valid_seq) == 0:
        return np.array([])
    start_offset = valid_seq[0]
    start_val = seq[valid_seq[0]][1]-start_offset
    end_offset = len(seq) - valid_seq[1]
    end_val = seq[valid_seq[1]][1] + end_offset
    
    return np.hstack([seq[:,0].reshape(-1,1), np.arange(start_val, end_val).reshape(-1,1)])

def get_missing_idx(seq):
    idxs = np.arange(0, len(seq))
    prev_max_idx = -1
    res = []
    for seq in get_valid_seqs(seq):
        idx_sel = idxs[(idxs > prev_max_idx) & (idxs < seq[0])]
        prev_max_idx = seq[1]
        if len(idx_sel) > 0:
            res.append(np.arange(idx_sel[0], idx_sel[-1]+1))
    return res

def get_valid_mask(seq, seq_size=4):
    mask = np.array([False] * len(seq))
    valid_seqs = get_valid_seqs(seq, seq_size=4)
    for val_seq in valid_seqs:
        mask[np.arange(val_seq[0], val_seq[1]+1)] = True
    return mask

def get_valid_seqs(seq, seq_size=4):
    candidates = get_candidates(seq)
    if candidates.size == 0:
        return np.array([])
    candidate_validity = validate_candidates(candidates, seq_size)
    return np.array([(c[0][0], c[-1][0]) for c in candidates[candidate_validity]]).astype(int)

def get_candidates(seq):
    idx_seq = np.array(list(enumerate(seq)))
    seq_val = idx_seq[:,1]
    one_diff = (seq_val[1:]-seq_val[:-1]) == 1
    idxs = np.arange(0, len(seq))
    split_idxs = sorted((np.where(one_diff[1:]& ~one_diff[:-1])[0]+1).tolist() + (np.where(~one_diff[1:] & one_diff[:-1])[0]+2).tolist())
    is_seq = np.split(one_diff, np.where(one_diff[1:]^one_diff[:-1])[0]+1)
    return np.array(np.split(idx_seq, split_idxs))[list(map(np.all, is_seq))]

def validate_candidates(candidates, seq_size=4):
    prev_max = -1
    results = []
    for candidate in candidates:
        prev_max, valid = is_valid(candidate[:,1], prev_max, seq_size)
        results.append(valid)
    return np.array(results)

def is_valid(candidate, prev_max, seq_size = 4):
    valid = len(candidate) >= seq_size and (candidate > prev_max).all()
    if valid:
        return candidate.max(), True
    else:
        return prev_max, False

def expand_nums(nums, texts):
    num_ranges = texts.str.extract('^([0-9]+)(?: à ([0-9]+))?')
    num_ranges = num_ranges.combine_first(texts.str.extract('^([0-9]+)(?: ?\- ?([0-9]+))?'))
    offset = 0
    for idx, start, end in num_ranges[num_ranges[1].notnull()].reset_index().values.astype(int):
        if start >= end or end-start > 100 or start > 1300:
            continue
        range_ = np.arange(start+1, end+1)
        nums = np.insert(nums, idx+1+offset, np.array(list(zip([idx] * len(range_), range_))), axis=0)
        offset += len(range_)
    return nums

def complete_document(document):
    seq_size= 4
    document = document.reset_index()

    nums = document['num'].reset_index().values
    texts = document['text']
    try:
        nums = expand_nums(nums, texts)
    except:
        print(document['doc'].iloc[0])
        das

    if len(nums) >= seq_size:
        try:
            completed_nums = complete_seq(nums)
        except:
            print(document['doc'])
            das
    else:
        completed_nums = nums

    values = np.stack([completed_nums[:,0], completed_nums[:,1], [np.nan] * len(completed_nums)]).T
    df_completed_nums = pd.DataFrame(completed_nums, columns=['idx', 'num'])
    df_completed_nums['idx'] = df_completed_nums['idx'].astype(int)
    df_completed_nums.set_index('idx', inplace=True)
    expanded_filter = df_completed_nums.groupby('idx').first()['num'] != df_completed_nums.groupby('idx').last()['num']
    expanded_values = df_completed_nums.groupby('idx').last()['num'][expanded_filter].values
    values = values[~df_completed_nums.index.duplicated(keep='first')]
    values[:,2][np.where(expanded_filter)[0]] = expanded_values
    df_completed_nums = pd.DataFrame(values, columns=['idx', 'num', 'last_num_completed'])
    df_completed_nums['idx'] = df_completed_nums['idx'].astype(int)
    df_completed_nums.set_index('idx', inplace=True)

    return document.join(df_completed_nums, rsuffix='_completed').set_index(['doc', 'page', 'entity'])
