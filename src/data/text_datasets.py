import random
import numpy as np
import json


AMAZON_CATEGORIES = ['beauty', 'fashion', 'games', 'toys', 'sports',
                     'office', 'pets', 'software', 'automotive', 'grocery']
_AMZ_CFG = {'beauty': 'All_Beauty', 'fashion': 'Amazon_Fashion', 'games': 'Video_Games',
            'toys': 'Toys_and_Games', 'sports': 'Sports_and_Outdoors', 'office': 'Office_Products',
            'pets': 'Pet_Supplies', 'software': 'Software', 'automotive': 'Automotive',
            'grocery': 'Grocery_and_Gourmet_Food'}
_AMZ_TASK = {'bin_15': [1.0, 5.0], 'bin_24': [2.0, 4.0], 'multi_5': [1.0, 2.0, 3.0, 4.0, 5.0]}


def load_text_dataset(name, task=None, data_root='./data', seed=42, max_per_class=8000, test_frac=0.2):
    if name == '20ng':
        return _load_20ng()
    if name == 'amazon':
        return _load_amazon(task or 'multi_5', data_root, seed, max_per_class, test_frac)
    raise ValueError(f"unknown text dataset '{name}'")

def _load_20ng():
    from sklearn.datasets import fetch_20newsgroups
    rm = ('headers', 'footers', 'quotes')
    tr = fetch_20newsgroups(subset='train', remove=rm)
    te = fetch_20newsgroups(subset='test', remove=rm)
    return (list(tr.data), list(map(int, tr.target))), (list(te.data), list(map(int, te.target))), list(tr.target_names)


def _load_20ng_canonical(test_frac=0.2, split_seed=0):
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset='all', remove=())
    texts, labels = list(d.data), np.asarray(d.target, dtype=int)
    perm = np.random.RandomState(split_seed).permutation(len(labels))
    n_test = int(len(labels) * test_frac)
    te_i, tr_i = perm[:n_test], perm[n_test:]
    return (([texts[i] for i in tr_i], labels[tr_i].tolist()),
            ([texts[i] for i in te_i], labels[te_i].tolist()), list(d.target_names))


def _load_amazon(task, data_root, seed, max_per_class, test_frac):
    ratings = _AMZ_TASK[task]
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    texts, labels = [], []
    per_cat_cap = max(1, max_per_class * len(ratings) // max(1, len(AMAZON_CATEGORIES)))
    keep = set(ratings)
    for cat in AMAZON_CATEGORIES:
        cat_name = _AMZ_CFG[cat]
        try:
            fp = hf_hub_download('McAuley-Lab/Amazon-Reviews-2023',
                                 f'raw/review_categories/{cat_name}.jsonl', repo_type='dataset')
        except Exception as e:
            print(f'[amazon] {cat_name}.jsonl unavailable ({type(e).__name__}: {e}) - skipping', flush=True)
            continue
        got = 0
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                if got >= per_cat_cap:
                    break
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                r = ex.get('rating')
                if r is None or float(r) not in keep:
                    continue
                r = float(r)
                t = ((ex.get('title') or '') + ' ' + (ex.get('text') or '')).strip()
                if len(t) < 5:
                    continue
                if task == 'bin_15':   y = 1 if r == 5.0 else 0
                elif task == 'bin_24': y = 1 if r == 4.0 else 0
                else:                  y = int(r) - 1
                texts.append(t); labels.append(y); got += 1
        print(f'[amazon] {cat_name}: {got} reviews kept', flush=True)

    texts, labels = np.array(texts, dtype=object), np.array(labels)
    n_classes = len(set(labels.tolist()))
    idx_keep = []
    for c in range(n_classes):
        ci = np.where(labels == c)[0]; rng.shuffle(list(ci))
        ci = np.random.RandomState(seed + c).permutation(ci)[:max_per_class]
        idx_keep.extend(ci.tolist())
    rng.shuffle(idx_keep)
    texts, labels = texts[idx_keep], labels[idx_keep]
    n_test = int(len(texts) * test_frac)
    te_txt, te_lab = list(texts[:n_test]), list(map(int, labels[:n_test]))
    tr_txt, tr_lab = list(texts[n_test:]), list(map(int, labels[n_test:]))
    class_names = ([f'r{int(r)}' for r in ratings] if task == 'multi_5'
                   else (['1', '5'] if task == 'bin_15' else ['2', '4']))
    print(f'[amazon:{task}] {len(tr_txt)} train / {len(te_txt)} test over {len(AMAZON_CATEGORIES)} categories, '
          f'{n_classes} classes', flush=True)
    return (tr_txt, tr_lab), (te_txt, te_lab), class_names
