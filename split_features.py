import json
import random
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Create train/val/test splits from features.jsonl')
    parser.add_argument('--input', default='features.jsonl', help='input features jsonl file')
    parser.add_argument('--train_out', default='train.jsonl', help='output train jsonl')
    parser.add_argument('--val_out', default='val.jsonl', help='output val jsonl')
    parser.add_argument('--test_out', default='test.jsonl', help='output test jsonl')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='fraction for validation (default 0.1)')
    parser.add_argument('--test_fraction', type=float, default=0.1, help='fraction for test (default 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def _get_start(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)) and len(t) >= 1:
        return float(t[0])
    try:
        return float(t)
    except Exception:
        return None

def find_existing_normal_for_window(acc_item, normals_by_video, pair_window):
    acc_start = _get_start(acc_item.get('timestamp'))
    if acc_start is None or acc_start < pair_window:
        return None
    window_start = acc_start - pair_window
    v = acc_item.get('video_path')
    if v not in normals_by_video:
        return None
    candidates = normals_by_video[v]
    best_idx = None
    best_dist = None
    for i, n in enumerate(candidates):
        ns = _get_start(n.get('timestamp'))
        ne = None
        if isinstance(n.get('timestamp'), (list, tuple)) and len(n.get('timestamp')) >= 2:
            ne = float(n.get('timestamp')[1])
        elif ns is not None and n.get('duration'):
            ne = ns + float(n.get('duration'))
        else:
            continue
        # require the normal segment to end no later than acc_start and have at least pair_window overlap
        if ne <= acc_start and (ne - ns) >= pair_window:
            # prefer the one whose end is closest to acc_start
            dist = acc_start - ne
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
    if best_idx is not None:
        return candidates.pop(best_idx)
    return None


def main():
    args = parse_args()
    random.seed(args.seed)

    lines = [json.loads(l) for l in open(args.input)]

    accident_lines = [l for l in lines if l.get('label') != 'normal']
    normal_lines = [l for l in lines if l.get('label') == 'normal']

    # organize normals by video for quick lookup
    normals_by_video = {}
    for n in normal_lines:
        v = n.get('video_path')
        normals_by_video.setdefault(v, []).append(n)

    random.shuffle(accident_lines)
    for v in normals_by_video:
        random.shuffle(normals_by_video[v])

    generated_map = {}  # acc id -> normal item

    paired_acc = []
    unpaired_acc = []

    for acc in accident_lines:
        # try to find an existing normal in same video that can serve as hard-negative
        existing = find_existing_normal_for_window(acc, normals_by_video, 2.0)
        if existing:
            generated_map[id(acc)] = existing
            paired_acc.append(acc)
        else:
            unpaired_acc.append(acc)

    # Count totals
    total_acc = len(accident_lines)
    total_norm = len(normal_lines)

    train_frac = 1.0 - args.val_fraction - args.test_fraction

    # desired counts for accidents
    train_acc_target = int(total_acc * train_frac)
    val_acc_target = int(total_acc * args.val_fraction)
    test_acc_target = total_acc - train_acc_target - val_acc_target

    train_acc = []
    val_acc = []
    test_acc = []

    # allocate accidents preferring paired ones for train/val
    def allocate_acc(target, dest_list, prefer_from_paired=True):
        while len(dest_list) < target and (paired_acc or unpaired_acc):
            if prefer_from_paired and paired_acc:
                dest_list.append(paired_acc.pop())
            elif unpaired_acc:
                dest_list.append(unpaired_acc.pop())
            elif paired_acc:
                dest_list.append(paired_acc.pop())
            else:
                break

    allocate_acc(train_acc_target, train_acc, prefer_from_paired=True)
    allocate_acc(val_acc_target, val_acc, prefer_from_paired=True)
    # test prefer unpaired
    while len(test_acc) < test_acc_target and (unpaired_acc or paired_acc):
        if unpaired_acc:
            test_acc.append(unpaired_acc.pop())
        else:
            test_acc.append(paired_acc.pop())

    # remaining accidents -> train
    for acc in (paired_acc + unpaired_acc):
        train_acc.append(acc)

    # collect normals assigned via pairing
    train_norm = []
    val_norm = []
    test_norm = []

    for acc in train_acc:
        n = generated_map.get(id(acc))
        if n:
            train_norm.append(n)
            # if n originated from normals_by_video removal, it's already removed; else it's generated

    for acc in val_acc:
        n = generated_map.get(id(acc))
        if n:
            val_norm.append(n)

    for acc in test_acc:
        n = generated_map.get(id(acc))
        if n:
            test_norm.append(n)

    # remaining normals in normals_by_video form the pool
    normal_pool = []
    for lst in normals_by_video.values():
        normal_pool.extend(lst)

    random.shuffle(normal_pool)

    train_norm_target = int(total_norm * train_frac)
    val_norm_target = int(total_norm * args.val_fraction)
    test_norm_target = total_norm - train_norm_target - val_norm_target

    def fill_normals(target, dest):
        while len(dest) < target and normal_pool:
            dest.append(normal_pool.pop())

    fill_normals(train_norm_target, train_norm)
    fill_normals(val_norm_target, val_norm)
    fill_normals(test_norm_target, test_norm)

    # Compose final splits
    train = train_acc + train_norm
    val = val_acc + val_norm
    test_full = test_acc + test_norm

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test_full)

    # Write train and val as full JSON objects (keep feature_path and timestamp)
    with open(args.train_out, 'w') as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    with open(args.val_out, 'w') as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

    # For test, write only video_path, label, fps per user's request
    with open(args.test_out, 'w') as f:
        for item in test_full:
            out = {"video_path": item.get('video_path'), "label": item.get('label'), "fps": item.get('fps')}
            f.write(json.dumps(out) + "\n")

    # Print statistics
    def counts(lst):
        acc = sum(1 for l in lst if l.get('label') != 'normal')
        norm = sum(1 for l in lst if l.get('label') == 'normal')
        return acc, norm

    print(f"Train/Val/Test sizes: {len(train)}/{len(val)}/{len(test_full)}")
    print(f"Train acc/norm: {counts(train)}")
    print(f"Val acc/norm:   {counts(val)}")
    print(f"Test acc/norm:  {counts(test_full)}")


if __name__ == '__main__':
    main()
