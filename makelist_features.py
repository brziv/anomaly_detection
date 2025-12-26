"""Generate legacy txt lists from train/val/test jsonl

 - train/val: write lines "{feature_path} {label}\n"
 - test: write lines "{video_path} {label} {fps}\n"
"""
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Make txt lists from jsonl splits')
    parser.add_argument('--train_json', default='train.jsonl')
    parser.add_argument('--val_json', default='val.jsonl')
    parser.add_argument('--test_json', default='test.jsonl')
    parser.add_argument('--x3d_train', default='x3d_train.txt')
    parser.add_argument('--x3d_val', default='x3d_val.txt')
    parser.add_argument('--x3d_test', default='x3d_test.txt')
    return parser.parse_args()


def write_feature_list(json_path, out_path):
    normal_files = []
    with open(out_path, 'w') as f:
        for line in open(json_path):
            item = json.loads(line)
            feature_path = item.get('feature_path')
            label = item.get('label')
            filename = None
            if label == 'normal':
                normal_files.append((feature_path, label))
            else:
                if feature_path is None:
                    # skip or warn
                    print(f"Warning: missing feature_path for anomaly in {json_path}: {item}")
                else:
                    f.write(f"{feature_path} {label}\n")
        for feature_path, label in normal_files:
            if feature_path is None:
                print(f"Warning: missing feature_path for normal in {json_path}")
            else:
                f.write(f"{feature_path} {label}\n")


def write_test_list(json_path, out_path):
    with open(out_path, 'w') as f:
        for line in open(json_path):
            item = json.loads(line)
            video_path = item.get('video_path')
            label = item.get('label')
            fps = item.get('fps')
            f.write(f"{video_path} {label} {fps}\n")


def main():
    args = parse_args()
    write_feature_list(args.train_json, args.x3d_train)
    write_feature_list(args.val_json, args.x3d_val)
    write_test_list(args.test_json, args.x3d_test)
    print(f"Wrote {args.x3d_train}, {args.x3d_val}, {args.x3d_test}")


if __name__ == '__main__':
    main()
