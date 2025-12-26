import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='STEAD')
    parser.add_argument('--rgb_list', default='x3d_train.txt', help='list of rgb features ')
    parser.add_argument('--test_rgb_list', default='x3d_val.txt', help='list of test rgb features ')

    # feature jsonl based lists and defaults
    parser.add_argument('--features_json', default='features.jsonl', help='input features jsonl')
    parser.add_argument('--features_train_list', default='train.jsonl', help='train jsonl list')
    parser.add_argument('--features_val_list', default='val.jsonl', help='val jsonl list')
    parser.add_argument('--features_test_list', default='test.jsonl', help='test jsonl list (video_path,label,fps)')
    parser.add_argument('--val_fraction', type=float, default=0.1, help='validation fraction when splitting features.jsonl')
    parser.add_argument('--test_fraction', type=float, default=0.1, help='test fraction when splitting features.jsonl')
    parser.add_argument('--use_val', action='store_true', help='enable validation loader usage')

    parser.add_argument('--comment', default='tiny', help='comment for the ckpt name of the training')

    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='attention dropout rate')
    parser.add_argument('--lr', type=str, default=2e-4, help='learning rates for steps default:2e-4')
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')


    parser.add_argument('--model_name', default='model', help='name to save model')
    parser.add_argument('--model_arch', default='fast', help='base or fast')
    parser.add_argument('--pretrained_ckpt', default= None, help='ckpt for pretrained model (for training)')
    parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train (default: 10)')
    parser.add_argument('--warmup', type=int, default=1, help='number of warmup epochs')



    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print(f"Warning: option.parse_args ignored unknown args: {unknown}")
    return args
