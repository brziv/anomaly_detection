import json
import numpy as np
import random
import pickle

import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, args=None, test_mode=False, mode='train', list_file=None):
        """Dataset that reads JSONL lists. For train/val `list_file` lines must include `feature_path`.
        For test, `list_file` should have JSON lines with `video_path`, `label`, `fps` (no feature_path required).

        Parameters:
            args: optional args object providing defaults (features_train_list, features_val_list, features_test_list)
            test_mode: legacy boolean (keeps old API)
            mode: one of 'train', 'val', 'test'
            list_file: override the file to read
        """
        # Determine which file to read
        if list_file is not None:
            self.list_file = list_file
        else:
            if args is None:
                raise ValueError('Provide args or list_file')
            if test_mode or mode == 'test':
                self.list_file = getattr(args, 'features_test_list', getattr(args, 'test_rgb_list', 'test.jsonl'))
            elif mode == 'val':
                self.list_file = getattr(args, 'features_val_list', 'val.jsonl')
            else:
                self.list_file = getattr(args, 'features_train_list', getattr(args, 'rgb_list', 'train.jsonl'))

        self.test_mode = test_mode or (mode == 'test')

        # Read list
        self.items = []
        if self.list_file.endswith('.jsonl'):
            for line in open(self.list_file):
                try:
                    self.items.append(json.loads(line))
                except Exception:
                    continue
        else:
            # legacy txt: feature_path label or video_path label fps for test
            for line in open(self.list_file):
                parts = line.strip().split()
                if len(parts) == 2:
                    self.items.append({'feature_path': parts[0], 'label': parts[1]})
                elif len(parts) >= 3:
                    # test legacy
                    self.items.append({'video_path': parts[0], 'label': parts[1], 'fps': float(parts[2])})

        # Build convenience lists
        if not self.test_mode:
            self.paths = [it.get('feature_path') for it in self.items]
            self.labels = [0.0 if it.get('label') == 'normal' else 1.0 for it in self.items]
            self.n_len = sum(1 for l in self.labels if l == 0.0)
            self.a_len = sum(1 for l in self.labels if l == 1.0)
        else:
            self.video_paths = [it.get('video_path') for it in self.items]
            self.labels = [0.0 if it.get('label') == 'normal' else 1.0 for it in self.items]
            self.fps = [it.get('fps') for it in self.items]
            # collect feature paths if present; may be None for some test entries
            self.paths = [it.get('feature_path') for it in self.items]
            # If some feature_path are missing, try to build a lookup from a master features.jsonl
            if any(p is None for p in self.paths):
                features_json = getattr(args, 'features_json', 'features.jsonl') if args is not None else 'features.jsonl'
                try:
                    mapping = {}
                    with open(features_json, 'r') as fin:
                        for line in fin:
                            try:
                                j = json.loads(line)
                                mapping[j.get('video_path')] = j.get('feature_path')
                            except Exception:
                                continue
                    for i, p in enumerate(self.paths):
                        if p is None:
                            vp = self.video_paths[i]
                            self.paths[i] = mapping.get(vp)
                except Exception:
                    # unable to build mapping; leave paths as-is (may be None)
                    pass

    def load_feature(self, path):
        if path is None:
            return None
        # Load file (npy/npz or pickle)
        if path.endswith('.npy') or path.endswith('.npz'):
            feat = np.load(path, allow_pickle=True)
        else:
            # assume pickle
            try:
                with open(path, 'rb') as f:
                    feat = pickle.load(f)
            except Exception:
                feat = np.load(path, allow_pickle=True)

        # Normalize common container types returned from feature files
        # If it's a dict, try common keys
        if isinstance(feat, dict):
            for k in ('embedding', 'features', 'feature', 'feat', 'data'):
                if k in feat:
                    feat = feat[k]
                    break
            else:
                # fallback: pick the first array-like value
                for v in feat.values():
                    if hasattr(v, 'shape') or isinstance(v, (list, tuple, np.ndarray)):
                        feat = v
                        break

        # If it's a list of dicts with 'embedding' or 'features', collect and concatenate
        if isinstance(feat, list):
            if all(isinstance(x, dict) and ('embedding' in x or 'features' in x) for x in feat):
                arrs = []
                for x in feat:
                    if 'embedding' in x:
                        arrs.append(np.asarray(x['embedding']))
                    else:
                        arrs.append(np.asarray(x.get('features')))
                try:
                    feat = np.concatenate(arrs, axis=0)
                except Exception:
                    feat = np.stack(arrs, axis=0)
            else:
                feat = np.asarray(feat)

        feat = np.asarray(feat, dtype=np.float32)
        return feat

    def __getitem__(self, index):
        if not self.test_mode:
            if index == 0:
                self.n_ind = [i for i, l in enumerate(self.labels) if l == 0.0]
                self.a_ind = [i for i, l in enumerate(self.labels) if l == 1.0]
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)

            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()

            nfeatures = self.load_feature(self.paths[nindex])
            nlabel = self.labels[nindex]

            afeatures = self.load_feature(self.paths[aindex])
            alabel = self.labels[aindex]

            return nfeatures, nlabel, afeatures, alabel
        else:
            # test mode: prefer returning loaded features if available (so test pipeline can call .to(device)).
            feature_path = self.paths[index] if hasattr(self, 'paths') else None
            if feature_path:
                feat = self.load_feature(feature_path)
                label = self.labels[index]
                return feat, label
            else:
                # fallback: return video_path, label, fps so external code can handle loading
                return self.video_paths[index], self.labels[index], self.fps[index]

    def __len__(self):
        if self.test_mode:
            return len(self.items)
        else:
            return min(self.a_len, self.n_len)
