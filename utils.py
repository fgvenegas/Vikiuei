import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset


def get_answers_mapper(question_paths):
    answers = set()
    for question_path in question_paths:
        with open(question_path, 'r') as f:
            data = json.load(f)
        answers.update(data['answer'].values())
    answer_mapper = {}
    for i, answer in enumerate(answers):
        answer_mapper[answer] = i
    return answer_mapper


class BottomFeaturesDataset(Dataset):

    def __init__(self, feats_dir, questions_path, questions_emb_path, answers_mapper, dataset, type_):
        self.questions_path = questions_path
        self.answers_mapper = answers_mapper
        self.q2img_mapper = self.load_mapper('image_index')
        self.q2answer_mapper = self.load_mapper('answer')
        self.q_emb = np.load(questions_emb_path)
        self.feats_dir = feats_dir
        if dataset == 'miniCLEVR':
            self.get_img_name = lambda x: f'CLEVR_{type_}_{str(x).zfill(6)}.npy'
    
    def load_mapper(self, type_):
        with open(self.questions_path, 'r') as f:
            mapper = json.load(f)[type_]
        # Sort keys by number.
        q_idxs = sorted(map(int, mapper.keys()))
        new_mapper = [None] * len(q_idxs)
        for i, q_idx in enumerate(q_idxs):
            new_mapper[i] = mapper[str(q_idx)]
        return new_mapper
    
    def __len__(self):
        return len(self.q2img_mapper)

    def __getitem__(self, idx):
        question = self.q_emb[idx]
        img_filename = self.get_img_name(self.q2img_mapper[idx])
        label = self.answers_mapper[self.q2answer_mapper[idx]]
        img = np.load(os.path.join(self.feats_dir, img_filename), allow_pickle=True).item()['features']
        return torch.from_numpy(img), torch.from_numpy(question), label