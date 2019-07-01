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

    def __init__(self, feats_dir, questions_path, questions_emb_path,
                 answers_mapper, dataset, type_, spatial_feats=False):
        self.questions_path = questions_path
        self.answers_mapper = answers_mapper
        self.q2img_mapper = self.load_mapper('image_index')
        self.q2answer_mapper = self.load_mapper('answer')
        self.q_emb = np.load(questions_emb_path)
        self.feats_dir = feats_dir
        self.spatial_feats = spatial_feats
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
        feats = np.load(os.path.join(self.feats_dir, img_filename), allow_pickle=True).item()
        img = feats['features']
        if self.spatial_feats:
            spatial_feats = self.get_spatial_feats(feats['boxes'])
            img = np.concatenate([img, spatial_feats], axis=1)
        return torch.from_numpy(img).float(), torch.from_numpy(question).float(), label
    
    def get_spatial_feats(self, boxes):
        spatial_feats = []
        for box in boxes:
            height_step = (box[2] - box[0]) / 16
            width_step = (box[3] - box[1]) / 16

            width_points = [box[1] + (i * width_step) + (width_step / 2) for i in range(16)]
            height_points = [box[0] + (i * height_step) + (height_step / 2) for i in range(16)]

            spatial_feats.append(np.concatenate([[x, y] for x in height_points for y in width_points]))
        return np.vstack(spatial_feats)