import h5py
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


class ClevrBottomFeaturesDataset(Dataset):

    def __init__(self, feats_path, questions_path, questions_emb_path,
                 answers_mapper, img2idx_mapper_path, spatial_feats=False):
        self.questions_path = questions_path
        self.answers_mapper = answers_mapper
        self.q2img_mapper = self.load_mapper('image_index')
        self.q2answer_mapper = self.load_mapper('answer')
        with open(img2idx_mapper_path, 'r') as f:
            self.img2idx_mapper = json.load(f)['image_id_to_ix']
        self.q_emb = np.load(questions_emb_path)
        self.feats = h5py.File(feats_path, 'r')
        self.spatial_feats = spatial_feats
    
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
        label = self.answers_mapper[self.q2answer_mapper[idx]]
        image_id = str(self.q2img_mapper[idx])
        feats = self.feats['image_features'][self.img2idx_mapper[image_id]]
        if self.spatial_feats:
            boxes = self.feats['spatial_features'][self.img2idx_mapper[image_id]]
            spatial_feats = self.get_spatial_feats(boxes[:,:4])
            feats = np.concatenate([feats, spatial_feats], axis=1)
        return torch.from_numpy(feats).float(), torch.from_numpy(question).float(), label
    
    def get_spatial_feats(self, boxes):
        spatial_feats = []
        for box in boxes:
            height_step = (box[2] - box[0]) / 16
            width_step = (box[3] - box[1]) / 16

            width_points = [box[1] + (i * width_step) + (width_step / 2) for i in range(16)]
            height_points = [box[0] + (i * height_step) + (height_step / 2) for i in range(16)]

            spatial_feats.append(np.concatenate([[x, y] for x in height_points for y in width_points]))
        return np.vstack(spatial_feats)



def get_gqa_answers_mapper(questions_paths):
    '''
    questions_path (list): paths to the training and validations jsons of the filtered questions.
    '''
    answers = set()
    for question_path in questions_paths:
        with open(question_path, 'r') as f:
            data = json.load(f)
        for question in data:
            answers.add(question['answer'])
    answer_mapper = {}
    for i, answer in enumerate(answers):
        answer_mapper[answer] = i
    return answer_mapper


class GqaBottomFeaturesDataset:
    
    def __init__(self, answers_mapper, questions_path, feats_dir, q_embs_dir=None, q_embs_path=None,
                 spatial_feats=True):
        '''
        answers_mapper: mapper from answer to int (label), returned grom get_gqa_answers_mapper.
        questions_path: path to the filtered question's json.
        feats_dir: directory of the bottom-up features.
        q_embs_dir: directory of the questions if the questions are separated in single files.
        q_embs_path: path to the questions embeddings if there is a single file with all the questions.
        '''
        self.feats_dir = feats_dir
        self.q_embs = None
        if q_embs_path:
            self.q_embs = np.load(q_embs_path)
        self.q_embs_dir = q_embs_dir
        self.questions_path = questions_path
        self.answers_mapper = answers_mapper
        self.qs_info_mapper = self.get_qs_info_mapper()
        self.spatial_feats = spatial_feats
    
    def get_qs_info_mapper(self):
        mapper = []
        with open(self.questions_path, 'r') as f:
            data = json.load(f)
        for question in data:
            mapper.append({
                'index': question['index'],
                'image_index': question['image_index'],
                'answer': self.answers_mapper[question['answer']],
            })
        return sorted(mapper, key=lambda x: x['index'])
    
    def __getitem__(self, idx):
        q_emb = self.get_q_emb(idx)
        q_info = self.qs_info_mapper[idx]
        feats_filename = f'{q_info["image_index"]}.npy'
        feats = np.load(os.path.join(self.feats_dir, feats_filename), allow_pickle=True).item()
        img = feats['features']
        label = q_info['answer'] # Already mapped to int in get_qs_info_mapper
        if self.spatial_feats:
            spatial_feats = self.get_spatial_feats(feats['boxes'])
            img = np.concatenate([img, spatial_feats], axis=1)
        return torch.from_numpy(img).float(), torch.from_numpy(q_emb).float(), label
    
    def get_q_emb(self, idx):
        '''
        Validation questions should be loaded from a file in memory and
        training questions should be loaded from a file in the specified
        directory.
        '''
        if self.q_embs_dir:
            filename = f'{str(idx).zfill(7)}.npy'
            file_path = os.path.join(self.q_embs_dir, filename)
            return np.load(file_path)
        elif self.q_embs:
            return self.q_embs[idx]
    
    def get_spatial_feats(self, boxes):
        spatial_feats = []
        for box in boxes:
            height_step = (box[2] - box[0]) / 16
            width_step = (box[3] - box[1]) / 16

            width_points = [box[1] + (i * width_step) + (width_step / 2) for i in range(16)]
            height_points = [box[0] + (i * height_step) + (height_step / 2) for i in range(16)]

            spatial_feats.append(np.concatenate([[x, y] for x in height_points for y in width_points]))
        return np.vstack(spatial_feats)
