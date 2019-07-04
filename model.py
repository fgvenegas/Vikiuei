import torch
import torch.nn as nn


class Ramen(nn.Module):
    def __init__(self, k, batch_size, n_classes, spatial_feats=True):
        super().__init__()
        self.k = k
        self.batch_size = batch_size
        self.q_emb = nn.GRU(input_size=300, hidden_size=512, bidirectional=True)
        if spatial_feats:
            self.bn = nn.BatchNorm1d(2560 + 1024)
            self.projection = Projector(2560 + 1024, 1024)
        else:
            self.bn = nn.BatchNorm1d(2048 + 1024)
            self.projection = Projector(2048 + 1024, 1024)
        self.agg = nn.GRU(input_size=2048, hidden_size=1024, bidirectional=True)
        self.fc_swish = nn.Linear(2048, 2048)
        self.fc_output = nn.Linear(2048, n_classes)

    def forward(self, imgs, q):
        # Take the concatenated features.
        out, last_hidden = self.q_emb(q)
        q_emb = out[-1, :, :]
        #print(f'out shape: {out.shape}')
        #print(f'out shape: {out[-1, :, :].shape}')
        #q_emb = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        #print(f'question embedding: {q_emb.shape}')
        c = self.early_fusion(imgs, q_emb)
        #print(f'early fusion: {c.shape}')
        c_reshaped = c.view(-1, c.shape[-1])
        #print(f'reshape of early fusion: {c_reshaped.shape}')
        c_reshaped = self.bn(c_reshaped)
        #print(f'Batch norm: {c_reshaped.shape}')
        b_reshaped = self.projection(c_reshaped)
        # print(f'After projection: {b_reshaped.shape}')
        b = b_reshaped.view(self.batch_size, self.k, -1)
        # print(f'Reshape of projection: {b.shape}')
        qb_emb = self.late_fusion(q_emb, b)
        # print(f'qb emb {qb_emb.shape}')
        _, last_hidden = self.agg(qb_emb.permute(1, 0, 2))
        agg_emb = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        # print(f'Agg bigru: {agg_emb.shape}')
        fc_swish = swish(self.fc_swish(agg_emb))
        # print(f'fc_swish shape: {fc_swish.shape}')
        fc_output = self.fc_output(fc_swish)
        # print(f'fc_output shape: {fc_output.shape}')
        output = torch.softmax(fc_output, 1)
        # print(f'Output: {output.shape}')
        return output

    def early_fusion(self, imgs, q_emb):
        q_emb = q_emb.view(q_emb.shape[0], 1, q_emb.shape[1]).repeat(1, imgs.shape[1], 1)
        # print(f'question expanding embedding: {q_emb.shape}')
        return torch.cat((imgs, q_emb), dim=2)
    
    def late_fusion(self, q_emb, b):
        q_emb = q_emb.view(q_emb.shape[0], 1, q_emb.shape[1]).repeat(1, b.shape[1], 1)
        return torch.cat((b, q_emb), dim=2)
        

class Projector(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.fc3 = nn.Linear(out_features, out_features)
        self.fc4 = nn.Linear(out_features, out_features)
    
    def forward(self, x):
        x = swish(self.fc1(x))
        residual = x
        x = swish(self.fc2(x)) + residual
        residual = x
        x = swish(self.fc3(x)) + residual
        residual = x
        output = swish(self.fc4(x)) + residual
        return output


def swish(x):
    return x * torch.sigmoid(x)



if __name__ == '__main__':

    N = 3
    k = 36
    n_classes = 1000
    D_img = (k, 2560)
    img_feats = torch.randn((N,) + D_img, dtype=torch.float)

    seq_length = 5
    word_size = 300
    q_feats = torch.randn((seq_length, N, word_size), dtype=torch.float)

    print(img_feats.shape)
    print(q_feats.shape)

    model = Ramen(k, N, n_classes)

    model.forward(img_feats, q_feats)