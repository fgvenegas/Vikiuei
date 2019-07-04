import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Ramen
from utils import ClevrBottomFeaturesDataset, get_answers_mapper, GqaBottomFeaturesDataset, get_gqa_answers_mapper


def main():

    parser = argparse.ArgumentParser()

    # Training options.
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--n_regions', default=36, type=int)
    parser.add_argument('--spatial_feats', default=0, type=int, help='Add spatial features to bottom-up features.')
    parser.add_argument('--print_every', default=36, type=int, help='Print every X mini batches.')

    # Datasets
    parser.add_argument('--dataset', default='miniCLEVR', type=str, choices=['miniCLEVR', 'miniGQA'])

    ## Validation
    parser.add_argument('--val_questions_path', default='./', type=str, help='Path to the original json file of the validation questions.')
    parser.add_argument('--val_feats_dir', default='./', type=str, help='Directory of the bottom-up features of the validation images.')
    parser.add_argument('--val_question_embeddings_path', default='./', type=str, help='Path to the glove embeddings of the validation questions')
    parser.add_argument('--val_feats_path', default='./', type=str, help='Path of the bottom-up features of the validation images.')
    parser.add_argument('--val_img2index_path', default='./', type=str)

    ## Training
    parser.add_argument('--train_questions_path', default='./', type=str, help='Path to the original json file of the training questions.')
    parser.add_argument('--train_feats_dir', default='./', type=str, help='Directory of the bottom-up features of the training images.')
    parser.add_argument('--train_question_embeddings_path', default='./', type=str, help='Path to the glove embeddings of the training questions')
    parser.add_argument('--train_question_embeddings_dir', default='./', type=str, help='Directory of the glove embeddings of the training questions')
    parser.add_argument('--train_img2index_path', default='./', type=str)
    parser.add_argument('--train_feats_path', default='./', type=str, help='Path of the bottom-up features of the validation images.')

    # Checkpoint
    parser.add_argument('--checkpoints_dir', default='./checkpoints', type=str, help='Directory for checkpoints.')
    parser.add_argument('--experiment_name', default='experiment', type=str, help='Name of the experiment. Used in the checkpoints dir.')

    # Debug
    parser.add_argument('--debug', default=0, type=int, help='Print debug messages.')

    args = parser.parse_args()

    # Get GPU support if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Make checkpoints directory.
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    # Path to the best model checkpoint.
    checkpoint_path = os.path.join(args.checkpoints_dir, f'{args.experiment_name}.tar')

    if args.dataset == 'miniCLEVR':
        print(args)

        # (FEATS_PATH, QUESTIONS_VAL_PATH, QUESTIONS_EMBEDDING_VAL,
        #                       answers_mapper, IMG2IDX_MAPPER_PATH, spatial_feats=True)

        # Get all possible answers and map them with an integer.
        answers_mapper = get_answers_mapper([args.val_questions_path, args.train_questions_path])

        train_dataset = ClevrBottomFeaturesDataset(args.train_feats_path,
                                                args.train_questions_path,
                                                args.train_question_embeddings_path,
                                                answers_mapper,
                                                args.train_img2index_path,
                                                spatial_feats=args.spatial_feats)
        
        val_dataset = ClevrBottomFeaturesDataset(args.val_feats_path,
                                                args.val_questions_path,
                                                args.val_question_embeddings_path,
                                                answers_mapper,
                                                args.val_img2index_path,
                                                spatial_feats=args.spatial_feats)
    elif args.dataset == 'miniGQA':

        # Get all possible answers and map them with an integer.
        answers_mapper = get_gqa_answers_mapper([args.val_questions_path, args.train_questions_path])

        train_dataset = GqaBottomFeaturesDataset(answers_mapper,
                                                args.train_questions_path,
                                                args.train_feats_dir,   
                                                q_embs_dir=args.train_question_embeddings_dir,
                                                spatial_feats=args.spatial_feats)
        
        val_dataset = GqaBottomFeaturesDataset(answers_mapper,
                                                args.train_questions_path,
                                                args.train_feats_dir,   
                                                q_embs_path=args.val_question_embeddings_path,
                                                spatial_feats=args.spatial_feats)
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                             drop_last=True)
    
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                           drop_last=True)
    
    n_classes = len(answers_mapper)

    print(answers_mapper)

    model = Ramen(k=args.n_regions, batch_size=args.batch_size, n_classes=n_classes,
                  spatial_feats=args.spatial_feats)
    model.to(device)
    

    criterion = nn.CrossEntropyLoss()

    lr_default = 1e-3
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10, 20, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]

    optimizer = optim.Adamax(model.parameters(), lr=lr_default, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    best_val_acc = 0

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        if epoch < len(gradual_warmup_steps):
            optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            if args.debug:
                print('gradual warmup lr: %.4f' % optimizer.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optimizer.param_groups[0]['lr'] *= lr_decay_rate
            if args.debug:
                print('decreased lr: %.4f' % optimizer.param_groups[0]['lr'])
        else:
            if args.debug:
                print('lr: %.4f' % optimizer.param_groups[0]['lr'])

        current_loss = 0.0
        tqdm_trainloader = tqdm(trainloader)

        train_batches = 0

        for data in tqdm_trainloader:
            imgs, questions, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            questions = questions.permute(1, 0, 2)
            # print(imgs.size(), questions.size(), labels.size())
    
            optimizer.zero_grad()
            outputs = model(imgs, questions)
            # print(outputs.size())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_batches += 1
            current_loss += loss.item()
            
            tqdm_trainloader.set_description('Epoch {}/{}'.format(epoch + 1, args.epochs))
            tqdm_trainloader.set_postfix(loss=current_loss / train_batches)
            
            if args.debug:
                if train_batches == 3:
                    break
    
        correct = 0
        total = 0
        tmp_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            tqdm_valloader = tqdm(valloader)
            for data in tqdm_valloader:
                imgs, questions, labels = data[0].to(device), data[1].to(device), data[2].to(device)
                questions = questions.permute(1, 0, 2)
                outputs = model(imgs, questions)
                _, predicted = torch.max(outputs.data, 1)
                tmp_val_loss += criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_batches += 1

                tqdm_valloader.set_description('Validation')
                tqdm_valloader.set_postfix(loss=tmp_val_loss / val_batches)

                if args.debug:
                    if val_batches == 3:
                        break
        
        val_loss = tmp_val_loss / val_batches
        val_acc = 100 * correct / total

        if args.debug:
            print('val_loss {:.4f}'.format(val_loss))
            print('prev_acc {:.2f} | current_acc {:.2f}'.format(best_val_acc, val_acc))

        if val_acc > best_val_acc:
            
            if args.debug:
                print('Saving..')

            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'answers_mapper': answers_mapper,
            }, checkpoint_path)

        print('Epoch {:d} | val_acc {:.2f}% | val_loss {:.4f}'.format(epoch + 1, val_acc, val_loss))

if __name__ == '__main__':
    main()