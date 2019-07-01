import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Ramen
from utils import BottomFeaturesDataset, get_answers_mapper


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

    ## Training
    parser.add_argument('--train_questions_path', default='./', type=str, help='Path to the original json file of the training questions.')
    parser.add_argument('--train_feats_dir', default='./', type=str, help='Directory of the bottom-up features of the training images.')
    parser.add_argument('--train_question_embeddings_path', default='./', type=str, help='Path to the glove embeddings of the training questions')

    # Checkpoint
    parser.add_argument('--checkpoints_dir', default='./checkpoints', type=str, help='Directory for checkpoints.')
    parser.add_argument('--experiment_name', default='experiment', type=str, help='Name of the experiment. Used in the checkpoints dir.')

    # Debug
    parser.add_argument('--debug', default=0, type=int, help='Print debug messages.')

    args = parser.parse_args()

    # Get all possible answers and map them with an integer.
    answers_mapper = get_answers_mapper([args.val_questions_path, args.train_questions_path])

    # Make checkpoints directory.
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    # Path to the best model checkpoint.
    checkpoint_path = os.path.join(args.checkpoints_dir, f'{args.experiment_name}.tar')

    n_classes = len(answers_mapper)

    model = Ramen(k=args.n_regions, batch_size=args.batch_size, n_classes=n_classes,
                  spatial_feats=args.spatial_feats)

    train_dataset = BottomFeaturesDataset(args.train_feats_dir,
                                          args.train_questions_path,
                                          args.train_question_embeddings_path,
                                          answers_mapper,
                                          args.dataset,
                                          'train',
                                          spatial_feats=args.spatial_feats)
    
    val_dataset = BottomFeaturesDataset(args.val_feats_dir,
                                        args.val_questions_path,
                                        args.val_question_embeddings_path,
                                        answers_mapper,
                                        args.dataset,
                                        'val',
                                        spatial_feats=args.spatial_feats)
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    

    criterion = nn.CrossEntropyLoss()
    '''
    First 4 epochs: 2.5 * epoch * 10-4
    Epoch 5 to 10: 5 * 10-4
    Decay 0.25 for every two epochs
    '''
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    best_val_acc = 0

    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            imgs, questions, labels = data
            questions = questions.permute(1, 0, 2)
            # print(imgs.size(), questions.size(), labels.size())
    
            optimizer.zero_grad()
            outputs = model(imgs, questions)
            # print(outputs.size())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % args.print_every == (args.print_every - 1):
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / args.print_every))
                running_loss = 0.0

            if args.debug:
                if i == 3:
                    break
    
        correct = 0
        total = 0
        tmp_val_loss = 0
        val_batches = 0

        if args.debug:
            val_batches = 0

        with torch.no_grad():
            for data in valloader:
                imgs, questions, labels = data
                questions = questions.permute(1, 0, 2)
                outputs = model(imgs, questions)
                _, predicted = torch.max(outputs.data, 1)
                tmp_val_loss += criterion(outputs, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if args.debug:
                    val_batches += 1
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
            }, checkpoint_path)

        print('Epoch {:d} | val_acc {:.2f}% | val_loss {:.4f}'.format(epoch + 1, val_acc, val_loss))

if __name__ == '__main__':
    main()