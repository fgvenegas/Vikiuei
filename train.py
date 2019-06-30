import argparse
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

    args = parser.parse_args()

    # Get all possible answers and map them with an integer.
    answers_mapper = get_answers_mapper([args.val_questions_path, args.train_questions_path])

    n_classes = len(answers_mapper)

    model = Ramen(k=args.n_regions, batch_size=args.batch_size, n_classes=n_classes, spatial_feats=False)

    train_dataset = BottomFeaturesDataset(args.train_feats_dir,
                                          args.train_questions_path,
                                          args.train_question_embeddings_path,
                                          answers_mapper,
                                          args.dataset,
                                          'train')
    
    val_dataset = BottomFeaturesDataset(args.val_feats_dir,
                                        args.val_questions_path,
                                        args.val_question_embeddings_path,
                                        answers_mapper,
                                        args.dataset,
                                        'val')
    
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    

    criterion = nn.CrossEntropyLoss()
    '''
    First 4 epochs: 2.5 * epoch * 10-4
    Epoch 5 to 10: 5 * 10-4
    Decay 0.25 for every two epochs
    '''
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            imgs, questions, labels = data

            questions = questions.permute(1, 0, 2)

            # print(imgs.size(), questions.size(), labels.size())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(imgs, questions)

            # print(outputs.size())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.print_every == (args.print_every - 1):    # print every 3 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / args.print_every))
                running_loss = 0.0
    
        correct = 0
        total = 0
        with torch.no_grad():
            for data in valloader:
                imgs, questions, labels = data
                questions = questions.permute(1, 0, 2)
                outputs = model(imgs, questions)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy on epoch {:d}: {:.2f} \%'.format(epoch + 1, 100 * correct / total))

if __name__ == '__main__':
    main()