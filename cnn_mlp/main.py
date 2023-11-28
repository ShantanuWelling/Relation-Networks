from __future__ import print_function
import os
import pickle
import random
import numpy as np
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from model import CNN_MLP


# HYPER-PARAMETERS
epochs = 20
batch_size = 64
seed = 42
print_freq = 10

# Setting seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Initialising summary writer for visualising
summary_writer = SummaryWriter()

# Initialising Model
model = CNN_MLP()

# Setting up the path variable
model_dirs = './model'

# Batch size variable
bs = batch_size

# Initialsing Empty tensors to load dataset
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 18)
label = torch.LongTensor(bs)

# Making code GPU compatible if available
if torch.cuda.is_available():
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

# Takes numpy dataset and converts it to torch tensors by filling the above intialised placeholders
def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.resize_(img.size()).copy_(img)
    input_qst.resize_(qst.size()).copy_(qst)
    label.resize_(ans.size()).copy_(ans)

# Extracting individual components of data(image, questins and answers) from the dataset
def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

# Main data loader function
def load_data():
    print('Loading Data')

    # Setting the path to dataset
    dirs = './data'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')

    # Opening the train and test dataset
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)

    # Place holder for various types of data present in the dataset
    ternary_train = []
    ternary_test = []
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('Processing Data')

    # Segregating different types of data from the train dataset
    for img, ternary, relations, norelations in train_datasets:
        img = np.swapaxes(img, 0, 2)

        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_train.append((img,qst,ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    # Segregating different types of data from the test dataset
    for img, ternary, relations, norelations in test_datasets:
        img = np.swapaxes(img, 0, 2)
        for qst, ans in zip(ternary[0], ternary[1]):
            ternary_test.append((img, qst, ans))
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))

    # Finally functtion returns the extracted train and test segregated data
    return (ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test)

# Called the above function to obtain the data
ternary_train, ternary_test, rel_train, rel_test, norel_train, norel_test = load_data()


# Train function used for training
def train(epoch, ternary, rel, norel):
    # Setting the model state as training
    model.train()

    # Shuffling the dataset to do uniform training across different runtime
    random.shuffle(ternary)
    random.shuffle(rel)
    random.shuffle(norel)

    # Separating multi dimensional data into various 1 dimensional data
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    # Array to store per batch accuracies across different types of data during training phase
    acc_ternary = []
    acc_rels = []
    acc_norels = []

    # Array to store per batch loss across different types of data during training phase
    l_ternary = []
    l_binary = []
    l_unary = []

    # Batchwise trainig is done here
    for batch_idx in range(len(rel[0]) // bs):
        # Convert numpy to tensor format and call train and then update loss and accuracy for ternary data type
        tensor_data(ternary, batch_idx)
        accuracy_ternary, loss_ternary = model.train_(input_img, input_qst, label)
        acc_ternary.append(accuracy_ternary.item())
        l_ternary.append(loss_ternary.item())

        # Convert numpy to tensor format and call train and then update loss and accuracy for relational data type
        tensor_data(rel, batch_idx)
        accuracy_rel, loss_binary = model.train_(input_img, input_qst, label)
        acc_rels.append(accuracy_rel.item())
        l_binary.append(loss_binary.item())

        # Convert numpy to tensor format and call train and then update loss and accuracy for non-relational data type
        tensor_data(norel, batch_idx)
        accuracy_norel, loss_unary = model.train_(input_img, input_qst, label)
        acc_norels.append(accuracy_norel.item())
        l_unary.append(loss_unary.item())

        # After every few batches(parameter set at the top of code) we print the train accuracy to keep the track of its graph
        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy_rel,
                   accuracy_norel))

    # After current epoch is finished (all batches are gone over once), we summarise the current epoch's accuracy into the summary writer
    avg_acc_ternary = sum(acc_ternary) / len(acc_ternary)
    avg_acc_binary = sum(acc_rels) / len(acc_rels)
    avg_acc_unary = sum(acc_norels) / len(acc_norels)

    summary_writer.add_scalars('Accuracy/train', {
        'ternary': avg_acc_ternary,
        'binary': avg_acc_binary,
        'unary': avg_acc_unary
    }, epoch)

    # After current epoch is finished (all batches are gone over once), we summarise the current epoch's loss into the summary writer
    avg_loss_ternary = sum(l_ternary) / len(l_ternary)
    avg_loss_binary = sum(l_binary) / len(l_binary)
    avg_loss_unary = sum(l_unary) / len(l_unary)

    summary_writer.add_scalars('Loss/train', {
        'ternary': avg_loss_ternary,
        'binary': avg_loss_binary,
        'unary': avg_loss_unary
    }, epoch)

    # return average accuracy for different types of training data
    return avg_acc_ternary, avg_acc_binary, avg_acc_unary

# Test function used for testing
def test(epoch, ternary, rel, norel):
    # Setting the model state as evaluating(or non-training phase)
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return

    # Separating multi dimensional data into various 1 dimensional data
    ternary = cvt_data_axis(ternary)
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    # Array to store per batch accuracies across different types of data during testing phase
    accuracy_ternary = []
    accuracy_rels = []
    accuracy_norels = []

    # Array to store per batch loss across different types of data during testing phase
    loss_ternary = []
    loss_binary = []
    loss_unary = []

    # Batchwise testing is done here
    for batch_idx in range(len(rel[0]) // bs):
        # Convert numpy to tensor format and call test and then update loss and accuracy for ternary data type
        tensor_data(ternary, batch_idx)
        acc_ter, l_ter = model.test_(input_img, input_qst, label)
        accuracy_ternary.append(acc_ter.item())
        loss_ternary.append(l_ter.item())

        # Convert numpy to tensor format and call test and then update loss and accuracy for relational data type
        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy_rels.append(acc_bin.item())
        loss_binary.append(l_bin.item())

        # Convert numpy to tensor format and call test and then update loss and accuracy for non-relational data type
        tensor_data(norel, batch_idx)
        acc_un, l_un = model.test_(input_img, input_qst, label)
        accuracy_norels.append(acc_un.item())
        loss_unary.append(l_un.item())

    # After current epoch is finished (all batches are gone over once), we summarise the current epoch's accuracy into the summary writer
    accuracy_ternary = sum(accuracy_ternary) / len(accuracy_ternary)
    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)

    # Printing the accuracies on the test data
    print('\n Test set: Binary accuracy: {:.0f}% | Unary accuracy: {:.0f}%\n'.format(
         accuracy_rel, accuracy_norel))

    # Pushing data to summary writer
    summary_writer.add_scalars('Accuracy/test', {
        'ternary': accuracy_ternary,
        'binary': accuracy_rel,
        'unary': accuracy_norel
    }, epoch)

    # After current epoch is finished (all batches are gone over once), we summarise the current epoch's loss into the summary writer
    loss_ternary = sum(loss_ternary) / len(loss_ternary)
    loss_binary = sum(loss_binary) / len(loss_binary)
    loss_unary = sum(loss_unary) / len(loss_unary)

    summary_writer.add_scalars('Loss/test', {
        'ternary': loss_ternary,
        'binary': loss_binary,
        'unary': loss_unary
    }, epoch)

    # return average accuracy for different types of testing data
    return accuracy_ternary, accuracy_rel, accuracy_norel




# We need to store model and hence we create a directory to do so, just checkig if it already exists or not
try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

# Final calling of train and test is done here
with open(f'./RN_{seed}_log.csv', 'w') as log_file:
    # Storing data into csv file to observe the graphs later
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc_ternary', 'train_acc_rel',
                     'train_acc_norel', 'train_acc_ternary', 'test_acc_rel', 'test_acc_norel'])

    print(f"Training RN model...")

    # Training occurs here and model is saved after it
    for epoch in range(1, epochs + 1):
        # Training function is called here
        train_acc_ternary, train_acc_binary, train_acc_unary = train(
            epoch, ternary_train, rel_train, norel_train)

        # Testing function is called here
        test_acc_ternary, test_acc_binary, test_acc_unary = test(
            epoch, ternary_test, rel_test, norel_test)

        # Writing accuracies in csv log file
        csv_writer.writerow([epoch, train_acc_ternary, train_acc_binary,
                         train_acc_unary, test_acc_ternary, test_acc_binary, test_acc_unary])

        # Saving the model in the directory opened above
        model.save_model(epoch)

