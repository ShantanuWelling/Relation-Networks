from __future__ import print_function
import argparse
import os
import random
import numpy as np
import csv
import glob

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision

from model import RN

import json
from PIL import Image


# HYPER-PARAMETERS
epochs = 20
batch_size = 64
seed = 42
print_freq = 10


os.environ['CUDA_VISIBLE_DEVICES'] = '6'


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

summary_writer = SummaryWriter()

model_dirs = './model'
bs = batch_size

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    # ensure qst is of type int
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)


def train(epoch, rel):
    model.train() # set training mode

    random.shuffle(rel) # shuffle training set

    rel = cvt_data_axis(rel) # convert data to axis

    acc = [] #accuracy

    l = [] #loss

    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        accuracy, loss = model.train_(input_img, input_qst, label)
        acc.append(accuracy.item())
        l.append(loss.item())

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] '
                  'Train accuracy: {:.0f}%'.format(
                   epoch,
                   batch_idx * bs * 2,
                   len(rel[0]) * 2,
                   100. * batch_idx * bs / len(rel[0]),
                   accuracy))

    avg_acc = sum(acc) / len(acc)

    summary_writer.add_scalars('Accuracy/train', {
        'all': avg_acc
    }, epoch)

    avg_loss = sum(l) / len(l)

    summary_writer.add_scalars('Loss/train', {
        'all': avg_loss
    }, epoch)

    # return average accuracy
    return avg_acc

def validate(epoch, rel):
    model.eval() # set evaluation mode

    rel = cvt_data_axis(rel)

    accuracy = []

    loss = []

    for batch_idx in range(len(rel[0]) // bs):

        tensor_data(rel, batch_idx)
        acc_bin, l_bin = model.test_(input_img, input_qst, label)
        accuracy.append(acc_bin.item())
        loss.append(l_bin.item())

    accuracy = sum(accuracy) / len(accuracy)
    print('\n Val set accuracy: {:.0f}%\n'.format(
        accuracy))

    summary_writer.add_scalars('Accuracy/val', {
        'all': accuracy
    }, epoch)

    loss = sum(loss) / len(loss)

    summary_writer.add_scalars('Loss/val', {
        'all': loss
    }, epoch)

    return accuracy


def load_data():
    print('loading data...')
    dirs = '../CLEVR_v1.0'
    rel_train = []
    rel_val = []
    print('processing data...')
    train_questions = json.load(open(os.path.join(dirs, 'questions', 'CLEVR_train_questions.json')))
    val_questions = json.load(open(os.path.join(dirs, 'questions', 'CLEVR_val_questions.json')))

    # create embedding (word to index) and inverse embedding (index to word)
    word_to_int = {}
    int_to_word = {}
    train_words = set()
    max_question_length = 0
    for q in train_questions['questions']:
        # remove trailing question mark and convert to lowercase
        if q['question'][-1] == '?':
            q['question'] = q['question'][:-1]
        q['question'] = q['question'].lower()
        for w in q['question'].split():
            train_words.add(w)
        # add answer to vocabulary, convert to lowercase
        train_words.add(q['answer'].lower())
        # keep track of longest question
        max_question_length = max(max_question_length, len(q['question'].split()))
    word_to_int['UNK'] = 0 # add unknown word
    int_to_word[0] = 'UNK'
    word_to_int['?'] = 1 # add question mark, will use this as padding
    int_to_word[1] = '?'
    for i, w in enumerate(train_words):
        word_to_int[w] = i+2
        int_to_word[i+2] = w
    vocab_size = len(word_to_int)

    input_img = torch.FloatTensor(bs, 3, 128, 128) # image is 3 x 128 x 128 (channels x width x height)
    input_qst = torch.LongTensor(bs, max_question_length)
    label = torch.LongTensor(bs)
    model = RN(vocab_size)

    if torch.cuda.is_available():
        model.cuda()
        input_img = input_img.cuda()
        input_qst = input_qst.cuda()
        label = label.cuda()

    input_img = Variable(input_img)
    input_qst = Variable(input_qst)
    label = Variable(label)

    num_train = glob.glob(os.path.join(dirs, 'images', 'train', '*.png'))
    num_val = glob.glob(os.path.join(dirs, 'images', 'val', '*.png'))

    question_index = 0
    for img_idx in range(len(num_train)):
        img = Image.open(os.path.join(dirs, 'images', 'train', f'CLEVR_train_{str(img_idx).zfill(6)}.png')).convert('RGB')
        # downsample image to (3, 128, 128) from (3, 320, 480)
        # pad image to (3,136,136) then random crop to (3,128,128)
        # add random rotation between -0.05 radians and 0.05 radians
        tf = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.Pad(4),
            transforms.RandomCrop(128),
            transforms.RandomRotation(0.05*360/2/np.pi),
        ])
        img = tf(img)
        img = np.asarray(img)
        img = np.swapaxes(img, 0, 2)

        while question_index<len(train_questions['questions']) and train_questions['questions'][question_index]['image_index'] == img_idx:
            question = train_questions['questions'][question_index]['question']
            # remove trailing question mark and convert to lowercase
            if question[-1] == '?':
                question = question[:-1]
            question = question.lower()
            question = question.split()
            question = [word_to_int[w] for w in question]
            # pad question if necessary
            question = question + [1]*(max_question_length-len(question))
            answer = train_questions['questions'][question_index]['answer']
            answer = word_to_int[answer.lower()]
            rel_train.append((img, question, answer))
            question_index += 1

    question_index = 0
    for img_idx in range(len(num_val)):
        img = Image.open(os.path.join(dirs, 'images', 'val', f'CLEVR_val_{str(img_idx).zfill(6)}.png')).convert('RGB')
        # downsample image to (3, 128, 128) from (3, 320, 480)
        # pad image to (3,136,136) then random crop to (3,128,128)
        # add random rotation between -0.05 radians and 0.05 radians
        tf = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.Pad(4),
            transforms.RandomCrop(128),
            transforms.RandomRotation(0.05*360/2/np.pi),
        ])
        img = tf(img)
        img = np.asarray(img)
        img = np.swapaxes(img, 0, 2)

        while question_index<len(val_questions['questions']) and val_questions['questions'][question_index]['image_index'] == img_idx:
            question = val_questions['questions'][question_index]['question']
            # remove trailing question mark and convert to lowercase
            if question[-1] == '?':
                question = question[:-1]
            question = question.lower()
            question = question.split()
            question = [word_to_int[w] for w in question]
            # pad question if necessary
            question = question + [1]*(max_question_length-len(question))
            answer = val_questions['questions'][question_index]['answer']
            answer = word_to_int[answer.lower()]
            rel_val.append((img, question, answer))
            question_index += 1

    return (rel_train, rel_val, input_img, input_qst, label, vocab_size, model)


rel_train, rel_val, input_img, input_qst, label, vocab_size, model = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))



with open(f'./RN_{seed}_log.csv', 'w') as log_file:
    csv_writer = csv.writer(log_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_acc','val_acc'])

    print("Training RN")

    for epoch in range(1, epochs + 1):
        train_acc = train(
            epoch, rel_train)
        val_acc = validate(
            epoch, rel_val)

        csv_writer.writerow([epoch, train_acc, val_acc])
        model.save_model(epoch)
