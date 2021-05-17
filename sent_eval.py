# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import fire

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tokenization
import models
import optim
import train

from utils import set_seeds, get_device, truncate_tokens_pair
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_bert import BertModel
from torch.autograd import Variable

class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, file, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        data = []
        with open(file, "r") as f:
            # list of splitted lines : line is also list
            #lines = csv.reader(f, delimiter='\t', quotechar=None)
            lines = csv.reader(f, delimiter=',', quotechar=None)
            for instance in self.get_instances(lines): # instance : tuple of fields
                for proc in pipeline: # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError


class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b

class SIM(CsvDataset):
    """ Dataset class for SIM """
    labels = ("0", "1") # label names
    def __init__(self, file, pipeline=[]):
        super().__init__(file, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[1], '' # label, text


def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'mrpc': MRPC, 'mnli': MNLI, 'sim': SIM}
    return table[task]


class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

#class Classifier(nn.Module):
class SentEmbedding(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels, local_pretrained=False):
        super().__init__()

        if(local_pretrained):
            self.transformer = models.Transformer(cfg)
        else:
          self.transformer = BertModel.from_pretrained('bert-base-uncased')
        
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        #self.classifier = nn.Linear(cfg.dim, n_labels)
        self.local_pretrained = local_pretrained

    def forward(self, input_ids, segment_ids, input_mask):
        if(self.local_pretrained):
            h = self.transformer(input_ids, segment_ids, input_mask)
            print(h)
            print(np.shape(h)) # [1, 64, 768]
            # only use the first h in the sequence
            pooled_h = self.activ(self.fc(h[:, 0]))
            return pooled_h
        else:
            #input_ids = torch.LongTensor(input_ids)
            #segment_ids = torch.LongTensor(segment_ids)
            #input_mask = torch.LongTensor(input_mask)
            
            h, _ = self.transformer(input_ids, segment_ids, input_mask)
            print(h)
            print(np.shape(h)) # (12,)
            print(np.shape(h[0])) # [1,64,768]
            h0 = h[0]
            pooled_h = self.activ(self.fc(h0[:,0]))
            #pooled_h = self.activ(self.fc(h))
            #pooled_h = self.activ(self.fc(h[:, 0]))
            return pooled_h

        #logits = self.classifier(self.drop(pooled_h))
        #return logits

class SentEvaluator(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_iter, optimizer, save_dir, device, local_pretrained):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_iter = data_iter # iterator to load data
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.device = device # device name
        self.local_pretrained = local_pretrained # google or local model

    def load(self, model_file, pretrain_file):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            self.model.load_state_dict(torch.load(model_file))

        elif pretrain_file: # use pretrained transformer
            print('Loading the pretrained model from', pretrain_file)
            if pretrain_file.endswith('.ckpt'): # checkpoint file in tensorflow
                checkpoint.load_model(self.model.transformer, pretrain_file)
            elif pretrain_file.endswith('.pt'): # pretrain model file in pytorch
                self.model.transformer.load_state_dict(
                    {key[12:]: value
                        for key, value in torch.load(pretrain_file).items()
                        if key.startswith('transformer')}
                ) # load only transformer parts

    def eval(self, evaluate, model_file, data_parallel=True):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        if(self.local_pretrained):
            self.load(model_file, None)
        
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
        for batch in iter_bar:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                result = evaluate(model, batch) # accuracy to print
            print('eval(batch) : ', result.shape)
            results.append(result)
            #results.append(result.cpu().tolist())

        return results

#pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
#pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',
pretrain_file='./pretrain/model_steps_9386.pt',
#data_file='./data/birds/example_dataset.txt'

def main(task='sim',
         train_cfg='config/train_mrpc.json',
         model_cfg='config/bert_base.json',
         data_file='../glue/MRPC/train.tsv',
         model_file=None,
         pretrain_file=pretrain_file,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/mrpc',
         max_len=128,
         batch_size=2,
         pretrained_type='local',
         mode='train'):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    #set_seeds(cfg.seed)

    if(pretrained_type == 'google'):
        local_pretrained = False
    else:
        local_pretrained = True

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task) # task dataset class according to the task
    pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                AddSpecialTokensWithTruncation(max_len),
                TokenIndexing(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]
    dataset = TaskDataset(data_file, pipeline)
    # batch_size
    #data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #model = Classifier(model_cfg, len(TaskDataset.labels))
    model = SentEmbedding(model_cfg, len(TaskDataset.labels), local_pretrained)

    #trainer = train.Trainer(cfg,
    evaluator = SentEvaluator(cfg,
                            model,
                            data_iter,
                            optim.optim4GPU(cfg, model),
                            save_dir, get_device(), local_pretrained)

    if(True):
        def evaluate(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            #logits = model(input_ids, segment_ids, input_mask)
            if(local_pretrained):
                print(np.shape(input_ids), np.shape(segment_ids), np.shape(input_mask))
                embed = model(input_ids, segment_ids, input_mask)
            else:
                #input_ids = torch.LongTensor(input_ids)
                #segment_ids = torch.LongTensor(segment_ids)
                #input_mask = torch.LongTensor(input_mask)
                print(np.shape(input_ids), np.shape(segment_ids), np.shape(input_mask))
                print(input_ids.shape, segment_ids.shape, input_mask.shape)
                embed = model(input_ids, segment_ids, input_mask)

            print('evaluate(embed) : ', embed.shape)
            return embed
            
        #results = trainer.eval(evaluate, model_file, data_parallel)
        results = evaluator.eval(evaluate, model_file, data_parallel)
        print(np.shape(results))

    similarities = []
    for svec in results:
        sim = cosine_similarity(results[0], svec)
        print(sim)
        similarities.append(sim.cpu().tolist())
    
    print(similarities)

if __name__ == '__main__':
    fire.Fire(main)
