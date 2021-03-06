# -*- coding: utf-8 -*-
"""EWC-prelims.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lY8dY5xPqo_n2Imo9D6_e-6ZtCIuzeLO

# Fine Tuning Transformer for MultiLabel Text Classification
"""

# Installing the transformers library and additional libraries if looking process 

!pip install transformers==3.0.2

# Code for TPU packages install
# !curl -q https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev

# Importing stock ml libraries
import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig

# Preparing for TPU usage
# import torch_xla
# import torch_xla.core.xla_model as xm
# device = xm.xla_device()

# # Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

!nvidia-smi

"""<a id='section02'></a>
### Importing and Pre-Processing the domain data

We will be working with the data and preparing for fine tuning purposes. 
*Assuming that the `train.csv` is already downloaded, unzipped and saved in your `data` folder*

* Import the file in a dataframe and give it the headers as per the documentation.
* Taking the values of all the categories and coverting it into a list.
* The list is appened as a new column and other columns are removed
"""

df = pd.read_csv('https://raw.githubusercontent.com/DLNoobs/snli/master/snli_1.0_test.csv')
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.1,random_state = 42,shuffle = True)

train.head(10)

def get_data_eng_eng(a):
  b = list(a['gold_label'])
  lab = []
  
  for i in b:
    if i=='contradiction':
        lab.append(0)
    elif i=='neutral':
        lab.append(1)
    elif i== 'entailment':
        lab.append(2)
    else:
        lab.append(3)
  sentence_1 = list(a['sentence1'])
  sentence_2 = list(a['sentence2'])
  raw_data_train = {'sentence1_eng': sentence_1, 
              'sentence2_eng': sentence_2,
          'label': lab}
  df = pd.DataFrame(raw_data_train, columns = ['sentence1_eng','sentence2_eng','label'])
  return df

train_eng_eng = get_data_eng_eng(train)
test_eng_eng = get_data_eng_eng(test)

# Sections of config

# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_BATCH_SIZE = 28
VALID_BATCH_SIZE = 28
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

i = tokenizer.encode_plus('I have a book', 'Not so good book',add_special_tokens=True,
            max_length=12,
            pad_to_max_length=True,
            return_token_type_ids=True)

i

j = i['input_ids']

tokenizer.convert_ids_to_tokens(j)

i = tokenizer.encode_plus('I have a book',None,add_special_tokens=True,
            max_length=12,
            pad_to_max_length=True,
            return_token_type_ids=True)

tokenizer.convert_ids_to_tokens(i['input_ids'])

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.sentence1 = dataframe.sentence1_eng
        self.sentence2 = dataframe.sentence2_eng
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, index):
        sentence1 = str(self.sentence1[index])
        sentence1 = " ".join(sentence1.split())
        sentence2 = str(self.sentence2[index])
        sentence2 = " ".join(sentence2.split())

        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

# Creating the dataset and dataloader for the neural network



training_set = CustomDataset(train_eng_eng, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_eng_eng, tokenizer, MAX_LEN)



train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self,nout,mod_name=None):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased') if mod_name==None else transformers.BertModel.from_pretrained(mod_name)
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, nout)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass(4)
model.to(device)

"""
all_bert_params = model.l1.state_dict()
for param in all_bert_params:
  print(all_bert_params[param])
"""

def save_and_get(model,PATH):
  model.l1.save_pretrained(PATH)
  return PATH

def get_linear_weight(model_from,model_to):
  with torch.no_grad():
    model_to.l3.weight = model_from.l3.weight
    model_to.l3.bias = model_from.l3.bias

!nvidia-smi

'''
del params['l3.weight']
del params['l3.bias']
'''

"""model = BERTClass(save_and_get(model,'trained_bert1'))
model.to(device)

all_bert_params = model.l1.state_dict()
for param in all_bert_params:
  print(all_bert_params[param])
"""

#from torch.nn import functional as F
def loss_fn(outputs, targets,ewc = False,star_vars = None,precision_matrices = None):
    if ewc:
      loss = torch.nn.functional.cross_entropy(outputs, targets)
      for n, p in model.named_parameters():
          if n!='l3.weight' or n!= 'l3.bias':
            _loss = precision_matrices[n] * (p - star_vars[n]) ** 2
            loss += _loss.sum()
      return loss
    else:
      return torch.nn.functional.cross_entropy(outputs, targets)

def get_optimizer(model):
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
  return optimizer

"""<a id='section05'></a>
### Fine Tuning the Model

After all the effort of loading and preparing the data and datasets, creating the model and defining its loss and optimizer. This is probably the easier steps in the process. 

Here we define a training function that trains the model on the training dataset created above, specified number of times (EPOCH), An epoch defines how many times the complete data will be passed through the network. 

Following events happen in this function to fine tune the neural network:
- The dataloader passes data to the model based on the batch size. 
- Subsequent output from the model and the actual category are compared to calculate the loss. 
- Loss value is used to optimize the weights of the neurons in the network.
- After every 5000 steps the loss value is printed in the console.

As you can see just in 1 epoch by the final step the model was working with a miniscule loss of 0.022 i.e. the network output is extremely close to the actual output.
"""

def train(epoch,training_loader,model):
    optimizer = get_optimizer(model)
    model.train()

    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)
        #print(outputs)
        #print(targets)
        
        loss = loss_fn(outputs, targets)
        if _==0:
           print(f'Epoch begin: {epoch}, Loss:  {loss.item()}')
   
        
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch end: {epoch}, Loss:  {loss.item()}')

def train_ewc(epoch,training_loader,model,star_vars,precision_matrices):
    #star_vars = star_vars.to(device)
    #precision_matrices = precision_matrices.to(device)
    optimizer = get_optimizer(model)
    model.train()

    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)
        #print(outputs)
        #print(targets)
        
        loss = loss_fn(outputs, targets,True,star_vars,precision_matrices)
        if _==0:
           print(f'Epoch begin: {epoch}, Loss:  {loss.item()}')
   
        
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch end: {epoch}, Loss:  {loss.item()}')

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)

for epoch in range(EPOCHS):
    train(epoch,training_loader,model)

"""<a id='section06'></a>
### Validating the Model

During the validation stage we pass the unseen data(Testing Dataset) to the model. This step determines how good the model performs on the unseen data. 

This unseen data is the 20% of `train.csv` which was seperated during the Dataset creation stage. 
During the validation stage the weights of the model are not updated. Only the final output is compared to the actual value. This comparison is then used to calcuate the accuracy of the model. 

As defined above to get a measure of our models performance we are using the following metrics. 
- Accuracy Score
- F1 Micro
- F1 Macro

We are getting amazing results for all these 3 categories just by training the model for 1 Epoch.
"""

def validation(testing_loader,model):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

"""
for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
"""

out,tar = validation(testing_loader,model)

from sklearn.metrics import accuracy_score
accuracy_score(np.argmax(out,axis = 1),tar)

from google.colab import drive
drive.mount("/content/drive")
!cp '/content/drive/My Drive/IMDB/IMDB Dataset.csv.zip' 'IMDB.zip'

!unzip IMDB.zip



df = pd.read_csv('IMDB Dataset.csv')
df = df.head(5000)

from sklearn.model_selection import train_test_split
train_imdb, test_imdb = train_test_split(df, test_size=0.15,random_state = 42,shuffle = True)

def get_data(a):
  b = list(a['sentiment'])
  lab = []
  
  for i in b:
    if i=='positive':
        lab.append(1)
    elif i=='negative':
        lab.append(0)
    
  sentence = list(a['review'])
 
  raw_data_train = {'sentence': sentence, 
              
          'label': lab}
  df = pd.DataFrame(raw_data_train, columns = ['sentence','label'])
  return df

train_imdb = get_data(train_imdb)
test_imdb = get_data(test_imdb)

train_imdb

class CustomDatasetIMDB(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.sentence = dataframe.sentence
        
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        sentence1 = str(self.sentence[index])
        sentence1 = " ".join(sentence1.split())
        

        inputs = self.tokenizer.encode_plus(
            sentence1,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

training_set_imdb = CustomDatasetIMDB(train_imdb, tokenizer, MAX_LEN)
testing_set_imdb = CustomDatasetIMDB(test_imdb, tokenizer, MAX_LEN)
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader_imdb = DataLoader(training_set_imdb, **train_params)
testing_loader_imdb = DataLoader(testing_set_imdb, **test_params)

!ls

model_imdb = BERTClass(2,save_and_get(model,'nli7'))
model_imdb.to(device)



for epoch in range(EPOCHS):
    train(epoch,training_loader_imdb,model_imdb)

out,tar = validation(testing_loader_imdb,model_imdb)
accuracy_score(np.argmax(out,axis = 1),tar)

model_nli = BERTClass(4,save_and_get(model_imdb,'imdb7'))
get_linear_weight(model,model_nli)
model_nli.to(device)

out,tar = validation(testing_loader,model_nli)
accuracy_score(np.argmax(out,axis = 1),tar)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return torch.autograd.Variable(t, **kwargs)

def compute_fisher(model,dataset):
  params = {n: p for n, p in model.named_parameters() if p.requires_grad}
  precision_matrices = {}
  for n, p in deepcopy(params).items():
      p.data.zero_()
      precision_matrices[n] = variable(p.data)

  model.eval()
  for _,data in enumerate(dataset, 0):
      ids = data['ids'].to(device, dtype = torch.long)
      mask = data['mask'].to(device, dtype = torch.long)
      token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
      targets = data['targets'].to(device, dtype = torch.long)
      
      output = model(ids, mask, token_type_ids).view(1, -1)
      #output = self.model(input).view(1, -1)
      label = output.max(1)[1].view(-1)
      loss = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), label)
      loss.backward()

      for n, p in model.named_parameters():
          precision_matrices[n].data += p.grad.data ** 2 / len(dataset)

  precision_matrices = {n: p for n, p in precision_matrices.items()}
  return precision_matrices

m1 = compute_fisher(model_imdb,testing_loader_imdb)
m2 = compute_fisher(model,testing_loader)

from copy import deepcopy

m1#imdb

m2#nli

model_imdb = BERTClass(2,save_and_get(model,'nli7'))
model_imdb.to(device)

star_vars = {}
for n, p in model.named_parameters():
  star_vars[n] = p

for epoch in range(EPOCHS):
    train_ewc(epoch,training_loader_imdb,model_imdb,star_vars,m2)

out,tar = validation(testing_loader_imdb,model_imdb)
accuracy_score(np.argmax(out,axis = 1),tar)

model_nli_ewc = BERTClass(4,save_and_get(model_imdb,'imdb8'))
get_linear_weight(model,model_nli_ewc)
model_nli_ewc.to(device)

out,tar = validation(testing_loader,model_nli_ewc)
accuracy_score(np.argmax(out,axis = 1),tar)

