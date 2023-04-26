import logging
import os
import time

from transformers import (
    HfArgumentParser,
)
from torch.utils.data import DataLoader
from utils.training_arguments import MyTrainingArguments
from utils.configue import Configure
from utils.process import DatasetForBert, DatasetForT5
from model.model import Model
import lightning.pytorch as pl

from lightning.pytorch.loggers import WandbLogger

pl.seed_everything(42)
# 获得配置
parser = HfArgumentParser((MyTrainingArguments,))
args, = parser.parse_args_into_dataclasses()
args = Configure.Get(args.cfg)
model_config = args.model
# 设置wandb
project = args.project.project
try:
    name = args.prject.name
except:
    name = time.strftime('%Y-%m-%d', time.localtime())
wandblogger = WandbLogger(project=project,
                          name=name,
                          save_dir="./logs")
# 加载数据
train_path = os.path.join(args.data.data_path,
                          args.data.train_path)
eval_path = os.path.join(args.data.data_path,
                          args.data.test_path)
train_loader = DataLoader(DatasetForBert(train_path, args.model), 
                             batch_size=args.train.batch_size, shuffle=True)
eval_lodaer = DataLoader(DatasetForBert(eval_path, args.model), 
                             batch_size=args.train.batch_size, shuffle=True)
# 定义model，设置trainer，开始训练
model = Model(args) 
trainer = pl.Trainer(gpus=[1,3], 
                     logger=wandblogger, 
                     max_epochs=args.train.epoch, 
                     strategy="ddp", 
                     )
trainer.fit(model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=eval_lodaer)