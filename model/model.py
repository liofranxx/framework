import torch
import lightning.pytorch as pl
from .Bert import BertForTHUCNews
from utils.utils import skl_metric

class Model(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.config = args.model
        self.model = BertForTHUCNews(self.config)
        self.lr = args.train.lr
        self.loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()
    def forward(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        logits = self.model(input_ids, attention_mask)
        return logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        # _, loss, f1 = self._get_preds_loss_f1(preds)
        # Log loss and metric
        return {"loss":loss,"preds":preds.detach().cpu(),"y":y.detach().cpu()}
    
    #定义各种metrics
    def training_step_end(self,outputs):
        _, f1 = self._get_preds_loss_f1(outputs['preds'], outputs['y']) 
        self.log('train_loss', outputs['loss'])
        self.log('train_f1', f1)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        # _, loss, f1 = self._get_preds_loss_f1(preds)
        # Log loss and metric
        return {"loss":loss,"preds":preds.detach().cpu(),"y":y.detach().cpu()}
    
    def validation_step_end(self,outputs):
        _, f1 = self._get_preds_loss_f1(outputs['preds'], outputs['y'])   
        self.log('eval_loss', outputs['loss'])
        self.log('eval_f1', f1)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def _get_preds_loss_f1(self, logits, labels):
        preds = torch.argmax(logits, dim=1)  # [batch_size, num_class] -> [batch_size, 1]
        
        accuracy, recall, precision, f1 = skl_metric(labels.flatten(), 
                                                     preds.flatten())
        return preds, f1
    
