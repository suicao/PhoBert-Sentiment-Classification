import torch
from torch import nn
from transformers import *

class AlbertPreTrainedModel(PreTrainedModel):
   config_class = AlbertConfig
   pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
   base_model_prefix = "albert"

   def _init_weights(self, module):
       if isinstance(module, (nn.Linear, nn.Embedding)):
           module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
           if isinstance(module, (nn.Linear)) and module.bias is not None:
               module.bias.data.zero_()
       elif isinstance(module, nn.LayerNorm):
           module.bias.data.zero_()
           module.weight.data.fill_(1.0)

class AlbertForAIViVN(AlbertPreTrainedModel):
   def __init__(self, config):
       super(AlbertForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.albert = AlbertModel(config)
       self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)

       self.init_weights()

   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

       outputs = self.albert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
       logits = self.qa_outputs(cls_output)
       return logits

class BertForAIViVN(BertPreTrainedModel):
   def __init__(self, config):
       super(BertForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.bert = BertModel(config)
       self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)

       self.init_weights()

   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

       outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
       logits = self.qa_outputs(cls_output)
       return logits

class RobertaForAIViVN(BertPreTrainedModel):
   config_class = RobertaConfig
   pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
   base_model_prefix = "roberta"
   def __init__(self, config):
       super(RobertaForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.roberta = RobertaModel(config)
       self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)

       self.init_weights()

   def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

       outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
       logits = self.qa_outputs(cls_output)
       return logits
