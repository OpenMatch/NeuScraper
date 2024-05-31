import math
import torch.nn as nn
from metrics import *
from transformers import BertConfig, XLMRobertaConfig, XLMRobertaModel
from transformers.models.bert.modeling_bert import BertEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.LayerNorm = nn.LayerNorm(d_model)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len / 2, dtype=torch.float).unsqueeze(1)
        position = position.repeat(1, 2).view(-1, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        if len(hidden_dim) >= 1:
            for hdim in hidden_dim:
                self.layers.append(nn.Linear(current_dim, hdim))
                self.layers.append(nn.ReLU())
                current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)
        return out


class ContentExtractionTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model_version = config.model_version
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.max_sequence_len = config.max_sequence_len
        self.num_classes = config.num_classes
        self.text_in_emb_dim = config.text_in_emb_dim
        self.text_emb_dim = config.text_emb_dim
        self.hidden = MLP(self.text_emb_dim, config.num_classes, []) 
        self.max_token_len = config.max_token_len
        self.enable_positional_encoding = not config.disable_positional_encoding

        print("Positional Encoding Enabled?: " + str(self.enable_positional_encoding))

        if self.enable_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model=self.text_emb_dim, max_len=config.max_sequence_len)

        self.textlinear = nn.Linear(
            config.text_in_emb_dim, config.text_emb_dim
        )  # 768 -> 256

        configuration = BertConfig(
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            hidden_size=self.text_emb_dim,
            intermediate_size=1024,
            output_hidden_states=False,
            output_attentions=False,
        )
        self.encoder = BertEncoder(configuration)

        text_roberta_config = XLMRobertaConfig.from_pretrained(
            "xlm-roberta-base",
            num_attention_heads=12,
            num_hidden_layers=config.text_encoder_num_hidden_layer,
        )
        self.text_roberta = XLMRobertaModel(text_roberta_config)


    def forward(self, x):
        [token_ids, token_masks] = x
        seq_len = self.max_sequence_len
        text_in_emb_dim = self.text_in_emb_dim
        max_token_len = self.max_token_len

        token_ids = token_ids.view(-1, max_token_len)  # [batch * max_sequence_len, max_token_len]
        token_masks = token_masks.view(-1, max_token_len)  # [batch * max_sequence_len, max_token_len]

        features = []

        text_output = self.text_roberta(input_ids=token_ids, attention_mask=token_masks)
        all_text_emb = text_output.pooler_output.reshape(-1, seq_len, text_in_emb_dim)

        text_x = self.textlinear(all_text_emb)
        features.append(text_x)

        text_visual_x = torch.cat(features, 2)

        if self.enable_positional_encoding:
            text_visual_x = text_visual_x.permute(1, 0, 2)

            text_visual_x = self.pos_encoder(text_visual_x)
            text_visual_x = text_visual_x.permute(1, 0, 2)

             
        if 'bert' in self.model_version:
            emb_output = self.encoder(text_visual_x, head_mask=[None, None, None])[0]
        else:
            emb_output = text_visual_x

        x_hidden = self.hidden(emb_output)
        output = self.sigmoid(x_hidden)
        return output
