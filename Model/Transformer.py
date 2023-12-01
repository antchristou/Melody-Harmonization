import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

# pure transformer model
class PositionalEncoding(nn.Module):
  def __init__(self,dim_model,dropout_p,max_len):
    super().__init__()

    self.dropout = nn.Dropout(dropout_p)

    pos_encoding = torch.zeros(max_len,dim_model)
    positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
    division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

    # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
    pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

    # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
    pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

    pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer("pos_encoding",pos_encoding)

  def forward(self, token_embedding: torch.tensor) -> torch.tensor:
      # Residual connection + pos encoding
      return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):

  def __init__(
      self,
      inputVocab,
      outputVocab,
      input_embedding_dim,
      output_embedding_dim,
      num_heads,
      num_encoder_layers,
      num_decoder_layers,
      dropout_p,
      dim_feedforward
  ):

    super().__init__()
    self.model_type = "Transformer"
    self.kwargs = {'inputVocab':inputVocab,'outputVocab':outputVocab, 'input_embedding_dim': input_embedding_dim, 'output_embedding_dim': output_embedding_dim,'num_heads':num_heads, 'num_encoder_layers': num_encoder_layers,'num_decoder_layers':num_decoder_layers,'dropout_p':dropout_p,'dim_feedforward':dim_feedforward}
    self.input_embedding_dim = input_embedding_dim
    self.output_embedding_dim = output_embedding_dim

    self.input_positional_encoder = PositionalEncoding(dim_model=input_embedding_dim,dropout_p=dropout_p,max_len=5000)
    self.output_positional_encoder = PositionalEncoding(dim_model=output_embedding_dim,dropout_p=dropout_p,max_len=5000)



    self.inputEmbedding = nn.Embedding(inputVocab,input_embedding_dim)
    self.targetEmbedding = nn.Embedding(outputVocab,output_embedding_dim)

    self.transformer = nn.Transformer(
            d_model=input_embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p, batch_first=True,
            dim_feedforward = dim_feedforward
        )

    self.out = nn.Linear(output_embedding_dim,outputVocab)

  def forward(self,src,tgt,tgt_mask=None):
       # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        src = self.inputEmbedding(src) * math.sqrt(self.input_embedding_dim)
        tgt = self.targetEmbedding(tgt) * math.sqrt(self.output_embedding_dim)

        src = self.input_positional_encoder(src)
        tgt = self.output_positional_encoder(tgt)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.out(transformer_out)


        return out


  def get_tgt_mask(self,size):
    mask = torch.tril(torch.ones(size,size) == 1)
    mask = mask.float()

    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))

    return mask