import torch
import torch.nn as nn
from torch.autograd import Variable
# from constant import PAD_INDEX


class DKT_LSTM(nn.Module):
    """
    LSTM based model
    """
    def __init__(self, input_dim, hidden_dim, num_layers, question_num, dropout, mmd):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self._encoder = nn.Embedding(num_embeddings=2*question_num+1, embedding_dim=input_dim)
        self._decoder = nn.Linear(hidden_dim, question_num)
        self.classifier = self._decoder
        self.mmd = mmd

#     def init_hidden(self, batch_size):
#         """
#         initialize hidden layer as zero tensor
#         batch_size: single integer
#         """
# #         weight = next(self.parameters())
#         return (weight.new_zeros(self._num_layers, batch_size, self._hidden_dim),
#                 weight.new_zeros(self._num_layers, batch_size, self._hidden_dim))

    def forward(self, x,target_id, src_x):
        """
        get model output (before taking sigmoid) for target_id
        input: (batch_size, sequence_size)
        target_id: (batch_size)
        return output, a tensor of shape (batch_size, 1)
        """
        batch_size = x.shape[0]
#         src_batch_size = src_x.shape[0]
#         hidden = self.init_hidden(batch_size)
        device = x.device
        hidden = Variable(torch.zeros(self._num_layers, batch_size, self._hidden_dim)).to(device)
        cell = Variable(torch.zeros(self._num_layers, batch_size, self._hidden_dim)).to(device)
#         src_hidden = Variable(torch.zeros(self._num_layers, src_batch_size, self._hidden_dim)).to(device)
#         src_cell = Variable(torch.zeros(self._num_layers, src_batch_size, self._hidden_dim)).to(device)
        
        x = self._encoder(x)
        output_mmd, _ = self._lstm(x, (hidden, cell))
        if self.mmd:
            src_x = self._encoder(src_x)
            src_output_mmd, _ = self._lstm(src_x, (hidden, cell))
        output = self._decoder(output_mmd[:, -1, :])
        output = torch.gather(output, -1, target_id)
        if self.mmd:
            return output,output_mmd,src_output_mmd
        return output,output_mmd,output_mmd




#     def __init__(self, input_dim, hidden_dim, question_num, dropout):

#         super(DKT_LSTM, self).__init__()
#         self.question_num = question_num
#         self.hidden_dim = hidden_dim

#         self.embedding = nn.Embedding(2*question_num+1, input_dim)

#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=dropout)

#         self.pred = nn.Linear(hidden_dim, question_num)
    
#     def forward(self, x):
#         bs = x.size(0)
#         device = x.device
#         hidden = Variable(torch.zeros(1, bs, self.hidden_dim)).to(device)
#         cell = Variable(torch.zeros(1, bs, self.hidden_dim)).to(device)

#         x = self.embedding(x)

#         x, _ = self.lstm(x, (hidden, cell)) # lstm output:[bs, seq_len, hidden] hidden [bs, hidden]
#         x = self.pred(x[:, -1, :])

#         return x
