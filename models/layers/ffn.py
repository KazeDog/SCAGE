from torch import nn

from utils.global_var_util import DEFAULTS


class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_hidden_dim, activation_fn="GELU", dropout=DEFAULTS.DROP_RATE):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_act_func = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.dropout(self.fc2(self.act_dropout(self.ffn_act_func(self.fc1(x)))))
        x += residual
        x = self.ffn_layer_norm(x)
        return x
