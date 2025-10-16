import torch
import torch.nn as nn

class MultiHeadAttentionFeatureExtractor(nn.Module):
    """
    Feature extractor that projects input features and applies a MultiheadAttention
    to produce a fixed-length feature vector for each sample.
    """
    def __init__(self, input_dim, num_heads=2):
        super(MultiHeadAttentionFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        # project input to a common embedding dimension used by attention
        self.input_projection = nn.Linear(input_dim, 24)
        # multi-head attention, batch_first=True expects (batch, seq, embed)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=24, num_heads=num_heads, batch_first=True)
        # final feature projection to reduce dimension
        self.feature_projection = nn.Sequential(nn.Linear(24, 20))

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_projection(x)      # (batch, 24)
        x = x.unsqueeze(1)               # (batch, 1, 24) as a single "token" sequence
        attn_output, _ = self.multihead_attn(x, x, x)
        x = attn_output.squeeze(1)       # (batch, 24)
        x = self.feature_projection(x)   # (batch, 20)
        return x

class MultiHeadAttentionFeatureExtractor1(MultiHeadAttentionFeatureExtractor):
    """
    Separate class kept for clarity; currently same architecture as above.
    """
    pass

class DNN(nn.Module):
    """
    Fully-connected network to combine features and make final regression output.
    """
    def __init__(self, input_dim, Drop=0.1):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.01), nn.BatchNorm1d(512), nn.Dropout(Drop),
            nn.Linear(512, 512), nn.LeakyReLU(0.01), nn.BatchNorm1d(512), nn.Dropout(Drop),
            nn.Linear(512, 512), nn.LeakyReLU(0.01), nn.BatchNorm1d(512), nn.Dropout(Drop),
            nn.Linear(512, 512), nn.LeakyReLU(0.01), nn.BatchNorm1d(512), nn.Dropout(Drop),
            nn.Linear(512, 256), nn.LeakyReLU(0.01), nn.BatchNorm1d(256), nn.Dropout(Drop),
            nn.Linear(256, 128), nn.LeakyReLU(0.01), nn.BatchNorm1d(128), nn.Dropout(Drop),
            nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.BatchNorm1d(64), nn.Dropout(Drop),
            nn.Linear(64, 32), nn.LeakyReLU(0.01), nn.BatchNorm1d(32), nn.Dropout(Drop),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

class CombinedModel(nn.Module):
    """
    Combined model that extracts attention features from earthquake inputs and structure inputs,
    concatenates them with small DNN inputs and feeds to a DNN for regression output.
    eq_input_dim: dimension of earthquake numeric input (e.g., 19)
    dnn_input_dim: small dense inputs (e.g., 3)
    struct_input_dim: structural categorical/one-hot length (e.g., 16)
    """
    def __init__(self, eq_input_dim, dnn_input_dim, struct_input_dim, Drop=0.0):
        super(CombinedModel, self).__init__()
        self.attention = MultiHeadAttentionFeatureExtractor(eq_input_dim)
        self.attentionstruct = MultiHeadAttentionFeatureExtractor1(struct_input_dim)
        # attention produces 20 + 20 features; so total to DNN = dnn_input_dim + 40
        self.dnn = DNN(dnn_input_dim + 40, Drop=Drop)

    def forward(self, X_struct, x_eq, x_dnn):
        # X_struct: structural features -> attentionstruct
        # x_eq: earthquake features -> attention
        # x_dnn: small dense features
        attention_features = self.attention(x_eq)
        attention_featuresstruct = self.attentionstruct(X_struct)
        combined_features = torch.cat([attention_features, attention_featuresstruct, x_dnn], dim=1)
        return self.dnn(combined_features)