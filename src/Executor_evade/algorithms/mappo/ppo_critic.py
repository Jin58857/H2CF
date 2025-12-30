import torch
import torch.nn as nn

from ..utils.mlp import MLPBase, MLPLayer, MLPBase2
from ..utils.gru import GRULayer
from ..utils.utils import check


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads=1):
        super(AttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.0)

    def forward(self, query, key, value, mask=None):
        """
        query: [1, batch_size, embed_dim]
        key: [num_keys, batch_size, embed_dim]
        value: [num_keys, batch_size, embed_dim]
        mask: [batch_size, num_keys] (optional)
        """
        attn_output, attn_weights = self.multihead_attn(
            query, key, value, key_padding_mask=~mask if mask is not None else None
        )
        # attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)  # 残差连接和归一化
        attn_output = self.activation(self.fc(attn_output))
        return attn_output, attn_weights


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 临时变量定义
        self.own_feat_dim = 17
        self.ally_feat_dim = 17
        self.enemy_feat_dim = 17
        self.embed_dim = 128
        self.hidden_dim = 128
        self.max_allies = 5
        self.max_enemies = 6
        self.batch_size = 6

        # 嵌入层
        self.own_embed = nn.Linear(self.own_feat_dim, self.embed_dim)
        self.ally_embed = nn.Linear(self.ally_feat_dim, self.embed_dim)
        self.enemy_embed = nn.Linear(self.enemy_feat_dim, self.embed_dim)

        # 注意力层
        self.attention = AttentionLayer(self.embed_dim, self.hidden_dim, 2)

        # (1) feature extraction module
        self.base = MLPBase2(self.hidden_dim, self.hidden_size, self.activation_id, self.use_feature_normalization)

        # (2) rnn module 
        input_size = self.base.output_size
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)


    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        batch_size = obs.size(0)
        # 分解 obs 为 own_feat, ally_feats, enemy_feats
        own_feat = obs[:, :self.own_feat_dim]  # [batch_size, 17]
        ally_feats = obs[:, self.own_feat_dim:self.own_feat_dim + self.max_allies * self.ally_feat_dim]
        ally_feats = ally_feats.view(batch_size, self.max_allies, self.ally_feat_dim)  # [batch_size, 5, 15]
        enemy_feats = obs[:, self.own_feat_dim + self.max_allies * self.ally_feat_dim:]
        enemy_feats = enemy_feats.view(batch_size, self.max_enemies, self.enemy_feat_dim)  # [batch_size, 6, 17]

        # 嵌入
        own_emb = self.own_embed(own_feat)  # [batch_size, embed_dim]
        ally_emb = self.ally_embed(ally_feats)  # [batch_size, max_allies, embed_dim]
        enemy_emb = self.enemy_embed(enemy_feats)  # [batch_size, max_enemies, embed_dim]

        # 拼接键和值（只包含友方和敌方的信息，不包含自身）
        combined_emb = torch.cat([ally_emb, enemy_emb], dim=1)  # [batch_size, max_keys, embed_dim]

        # 转换为注意力需要的形状：[num_keys, batch_size, embed_dim]
        combined_emb = combined_emb.permute(1, 0, 2)  # [max_keys, batch_size, embed_dim]
        own_emb = own_emb.unsqueeze(0)  # [1, batch_size, embed_dim]

        # 创建掩码
        max_keys = combined_emb.size(0)  # max_keys = max_allies + max_enemies
        mask = torch.zeros(batch_size, max_keys, dtype=torch.bool, device=own_feat.device)

        # 注意力机制
        attn_output, attn_weights = self.attention(own_emb, combined_emb, combined_emb)
        # attn_output: [1, batch_size, hidden_dim]
        attn_output = attn_output.squeeze(0)  # [batch_size, hidden_dim]
        attn_weights = attn_weights.squeeze(1)

        critic_features = self.base(attn_output)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states

