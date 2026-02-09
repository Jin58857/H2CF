import torch
import torch.nn as nn
import numpy as np

from ..utils.mlp import MLPBase, MLPBase2
from ..utils.gru import GRULayer
from ..utils.act import ACTLayer
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


class PPOActor(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(PPOActor, self).__init__()
        # network config
        self.gain = args.gain
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 临时变量定义
        self.own_feat_dim = 14
        self.ally_feat_dim = 14
        self.enemy_feat_dim = 14
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
        # (3) act module
        self.act = ACTLayer(act_space, input_size, self.act_hidden_size, self.activation_id, self.gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, deterministic=False):
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
        # for i in range(batch_size):
        #     # 假设 ally_num 和 enemy_num 已经包含在 obs 中，或者需要另外传入
        #     # 如果不在 obs 中，您需要将 ally_num 和 enemy_num 作为额外参数传入
        #     # 这里假设 ally_num 和 enemy_num 是固定的最大数
        #     # 如有变化，请调整代码以传入实际的 ally_num 和 enemy_num
        #     mask[i, :self.max_allies + self.max_enemies] = 1

        # 注意力机制
        attn_output, attn_weights = self.attention(own_emb, combined_emb, combined_emb)
        # attn_output: [1, batch_size, hidden_dim]
        attn_output = attn_output.squeeze(0)  # [batch_size, hidden_dim]
        attn_weights = attn_weights.squeeze(1)
        # # 将自身嵌入向量与注意力输出拼接
        # combined_features = torch.cat([attn_output, own_emb.squeeze(0)], dim=-1)  # [batch_size, hidden_dim + embed_dim]
        #
        # actor_features = self.base(combined_features)

        actor_features = self.base(attn_output)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
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
        # for i in range(batch_size):
        #     # 假设 ally_num 和 enemy_num 已经包含在 obs 中，或者需要另外传入
        #     # 如果不在 obs 中，您需要将 ally_num 和 enemy_num 作为额外参数传入
        #     # 这里假设 ally_num 和 enemy_num 是固定的最大数
        #     # 如有变化，请调整代码以传入实际的 ally_num 和 enemy_num
        #     mask[i, :self.max_allies + self.max_enemies] = 1

        # 注意力机制
        attn_output, attn_weights = self.attention(own_emb, combined_emb, combined_emb)
        # attn_output: [1, batch_size, hidden_dim]
        attn_output = attn_output.squeeze(0)  # [batch_size, hidden_dim]

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # # 将自身嵌入向量与注意力输出拼接
        # combined_features = torch.cat([attn_output, own_emb.squeeze(0)], dim=-1)  # [batch_size, hidden_dim + embed_dim]
        #
        # actor_features = self.base(combined_features)

        actor_features = self.base(attn_output)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, active_masks)

        return action_log_probs, dist_entropy

    def forward_2(self, obs, rnn_states, masks, deterministic=False):
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
        for i in range(batch_size):
            # 假设 ally_num 和 enemy_num 已经包含在 obs 中，或者需要另外传入
            # 如果不在 obs 中，您需要将 ally_num 和 enemy_num 作为额外参数传入
            # 这里假设 ally_num 和 enemy_num 是固定的最大数
            # 如有变化，请调整代码以传入实际的 ally_num 和 enemy_num
            mask[i, :self.max_allies + self.max_enemies] = 1

        # 注意力机制
        attn_output, attn_weights = self.attention(own_emb, combined_emb, combined_emb)
        # attn_output: [1, batch_size, hidden_dim]
        attn_output = attn_output.squeeze(0)  # [batch_size, hidden_dim]
        attn_weights = attn_weights.squeeze(1)
        # # 将自身嵌入向量与注意力输出拼接
        # combined_features = torch.cat([attn_output, own_emb.squeeze(0)], dim=-1)  # [batch_size, hidden_dim + embed_dim]
        #
        # actor_features = self.base(combined_features)

        actor_features = self.base(attn_output)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, deterministic)

        return actions, action_log_probs, rnn_states, attn_weights



