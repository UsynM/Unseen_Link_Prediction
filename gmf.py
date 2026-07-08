# -*- coding: UTF-8 -*-
import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint


class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.num_ASnode1_info_type = config['num_ASnode1_info_type']
        self.num_ASnode2_info_type = config['num_ASnode2_info_type']
        self.num_ASnode1_AS_tier = config['num_ASnode1_AS_tier']
        self.num_ASnode2_AS_tier = config['num_ASnode2_AS_tier']
        self.num_ASnode1_info_traffic = config['num_ASnode1_info_traffic']
        self.num_ASnode2_info_traffic = config['num_ASnode2_info_traffic']
        self.num_ASnode1_info_ratio = config['num_ASnode1_info_ratio']
        self.num_ASnode2_info_ratio = config['num_ASnode2_info_ratio']
        self.num_ASnode1_info_scope = config['num_ASnode1_info_scope']
        self.num_ASnode2_info_scope = config['num_ASnode2_info_scope']
        self.num_ASnode1_policy_general = config['num_ASnode1_policy_general']
        self.num_ASnode2_policy_general = config['num_ASnode2_policy_general']
        self.num_ASnode1_policy_locations = config[
            'num_ASnode1_policy_locations']
        self.num_ASnode2_policy_locations = config[
            'num_ASnode2_policy_locations']
        self.num_ASnode1_policy_ratio = config['num_ASnode1_policy_ratio']
        self.num_ASnode2_policy_ratio = config['num_ASnode2_policy_ratio']
        self.num_ASnode1_policy_contracts = config[
            'num_ASnode1_policy_contracts']
        self.num_ASnode2_policy_contracts = config[
            'num_ASnode2_policy_contracts']

        self.num_ASnode1_appearIXP = config['num_ASnode1_appearIXP']
        self.num_ASnode2_appearIXP = config['num_ASnode2_appearIXP']
        self.num_ASnode1_appearFac = config['num_ASnode1_appearFac']
        self.num_ASnode2_appearFac = config['num_ASnode2_appearFac']



        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users,
                                                 embedding_dim=self.latent_dim)
        self.ASnode1_info_type = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_type,
            embedding_dim=self.num_ASnode1_info_type)
        self.ASnode1_AS_tier = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_AS_tier,
            embedding_dim=self.num_ASnode1_AS_tier)
        self.ASnode1_info_traffic = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_traffic,
            embedding_dim=self.num_ASnode1_info_traffic)
        self.ASnode1_info_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_ratio,
            embedding_dim=self.num_ASnode1_info_ratio)
        self.ASnode1_info_scope = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_info_scope,
            embedding_dim=self.num_ASnode1_info_scope)
        self.ASnode1_policy_general = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_general,
            embedding_dim=self.num_ASnode1_policy_general)
        self.ASnode1_policy_locations = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_locations,
            embedding_dim=self.num_ASnode1_policy_locations)
        self.ASnode1_policy_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_ratio,
            embedding_dim=self.num_ASnode1_policy_ratio)
        self.ASnode1_policy_contracts = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_policy_contracts,
            embedding_dim=self.num_ASnode1_policy_contracts)
        self.ASnode1_appearIXP = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_appearIXP, embedding_dim=15, padding_idx=0)

        self.ASnode1_appearFac = torch.nn.Embedding(
            num_embeddings=self.num_ASnode1_appearFac, embedding_dim=20, padding_idx=0)


        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items,
                                                 embedding_dim=self.latent_dim)
        self.ASnode2_info_type = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_type,
            embedding_dim=self.num_ASnode2_info_type)
        self.ASnode2_AS_tier = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_AS_tier,
            embedding_dim=self.num_ASnode2_AS_tier)
        self.ASnode2_info_traffic = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_traffic,
            embedding_dim=self.num_ASnode2_info_traffic)
        self.ASnode2_info_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_ratio,
            embedding_dim=self.num_ASnode2_info_ratio)
        self.ASnode2_info_scope = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_info_scope,
            embedding_dim=self.num_ASnode2_info_scope)
        self.ASnode2_policy_general = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_general,
            embedding_dim=self.num_ASnode2_policy_general)
        self.ASnode2_policy_locations = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_locations,
            embedding_dim=self.num_ASnode2_policy_locations)
        self.ASnode2_policy_ratio = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_ratio,
            embedding_dim=self.num_ASnode2_policy_ratio)
        self.ASnode2_policy_contracts = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_policy_contracts,
            embedding_dim=self.num_ASnode2_policy_contracts)
        self.ASnode2_appearIXP = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_appearIXP, embedding_dim=15, padding_idx=0)
        self.ASnode2_appearFac = torch.nn.Embedding(
            num_embeddings=self.num_ASnode2_appearFac, embedding_dim=20, padding_idx=0)



        self.affine_output = torch.nn.Linear(
            in_features=self.latent_dim + self.num_ASnode1_info_type +
            self.num_ASnode1_AS_tier + self.num_ASnode1_info_traffic +
            self.num_ASnode1_info_ratio + self.num_ASnode1_info_scope +
            self.num_ASnode1_policy_general +
            self.num_ASnode1_policy_locations + self.num_ASnode1_policy_ratio +
            self.num_ASnode1_policy_contracts+35 ,
            out_features=1)

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, ASnode1_info_type,
                ASnode1_AS_tier,  ASnode1_info_traffic,
                ASnode1_info_ratio, ASnode1_info_scope, ASnode1_policy_general,
                ASnode1_policy_locations, ASnode1_policy_ratio,
                ASnode1_policy_contracts,ASnode1_appearIXP, ASnode1_appearFac,
                ASnode2_info_type, ASnode2_AS_tier, ASnode2_info_traffic,
                ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general,
                ASnode2_policy_locations, ASnode2_policy_ratio,
                ASnode2_policy_contracts,ASnode2_appearIXP,
                ASnode2_appearFac):

        # 辅助函数：对 padded 序列做 masked mean，忽略 padding_idx=0 的位置
        def masked_mean(emb_layer, indices):
            """对 Embedding 输出沿 dim=1 做 mask mean，忽略 padding (index==0)"""
            emb = emb_layer(indices)  # [B, seq_len, emb_dim]
            mask = (indices != 0).float().unsqueeze(-1)  # [B, seq_len, 1]
            emb = emb * mask
            denom = mask.sum(dim=1).clamp(min=1)  # [B, 1], 至少为1防止除0
            return emb.sum(dim=1) / denom  # [B, emb_dim]

        user_embedding = torch.cat(
            (self.embedding_user(user_indices),
             self.ASnode1_info_type(ASnode1_info_type),
             self.ASnode1_AS_tier(ASnode1_AS_tier),
             self.ASnode1_info_traffic(ASnode1_info_traffic),
             self.ASnode1_info_ratio(ASnode1_info_ratio),
             self.ASnode1_info_scope(ASnode1_info_scope),
             self.ASnode1_policy_general(ASnode1_policy_general),
             self.ASnode1_policy_locations(ASnode1_policy_locations),
             self.ASnode1_policy_ratio(ASnode1_policy_ratio),
             self.ASnode1_policy_contracts(ASnode1_policy_contracts),
             masked_mean(self.ASnode1_appearIXP, ASnode1_appearIXP),
             masked_mean(self.ASnode1_appearFac, ASnode1_appearFac)), 1)

        item_embedding = torch.cat(
            (self.embedding_item(item_indices),
             self.ASnode2_info_type(ASnode2_info_type),
             self.ASnode2_AS_tier(ASnode2_AS_tier),
             self.ASnode2_info_traffic(ASnode2_info_traffic),
             self.ASnode2_info_ratio(ASnode2_info_ratio),
             self.ASnode2_info_scope(ASnode2_info_scope),
             self.ASnode2_policy_general(ASnode2_policy_general),
             self.ASnode2_policy_locations(ASnode2_policy_locations),
             self.ASnode2_policy_ratio(ASnode2_policy_ratio),
             self.ASnode2_policy_contracts(ASnode2_policy_contracts),
             masked_mean(self.ASnode2_appearIXP, ASnode2_appearIXP),
             masked_mean(self.ASnode2_appearFac, ASnode2_appearFac)), 1)


        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = logits
        return rating

    def init_weight(self):
        pass

    def load_pretrained_model(self, model_path, device_id):
        """加载完整的GMF预训练模型参数"""
        resume_checkpoint(self, model_path, device_id)


class GMFEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = GMF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(GMFEngine, self).__init__(config)