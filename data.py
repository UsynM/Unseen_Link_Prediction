# -*- coding: UTF-8 -*-
import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from ast import literal_eval
random.seed(0)
import numpy as np


def dealpadding(lis_str, max_len):
    """将字符串列表padding到固定长度，处理引号问题"""
    lis_str = str(lis_str).replace('"', '')  # 去掉双引号
    list1 = np.zeros(max_len, dtype=np.int64)
    eval_lis = literal_eval(lis_str)
    lis_len = len(eval_lis)
    for i in range(0, lis_len):
        list1[i] = int(eval_lis[i] + 1)
    return list1


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor,
                 ASnode1_info_type_tensor, ASnode1_AS_tier_tensor,
                 ASnode1_info_traffic_tensor, ASnode1_info_ratio_tensor,
                 ASnode1_info_scope_tensor, ASnode1_policy_general_tensor,
                 ASnode1_policy_locations_tensor, ASnode1_policy_ratio_tensor,
                 ASnode1_policy_contracts_tensor, ASnode1_appearIXP_tensor,
                 ASnode1_appearFac_tensor,
                 ASnode2_info_type_tensor, ASnode2_AS_tier_tensor,
                 ASnode2_info_traffic_tensor, ASnode2_info_ratio_tensor,
                 ASnode2_info_scope_tensor, ASnode2_policy_general_tensor,
                 ASnode2_policy_locations_tensor, ASnode2_policy_ratio_tensor,
                 ASnode2_policy_contracts_tensor, ASnode2_appearIXP_tensor,
                 ASnode2_appearFac_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.ASnode1_info_type_tensor = ASnode1_info_type_tensor
        self.ASnode1_AS_tier_tensor = ASnode1_AS_tier_tensor
        self.ASnode1_info_traffic_tensor = ASnode1_info_traffic_tensor
        self.ASnode1_info_ratio_tensor = ASnode1_info_ratio_tensor
        self.ASnode1_info_scope_tensor = ASnode1_info_scope_tensor
        self.ASnode1_policy_general_tensor = ASnode1_policy_general_tensor
        self.ASnode1_policy_locations_tensor = ASnode1_policy_locations_tensor
        self.ASnode1_policy_ratio_tensor = ASnode1_policy_ratio_tensor
        self.ASnode1_policy_contracts_tensor = ASnode1_policy_contracts_tensor
        self.ASnode1_appearIXP_tensor = ASnode1_appearIXP_tensor
        self.ASnode1_appearFac_tensor = ASnode1_appearFac_tensor
        self.ASnode2_info_type_tensor = ASnode2_info_type_tensor
        self.ASnode2_AS_tier_tensor = ASnode2_AS_tier_tensor
        self.ASnode2_info_traffic_tensor = ASnode2_info_traffic_tensor
        self.ASnode2_info_ratio_tensor = ASnode2_info_ratio_tensor
        self.ASnode2_info_scope_tensor = ASnode2_info_scope_tensor
        self.ASnode2_policy_general_tensor = ASnode2_policy_general_tensor
        self.ASnode2_policy_locations_tensor = ASnode2_policy_locations_tensor
        self.ASnode2_policy_ratio_tensor = ASnode2_policy_ratio_tensor
        self.ASnode2_policy_contracts_tensor = ASnode2_policy_contracts_tensor
        self.ASnode2_appearIXP_tensor = ASnode2_appearIXP_tensor
        self.ASnode2_appearFac_tensor = ASnode2_appearFac_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], \
               self.ASnode1_info_type_tensor[index], self.ASnode1_AS_tier_tensor[index], \
               self.ASnode1_info_traffic_tensor[index], self.ASnode1_info_ratio_tensor[index], \
               self.ASnode1_info_scope_tensor[index], self.ASnode1_policy_general_tensor[index], \
               self.ASnode1_policy_locations_tensor[index], self.ASnode1_policy_ratio_tensor[index], \
               self.ASnode1_policy_contracts_tensor[index], self.ASnode1_appearIXP_tensor[index], \
               self.ASnode1_appearFac_tensor[index], \
               self.ASnode2_info_type_tensor[index], \
               self.ASnode2_AS_tier_tensor[index], self.ASnode2_info_traffic_tensor[index], \
               self.ASnode2_info_ratio_tensor[index], self.ASnode2_info_scope_tensor[index], \
               self.ASnode2_policy_general_tensor[index], self.ASnode2_policy_locations_tensor[index], \
               self.ASnode2_policy_ratio_tensor[index], self.ASnode2_policy_contracts_tensor[index], \
               self.ASnode2_appearIXP_tensor[index], self.ASnode2_appearFac_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class ASFeatureStore(object):
    """AS-id keyed tensor store for compact hop matrix datasets."""
    scalar_features = (
        "info_type",
        "AS_tier",
        "info_traffic",
        "info_ratio",
        "info_scope",
        "policy_general",
        "policy_locations",
        "policy_ratio",
        "policy_contracts",
    )

    def __init__(self, feature_frame, ixp_pad_len=1125, fac_pad_len=4380):
        feature_frame = feature_frame.copy()
        feature_frame["asId"] = feature_frame["asId"].astype(np.int64)
        self.max_as_id = int(feature_frame["asId"].max()) if len(feature_frame) else 0
        self.scalars = {
            name: torch.zeros(self.max_as_id + 1, dtype=torch.long)
            for name in self.scalar_features
        }
        self.appear_ixp = torch.zeros((self.max_as_id + 1, ixp_pad_len), dtype=torch.long)
        self.appear_fac = torch.zeros((self.max_as_id + 1, fac_pad_len), dtype=torch.long)

        for row in feature_frame.itertuples(index=False):
            as_id = int(row.asId)
            for name in self.scalar_features:
                self.scalars[name][as_id] = int(getattr(row, name))
            self.appear_ixp[as_id] = torch.from_numpy(dealpadding(row.appearIXP, ixp_pad_len))
            self.appear_fac[as_id] = torch.from_numpy(dealpadding(row.appearFac, fac_pad_len))

        self.approx_bytes = sum(t.nelement() * t.element_size() for t in self.scalars.values())
        self.approx_bytes += self.appear_ixp.nelement() * self.appear_ixp.element_size()
        self.approx_bytes += self.appear_fac.nelement() * self.appear_fac.element_size()

    def side_tuple(self, as_id):
        return (
            self.scalars["info_type"][as_id],
            self.scalars["AS_tier"][as_id],
            self.scalars["info_traffic"][as_id],
            self.scalars["info_ratio"][as_id],
            self.scalars["info_scope"][as_id],
            self.scalars["policy_general"][as_id],
            self.scalars["policy_locations"][as_id],
            self.scalars["policy_ratio"][as_id],
            self.scalars["policy_contracts"][as_id],
            self.appear_ixp[as_id],
            self.appear_fac[as_id],
        )


class CompactHopDataset(Dataset):
    """Dataset backed by userId,itemId,rating plus an ASFeatureStore."""
    def __init__(self, hop_frame, feature_store):
        self.users = torch.from_numpy(hop_frame["userId"].to_numpy(dtype=np.int64, copy=True))
        self.items = torch.from_numpy(hop_frame["itemId"].to_numpy(dtype=np.int64, copy=True))
        self.ratings = torch.from_numpy(hop_frame["rating"].to_numpy(dtype=np.float32, copy=True))
        self.feature_store = feature_store

    def __len__(self):
        return self.ratings.size(0)

    def __getitem__(self, idx):
        user = self.users[idx]
        item = self.items[idx]
        return (
            user,
            item,
            *self.feature_store.side_tuple(user),
            *self.feature_store.side_tuple(item),
            self.ratings[idx],
        )


class SampleGenerator(object):
    """Construct dataset for NCF"""
    def __init__(self, train_ratings, valid_ratings,
                 ixp_pad_len=1125, fac_pad_len=4380, seed=None,
                 as_features=None, evaluate_batch_size=2048):
        self.train_ratings = train_ratings
        self.valid_ratings = valid_ratings
        self.ixp_pad_len = ixp_pad_len
        self.fac_pad_len = fac_pad_len
        self.seed = seed
        self._train_loader_calls = 0
        self.evaluate_batch_size = evaluate_batch_size
        self.train_dataset = None
        self.valid_dataset = None

        if as_features is not None:
            print("Building AS feature tensor store...")
            self.feature_store = ASFeatureStore(as_features, ixp_pad_len, fac_pad_len)
            print("AS feature store: {} AS rows, max_id={}, tensor_memory={:.1f} MB".format(
                len(as_features),
                self.feature_store.max_as_id,
                self.feature_store.approx_bytes / (1024.0 * 1024.0)))
            self.train_dataset = CompactHopDataset(train_ratings, self.feature_store)
            self.valid_dataset = CompactHopDataset(valid_ratings, self.feature_store)

    # ========== 懒加载 Train DataLoader（仅持有DataFrame引用，延迟到getitem才处理） ==========
    def instance_a_train_loader(self, batch_size):
        if self.train_dataset is not None:
            generator = None
            if self.seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(self.seed) + self._train_loader_calls)
                self._train_loader_calls += 1
            return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=False, num_workers=0,
                              generator=generator)

        """每epoch调用但不再重新遍历DataFrame，DataLoader内部按需逐条getitem"""
        ixp_pad = self.ixp_pad_len
        fac_pad = self.fac_pad_len

        class TrainDataset(torch.utils.data.Dataset):
            def __init__(self, df):
                self.df = df

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                return (
                    torch.tensor(int(row.userId), dtype=torch.long),
                    torch.tensor(int(row.itemId), dtype=torch.long),
                    # ASnode1 side info
                    torch.tensor(int(row.ASnode1_info_type), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_AS_tier), dtype=torch.long),       # BUGFIX: 原来误写成 info_type
                    torch.tensor(int(row.ASnode1_info_traffic), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_info_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_info_scope), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_general), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_locations), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_contracts), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode1_appearIXP, ixp_pad), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode1_appearFac, fac_pad), dtype=torch.long),
                    # ASnode2 side info
                    torch.tensor(int(row.ASnode2_info_type), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_AS_tier), dtype=torch.long),       # BUGFIX: 原来误写成 info_type
                    torch.tensor(int(row.ASnode2_info_traffic), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_info_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_info_scope), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_general), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_locations), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_contracts), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode2_appearIXP, ixp_pad), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode2_appearFac, fac_pad), dtype=torch.long),
                    torch.tensor(float(row.rating), dtype=torch.float32),
                )

        dataset = TrainDataset(self.train_ratings)
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.seed) + self._train_loader_calls)
            self._train_loader_calls += 1
        return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          pin_memory=False, num_workers=0,
                          generator=generator)

    # ========== 批量 Validation DataLoader（替代原来一次性返回大tensor的 evaluate_data） ==========
    @property
    def evaluate_data(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset, batch_size=self.evaluate_batch_size, shuffle=False,
                              pin_memory=False, num_workers=0)

        """返回 DataLoader，支持逐batch评估，避免显存OOM"""
        ixp_pad = self.ixp_pad_len
        fac_pad = self.fac_pad_len

        class ValDataset(torch.utils.data.Dataset):
            def __init__(self, df):
                self.df = df

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                return (
                    torch.tensor(int(row.userId), dtype=torch.long),
                    torch.tensor(int(row.itemId), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_info_type), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_AS_tier), dtype=torch.long),       # BUGFIX
                    torch.tensor(int(row.ASnode1_info_traffic), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_info_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_info_scope), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_general), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_locations), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode1_policy_contracts), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode1_appearIXP, ixp_pad), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode1_appearFac, fac_pad), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_info_type), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_AS_tier), dtype=torch.long),       # BUGFIX
                    torch.tensor(int(row.ASnode2_info_traffic), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_info_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_info_scope), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_general), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_locations), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_ratio), dtype=torch.long),
                    torch.tensor(int(row.ASnode2_policy_contracts), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode2_appearIXP, ixp_pad), dtype=torch.long),
                    torch.tensor(dealpadding(row.ASnode2_appearFac, fac_pad), dtype=torch.long),
                    torch.tensor(float(row.rating), dtype=torch.float32),
                )

        dataset = ValDataset(self.valid_ratings)
        return DataLoader(dataset, batch_size=self.evaluate_batch_size, shuffle=False,
                          pin_memory=False, num_workers=0)
