# -*- coding: UTF-8 -*-
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, precision_score
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK
import numpy as np


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """
    def __init__(self, config):
        self.config = config  # model configuration
        self._writer = SummaryWriter(log_dir='runs/{}'.format(
            config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # loss function
        self.crit = torch.nn.MSELoss()

    def train_single_batch(
            self, users, items, ASnode1_info_type, ASnode1_AS_tier,
            ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
            ASnode1_policy_general, ASnode1_policy_locations,
            ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP,
            ASnode1_appearFac, ASnode2_info_type, ASnode2_AS_tier,
            ASnode2_info_traffic, ASnode2_info_ratio, ASnode2_info_scope,
            ASnode2_policy_general, ASnode2_policy_locations,
            ASnode2_policy_ratio, ASnode2_policy_contracts, ASnode2_appearIXP,
            ASnode2_appearFac, ratings):

        assert hasattr(self, 'model'), 'Please specify the exact model !'
        # move to GPU
        if self.config['use_cuda'] is True:
            users, items, ASnode1_info_type, ASnode1_AS_tier, ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope, \
            ASnode1_policy_general, ASnode1_policy_locations, ASnode1_policy_ratio, \
            ASnode1_policy_contracts, ASnode1_appearIXP, ASnode1_appearFac, ASnode2_info_type, \
            ASnode2_AS_tier, ASnode2_info_traffic, \
            ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general, ASnode2_policy_locations, \
            ASnode2_policy_ratio, ASnode2_policy_contracts, ASnode2_appearIXP, ASnode2_appearFac, ratings = \
                users.cuda(), items.cuda(), ASnode1_info_type.cuda(), ASnode1_AS_tier.cuda(), \
                ASnode1_info_traffic.cuda(), \
                ASnode1_info_ratio.cuda(), ASnode1_info_scope.cuda(), ASnode1_policy_general.cuda(), \
                ASnode1_policy_locations.cuda(), ASnode1_policy_ratio.cuda(), ASnode1_policy_contracts.cuda(), \
                ASnode1_appearIXP.cuda(), ASnode1_appearFac.cuda(), ASnode2_info_type.cuda(), \
                ASnode2_AS_tier.cuda(), \
                ASnode2_info_traffic.cuda(), ASnode2_info_ratio.cuda(), ASnode2_info_scope.cuda(), \
                ASnode2_policy_general.cuda(), ASnode2_policy_locations.cuda(), ASnode2_policy_ratio.cuda(), \
                ASnode2_policy_contracts.cuda(), ASnode2_appearIXP.cuda(), ASnode2_appearFac.cuda(), ratings.cuda()

        self.opt.zero_grad()

        ratings_pred = self.model(
            users, items, ASnode1_info_type, ASnode1_AS_tier,
            ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
            ASnode1_policy_general, ASnode1_policy_locations,
            ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP,
            ASnode1_appearFac, ASnode2_info_type, ASnode2_AS_tier,
            ASnode2_info_traffic, ASnode2_info_ratio, ASnode2_info_scope,
            ASnode2_policy_general, ASnode2_policy_locations,
            ASnode2_policy_ratio, ASnode2_policy_contracts, ASnode2_appearIXP,
            ASnode2_appearFac)
        # calculate loss
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0

        for batch_id, batch in enumerate(train_loader):
            # 兼容 lazy DataLoader 返回 LongTensor（不再是 ShortTensor）
            user, item, \
            ASnode1_info_type, ASnode1_AS_tier, ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope, ASnode1_policy_general, ASnode1_policy_locations, \
            ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP, ASnode1_appearFac, \
            ASnode2_info_type, ASnode2_AS_tier, ASnode2_info_traffic, ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general, ASnode2_policy_locations, \
            ASnode2_policy_ratio, ASnode2_policy_contracts, ASnode2_appearIXP, ASnode2_appearFac, rating = (
                batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8],
                batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15], batch[16], batch[17],
                batch[18], batch[19], batch[20], batch[21], batch[22], batch[23], batch[24])

            rating = rating.float()
            loss = self.train_single_batch(
                user, item, ASnode1_info_type, ASnode1_AS_tier,
                ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
                ASnode1_policy_general, ASnode1_policy_locations,
                ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP, ASnode1_appearFac,
                ASnode2_info_type,
                ASnode2_AS_tier, ASnode2_info_traffic,
                ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general,
                ASnode2_policy_locations, ASnode2_policy_ratio,
                ASnode2_policy_contracts, ASnode2_appearIXP, ASnode2_appearFac, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(
                epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    # ========== 批量评估（替代原来一次性上GPU的版本，避免显存OOM） ==========
    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        test_ratings_all = []
        test_scores_all = []

        with torch.no_grad():
            for batch in tqdm(evaluate_data, desc='Evaluating'):
                test_users, test_items, ASnode1_info_type, ASnode1_AS_tier, ASnode1_info_traffic, \
                ASnode1_info_ratio, ASnode1_info_scope, ASnode1_policy_general, \
                ASnode1_policy_locations, ASnode1_policy_ratio, ASnode1_policy_contracts, \
                ASnode1_appearIXP, ASnode1_appearFac, ASnode2_info_type, ASnode2_AS_tier, ASnode2_info_traffic, \
                ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general, \
                ASnode2_policy_locations, ASnode2_policy_ratio, ASnode2_policy_contracts, \
                ASnode2_appearIXP, ASnode2_appearFac, batch_test_ratings = (
                    batch[0], batch[1], batch[2], batch[3],
                    batch[4], batch[5], batch[6], batch[7],
                    batch[8], batch[9], batch[10], batch[11],
                    batch[12], batch[13], batch[14], batch[15],
                    batch[16], batch[17], batch[18], batch[19],
                    batch[20], batch[21], batch[22], batch[23],
                    batch[24])

                # Move batch to GPU
                if self.config['use_cuda'] is True:
                    test_users = test_users.cuda()
                    test_items = test_items.cuda()
                    ASnode1_info_type = ASnode1_info_type.cuda()
                    ASnode1_AS_tier = ASnode1_AS_tier.cuda()
                    ASnode1_info_traffic = ASnode1_info_traffic.cuda()
                    ASnode1_info_ratio = ASnode1_info_ratio.cuda()
                    ASnode1_info_scope = ASnode1_info_scope.cuda()
                    ASnode1_policy_general = ASnode1_policy_general.cuda()
                    ASnode1_policy_locations = ASnode1_policy_locations.cuda()
                    ASnode1_policy_ratio = ASnode1_policy_ratio.cuda()
                    ASnode1_policy_contracts = ASnode1_policy_contracts.cuda()
                    ASnode1_appearIXP = ASnode1_appearIXP.cuda()
                    ASnode1_appearFac = ASnode1_appearFac.cuda()
                    ASnode2_info_type = ASnode2_info_type.cuda()
                    ASnode2_AS_tier = ASnode2_AS_tier.cuda()
                    ASnode2_info_traffic = ASnode2_info_traffic.cuda()
                    ASnode2_info_ratio = ASnode2_info_ratio.cuda()
                    ASnode2_info_scope = ASnode2_info_scope.cuda()
                    ASnode2_policy_general = ASnode2_policy_general.cuda()
                    ASnode2_policy_locations = ASnode2_policy_locations.cuda()
                    ASnode2_policy_ratio = ASnode2_policy_ratio.cuda()
                    ASnode2_policy_contracts = ASnode2_policy_contracts.cuda()
                    ASnode2_appearIXP = ASnode2_appearIXP.cuda()
                    ASnode2_appearFac = ASnode2_appearFac.cuda()

                batch_test_scores = self.model(
                    test_users, test_items, ASnode1_info_type, ASnode1_AS_tier,
                    ASnode1_info_traffic, ASnode1_info_ratio, ASnode1_info_scope,
                    ASnode1_policy_general, ASnode1_policy_locations,
                    ASnode1_policy_ratio, ASnode1_policy_contracts, ASnode1_appearIXP, ASnode1_appearFac,
                    ASnode2_info_type,
                    ASnode2_AS_tier, ASnode2_info_traffic,
                    ASnode2_info_ratio, ASnode2_info_scope, ASnode2_policy_general,
                    ASnode2_policy_locations, ASnode2_policy_ratio,
                    ASnode2_policy_contracts, ASnode2_appearIXP, ASnode2_appearFac)

                test_ratings_all.extend(batch_test_ratings.cpu().numpy())
                test_scores_all.extend(batch_test_scores.cpu().numpy())

        test_ratings_all = np.array(test_ratings_all)
        test_scores_all = np.array(test_scores_all)

        # ===== RMSE =====
        rmse = np.sqrt(
            mean_squared_error(y_pred=test_scores_all,
                               y_true=test_ratings_all))
        self._writer.add_scalar('performance/RMSE', rmse, epoch_id)

        # ===== Hop-count Binary Classification Metrics (hop=1 as direct link) =====
        metrics = self.evaluate_hop_binary(test_ratings_all, test_scores_all,
                                          threshold=0.5, verbose=True)

        # ===== Log all metrics =====
        metrics['rmse'] = rmse
        with open('metric.res', 'a') as f:
            f.write(f'Epoch {epoch_id} Evaluation Metrics:\n')
            for key, value in metrics.items():
                f.write(f'  {key}: {value}\n')
            f.write('-' * 50 + '\n')

        return rmse

    # ========== 核心映射函数：score → hop → 直连概率 ==========
    @staticmethod
    def map_score_to_link_prob(score_np):
        """
        输入：模型输出的score数组（numpy）
        输出：(hop_pred, link_prob)
          - hop_pred: 预测hop数（连续值）
          - link_prob: 存在直连的概率（概率 = 1/hop，hop越小越接近1）
        """
        score_np = score_np.astype(float).copy()

        # 找到最小正数，将 ≤0 的值替换为最小正数
        min_positive = np.min(score_np[score_np > 0]) if np.any(score_np > 0) else 1.0
        score_np = np.where(score_np <= 0, min_positive, score_np)

        # 直连概率 = min_positive / score
        link_prob = np.clip(min_positive / score_np, 0.0, 1.0)
        return score_np, link_prob

    # ========== Hop-count 二分评估 (hop=1 视为直连边) ==========
    @staticmethod
    def evaluate_hop_binary(test_ratings, test_scores, threshold=0.5, verbose=True):
        """
        评估 hop-count 预测中的"直连判断"效果
          - test_ratings: 真实 hop 值 (numpy array)
          - test_scores:  模型预测分数 (numpy array)
          - threshold:    概率阈值，默认0.5
          - verbose:      是否打印结果
        Returns: dict of metrics
        """
        test_ratings_np = np.array(test_ratings).flatten()
        test_scores_np = np.array(test_scores).flatten()

        # 二值化：hop=1 → 直连(1)，其余 → 非直连(0)
        test_ratings_binary = np.where(test_ratings_np == 1, 1, 0)
        _, link_prob = Engine.map_score_to_link_prob(test_scores_np)

        pos_count = int(np.sum(test_ratings_binary))
        neg_count = int(len(test_ratings_binary) - pos_count)

        metrics = {
            "auc": 0.0,
            "tpr_at_fpr_01": 0.0,
            "precision": 0.0,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "unique_hop_values": list(np.unique(test_ratings_np)),
            "threshold": threshold,
        }

        # 只有一类标签时跳过
        if len(np.unique(test_ratings_binary)) < 2:
            if verbose:
                print("⚠️  Warning: Only one class in binary labels, skipping classification metrics.")
                print(f"  Hop values: {metrics['unique_hop_values']}, "
                      f"Positive: {pos_count}, Negative: {neg_count}")
            return metrics

        # AUC
        metrics["auc"] = roc_auc_score(test_ratings_binary, link_prob)

        # TPR @ FPR=0.1
        fpr, tpr, _ = roc_curve(test_ratings_binary, link_prob)
        valid_idx = np.where(fpr <= 0.1)[0]
        if len(valid_idx) > 0:
            metrics["tpr_at_fpr_01"] = float(tpr[valid_idx[-1]])
            fp_cnt = int(fpr[valid_idx[-1]] * len(test_ratings_binary))
            tp_cnt = int(metrics["tpr_at_fpr_01"] * pos_count)
            print(f"  At FPR={fpr[valid_idx[-1]]:.4f}, FP Count={fp_cnt}, TP Count={tp_cnt}")

        # Precision
        test_pred_labels = (link_prob >= threshold).astype(int)
        metrics["precision"] = precision_score(test_ratings_binary, test_pred_labels, zero_division=0)

        if verbose:
            print("\n" + "=" * 55)
            print("  Hop-count Binary Classification (hop=1 = direct link)")
            print("=" * 55)
            print(f"  Unique hop values : {metrics['unique_hop_values']}")
            print(f"  Data distribution : direct={pos_count}, indirect={neg_count}")
            print(f"  AUC               : {metrics['auc']:.4f}")
            print(f"  TPR @ FPR=0.1      : {metrics['tpr_at_fpr_01']:.4f}")
            print(f"  Precision (thr={threshold}): {metrics['precision']:.4f}")
            print("=" * 55 + "\n")

        return metrics

    def save(self, alias, epoch_id, rmse):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, rmse)
        save_checkpoint(self.model, model_dir)