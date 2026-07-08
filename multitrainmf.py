# -*- coding: UTF-8 -*-
"""
SGMF Training Entry Point (CUDA version)
用法: python multitrainmf.py
数据默认从 ../sgmf_myx/data/AShopmatrix/ 读取
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator


def train(sample_generator, gmf_config):
    print('=== Training started ===')
    config = gmf_config
    engine = GMFEngine(config)

    for epoch in range(config['num_epoch']):
        print('\nEpoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        evaluate_data = sample_generator.evaluate_data  # 每次取 DataLoader（轻量）
        rmse = engine.evaluate(evaluate_data, epoch_id=epoch)
        print('[Epoch {}] RMSE = {:.4f}'.format(epoch, rmse))
        engine.save(config['alias'], epoch, rmse)


def main():
    parser = argparse.ArgumentParser(description='SGMF Training (CUDA)')
    parser.add_argument('--data_dir', type=str, default='../sgmf_myx/data/AShopmatrix',
                        help='Path to data directory containing train/validate CSV files')
    parser.add_argument('--train_file', type=str, default='train_hop_matrix_sample_1000.csv',
                        help='Training file name')
    parser.add_argument('--valid_file', type=str, default='validate_hop_matrix_sample_1000.csv',
                        help='Validation file name')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--device_id', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_users', type=int, default=77438, help='Number of users (max ASN)')
    parser.add_argument('--ixp_pad', type=int, default=1125, help='appearIXP padding length')
    parser.add_argument('--fac_pad', type=int, default=4380, help='appearFac padding length')

    # Feature cardinalities (根据数据统计设置)
    parser.add_argument('--num_info_type', type=int, default=11)
    parser.add_argument('--num_as_tier', type=int, default=5)
    parser.add_argument('--num_info_traffic', type=int, default=19)
    parser.add_argument('--num_info_ratio', type=int, default=6)
    parser.add_argument('--num_info_scope', type=int, default=10)
    parser.add_argument('--num_policy_general', type=int, default=5)
    parser.add_argument('--num_policy_locations', type=int, default=6)
    parser.add_argument('--num_policy_ratio', type=int, default=3)
    parser.add_argument('--num_policy_contracts', type=int, default=4)

    args = parser.parse_args()

    # 数据路径
    train_path = os.path.join(args.data_dir, args.train_file)
    valid_path = os.path.join(args.data_dir, args.valid_file)

    if not os.path.exists(train_path):
        print(f"ERROR: Training file not found: {train_path}")
        print("Please check --data_dir and --train_file arguments.")
        sys.exit(1)
    if not os.path.exists(valid_path):
        print(f"ERROR: Validation file not found: {valid_path}")
        print("Please check --data_dir and --valid_file arguments.")
        sys.exit(1)

    print(f"Loading training data: {train_path}")
    # 使用逗号分隔，兼容含引号的字段
    AShop_train_rating = pd.read_csv(
        train_path,
        sep=r',(?=(?:[^"]*"[^"]*")*[^"]*$)',
        header=None,
        names=['userId', 'itemId', 'rating',
               'ASnode1_info_type', 'ASnode1_AS_tier', 'ASnode1_info_traffic',
               'ASnode1_info_ratio', 'ASnode1_info_scope', 'ASnode1_policy_general',
               'ASnode1_policy_locations', 'ASnode1_policy_ratio',
               'ASnode1_policy_contracts', 'ASnode1_appearIXP', 'ASnode1_appearFac',
               'ASnode2_info_type', 'ASnode2_AS_tier', 'ASnode2_info_traffic',
               'ASnode2_info_ratio', 'ASnode2_info_scope', 'ASnode2_policy_general',
               'ASnode2_policy_locations', 'ASnode2_policy_ratio',
               'ASnode2_policy_contracts', 'ASnode2_appearIXP', 'ASnode2_appearFac'],
        engine='python')
    AShop_train_rating.sort_values(by=["userId", "itemId"], inplace=True)
    AShop_train_rating.reset_index(drop=True, inplace=True)
    print(f"Train data shape: {AShop_train_rating.shape}")

    print(f"Loading validation data: {valid_path}")
    AShop_valid_rating = pd.read_csv(
        valid_path,
        sep=r',(?=(?:[^"]*"[^"]*")*[^"]*$)',
        header=None,
        names=['userId', 'itemId', 'rating',
               'ASnode1_info_type', 'ASnode1_AS_tier', 'ASnode1_info_traffic',
               'ASnode1_info_ratio', 'ASnode1_info_scope', 'ASnode1_policy_general',
               'ASnode1_policy_locations', 'ASnode1_policy_ratio',
               'ASnode1_policy_contracts', 'ASnode1_appearIXP', 'ASnode1_appearFac',
               'ASnode2_info_type', 'ASnode2_AS_tier', 'ASnode2_info_traffic',
               'ASnode2_info_ratio', 'ASnode2_info_scope', 'ASnode2_policy_general',
               'ASnode2_policy_locations', 'ASnode2_policy_ratio',
               'ASnode2_policy_contracts', 'ASnode2_appearIXP', 'ASnode2_appearFac'],
        engine='python')
    AShop_valid_rating.sort_values(by=["userId", "itemId"], inplace=True)
    AShop_valid_rating.reset_index(drop=True, inplace=True)
    print(f"Valid data shape: {AShop_valid_rating.shape}")

    sample_generator = SampleGenerator(
        train_ratings=AShop_train_rating,
        valid_ratings=AShop_valid_rating,
        ixp_pad_len=args.ixp_pad,
        fac_pad_len=args.fac_pad)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print(f"CUDA available, using device {args.device_id}")
    else:
        print("Using CPU")

    alias = 'gmf_nfactors{}_nepochs{}_nbatch{}_lr{}'.format(
        args.latent_dim, args.epochs, args.batch_size, args.lr)

    gmf_config = {
        'alias': alias,
        'num_epoch': args.epochs,
        'batch_size': args.batch_size,
        'optimizer': 'adam',
        'adam_lr': args.lr,
        'num_users': args.num_users,
        'num_items': args.num_users,
        'num_ASnode1_info_type': args.num_info_type,
        'num_ASnode2_info_type': args.num_info_type,
        'num_ASnode1_AS_tier': args.num_as_tier,
        'num_ASnode2_AS_tier': args.num_as_tier,
        'num_ASnode1_info_traffic': args.num_info_traffic,
        'num_ASnode2_info_traffic': args.num_info_traffic,
        'num_ASnode1_info_ratio': args.num_info_ratio,
        'num_ASnode2_info_ratio': args.num_info_ratio,
        'num_ASnode1_info_scope': args.num_info_scope,
        'num_ASnode2_info_scope': args.num_info_scope,
        'num_ASnode1_policy_general': args.num_policy_general,
        'num_ASnode2_policy_general': args.num_policy_general,
        'num_ASnode1_policy_locations': args.num_policy_locations,
        'num_ASnode2_policy_locations': args.num_policy_locations,
        'num_ASnode1_policy_ratio': args.num_policy_ratio,
        'num_ASnode2_policy_ratio': args.num_policy_ratio,
        'num_ASnode1_policy_contracts': args.num_policy_contracts,
        'num_ASnode2_policy_contracts': args.num_policy_contracts,
        'num_ASnode1_appearIXP': args.ixp_pad,
        'num_ASnode2_appearIXP': args.ixp_pad,
        'num_ASnode1_appearFac': args.fac_pad,
        'num_ASnode2_appearFac': args.fac_pad,
        'latent_dim': args.latent_dim,
        'l2_regularization': 0,
        'use_cuda': use_cuda,
        'device_id': args.device_id,
        'model_dir': 'checkpoints/' + alias + '/{}_Epoch{}_RMSE{:.4f}.model'
    }

    # 确保 checkpoint 目录存在
    os.makedirs(os.path.dirname(gmf_config['model_dir'].format('', 0, 0.0)), exist_ok=True)

    train(sample_generator, gmf_config)


if __name__ == '__main__':
    import torch
    main()
