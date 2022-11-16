import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='CMU-Mosi Dataset Config')

    # basic settings
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='whether to use gpu')
    parser.add_argument('-v', '--version', type=str, default='01',
                        help='the version of model')
    parser.add_argument('-s', '--seed', type=int, default=12,
                        help='random seed')
    parser.add_argument('--data_dir', type=str, default='../data/mosi_data/mosi_data.pkl')
    parser.add_argument('--model_path', type=str, default='./checkpoint/')

    # training
    parser.add_argument('--is_pretrain', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=200,
                        help='the total number of epoch in the training')
    parser.add_argument('--pre_epoch', type=int, default=30,
                        help='the number of epoch in the pre-training')
    parser.add_argument('--batch', type=int, default=1,
                        help='the batch size for instance-specific training')
    parser.add_argument('--pre_batch', type=int, default=64)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--multi_lr', type=float, default=0.0001)
    parser.add_argument('--text_ft_lr', type=float, default=5e-5)
    parser.add_argument('--vision_ft_lr', type=float, default=1e-5)
    parser.add_argument('--audio_ft_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay_text', type=float, default=0.001)
    parser.add_argument('--weight_decay_vision', type=float, default=0.001)
    parser.add_argument('--weight_decay_audio', type=float, default=0.001)
    parser.add_argument('--weight_decay_multi', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)

    parser.add_argument('--num_label', type=int, default=7)
    parser.add_argument('--text_dim', type=int, default=300)
    parser.add_argument('--vision_dim', type=int, default=20)
    parser.add_argument('--audio_dim', type=int, default=5)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--fusion', type=str, default='concat',
                        help='type of fusion (concat, mean, max)')
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.5)

    args = parser.parse_args()

    return args






