import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Multimodal Resume Evaluation')

    # basic settings
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='whether to use gpu')
    parser.add_argument('-v', '--version', type=str, default='01',
                        help='the version of model')
    parser.add_argument('-s', '--seed', type=int, default=12,
                        help='random seed')
    parser.add_argument('--data_dir', type=str, default='../data/resume_data',
                        help='dir of data')
    parser.add_argument('--model_path', type=str, default='./checkpoint/')

    # training
    parser.add_argument('--is_pretrain', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=100,
                        help='the total number of epoch in the training')
    parser.add_argument('--pre_epoch', type=int, default=15,
                        help='the number of epoch in the pre-training')
    parser.add_argument('--batch', type=int, default=1,
                        help='the batch size for instance-specific training')
    parser.add_argument('--pre_batch', type=int, default=64,
                        help='the batch size for pre-training')
    parser.add_argument('--pre_lr', type=float, default=0.001,
                        help='the learning rate for pre-training')
    parser.add_argument('--multi_lr', type=float, default=0.0001,
                        help='the learning rate for multimodal encoder')
    parser.add_argument('--text_ft_lr', type=float, default=5e-5,
                        help='the learning rate for fine-tune the text encoder')
    parser.add_argument('--image_ft_lr', type=float, default=1e-5,
                        help='the learning rate for fine-tune the image encoder')
    parser.add_argument('--multi_ft_lr', type=float, default=1e-5,
                        help='the learning rate for the multimodal encoder in the finetune process')
    parser.add_argument('--weight_decay_text', type=float, default=0.001,
                        help='L2 penalty factor of the text Adam optimizer')
    parser.add_argument('--weight_decay_image', type=float, default=0.001,
                        help='L2 penalty factor of the image Adam optimizer')
    parser.add_argument('--weight_decay_multi', type=float, default=0.001,
                        help='L2 penalty factor of the multimodal Adam optimizer')

    # encoder
    parser.add_argument('--num_label', type=int, default=2,
                        help='number of class')
    parser.add_argument('--image_dim', type=int, default=2048,
                        help='hidden dime of image embedding')
    parser.add_argument('--context_dim', type=int, default=768,
                        help='hidden dim of contextual embedding')
    parser.add_argument('--attr_dim', type=int, default=58,
                        help='hidden dim of attribute one-hot vector')
    parser.add_argument('--feature_dim', type=int, default=128,
                        help='hidden dim of feature embedding')
    parser.add_argument('--fusion', type=str, default='concat',
                        help='type of fusion (concat, mean, max)')
    # hyper-parameter
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.2)

    args = parser.parse_args()
    return args




