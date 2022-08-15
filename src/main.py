import argparse
import os
from data_loader import load_data
from train import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def print_setting(args):
    assert args.use_context or args.use_path
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('epoch: ' + str(args.epoch))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('feature_type: ' + args.feature_type)

    print('use relational context: ' + str(args.use_context))
    if args.use_context:
        print('context_hops: ' + str(args.context_hops))
        print('neighbor_samples: ' + str(args.neighbor_samples))
        print('neighbor_agg: ' + args.neighbor_agg)

    print('use relational path: ' + str(args.use_path))
    if args.use_path:
        print('max_path_len: ' + str(args.max_path_len))
        print('path_type: ' + args.path_type)
        if args.path_type == 'rnn':
            print('path_samples: ' + str(args.path_samples))
            print('path_agg: ' + args.path_agg)
    print('=============================================')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='wn18rr', help='dataset name')
    args = parser.parse_known_args()[0]

    parser.add_argument('--cuda', default=False, help='use gpu', action='store_true')
    if args.dataset == 'FB15k':
        # ===== FB15k ===== #
        parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
        parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')
        # settings for relational context
        parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
        parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
        parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
        parser.add_argument('--neighbor_agg', type=str, default='concat',
                            help='neighbor aggregator: mean, concat, cross')

        # settings for relational path
        parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
        parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')  # default=2
        parser.add_argument('--path_type', type=str, default='embedding',
                            help='path representation type: embedding, rnn')
        parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
        parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    elif args.dataset == 'FB15k-237':
        # ===== FB15k-237 ===== #
        parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
        parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

        # settings for relational context
        parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
        parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
        parser.add_argument('--neighbor_samples', type=int, default=32, help='number of sampled neighbors for one hop')
        parser.add_argument('--neighbor_agg', type=str, default='concat',
                            help='neighbor aggregator: mean, concat, cross')

        # settings for relational path
        parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
        parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
        parser.add_argument('--path_type', type=str, default='embedding',
                            help='path representation type: embedding, rnn')
        parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
        parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    elif args.dataset == 'wn18':
        # ===== wn18 ===== #
        parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
        parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

        # settings for relational context
        parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
        parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')  # default=3
        parser.add_argument('--neighbor_samples', type=int, default=16, help='number of sampled neighbors for one hop')
        parser.add_argument('--neighbor_agg', type=str, default='concat',
                            help='neighbor aggregator: mean, concat, cross')  # default='cross'

        # settings for relational path
        parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
        parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
        parser.add_argument('--path_type', type=str, default='embedding',
                            help='path representation type: embedding, rnn')
        parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
        parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    elif args.dataset == 'wn18rr':
        # ===== wn18rr ===== #
        parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
        parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

        # settings for relational context
        parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
        parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')  # default=3
        parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
        parser.add_argument('--neighbor_agg', type=str, default='concat',
                            help='neighbor aggregator: mean, concat, cross')  # default='cross'

        # settings for relational path
        parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
        parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')  # default=4
        parser.add_argument('--path_type', type=str, default='embedding',
                            help='path representation type: embedding, rnn')
        parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
        parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    elif args.dataset == 'NELL995':
        # ===== NELL995 ===== #
        parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
        parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

        # settings for relational context
        parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
        parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
        parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
        parser.add_argument('--neighbor_agg', type=str, default='concat',
                            help='neighbor aggregator: mean, concat, cross')

        # settings for relational path
        parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
        parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')
        parser.add_argument('--path_type', type=str, default='embedding',
                            help='path representation type: embedding, rnn')
        parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
        parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')
    elif args.dataset == 'DDB14':
        # ===== DDB14 ===== #
        parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size')
        parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
        parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')

        # settings for relational context
        parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
        parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')  # default=3
        parser.add_argument('--neighbor_samples', type=int, default=8, help='number of sampled neighbors for one hop')
        parser.add_argument('--neighbor_agg', type=str, default='concat',
                            help='neighbor aggregator: mean, concat, cross')  # default='cross'

        # settings for relational path
        parser.add_argument('--use_path', type=bool, default=True, help='whether use relational path')
        parser.add_argument('--max_path_len', type=int, default=3, help='max length of a path')  # default=4
        parser.add_argument('--path_type', type=str, default='embedding',
                            help='path representation type: embedding, rnn')
        parser.add_argument('--path_samples', type=int, default=8, help='number of sampled paths if using rnn')
        parser.add_argument('--path_agg', type=str, default='att', help='path aggregator if using rnn: mean, att')

    args = parser.parse_args()
    print_setting(args)
    data = load_data(args)
    train(args, data)

if __name__ == "__main__":
    main()