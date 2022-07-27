import argparse
import os

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

