import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aggregators import ConcatAggregator


class PathCon(nn.Module):
    def __init__(self, args, n_relations, params_for_neighbors, params_for_paths):
        super(PathCon, self).__init__()
        self._parse_args(args, n_relations, params_for_neighbors, params_for_paths)
        self._build_model()

    def _parse_args(self, args, n_relations, params_for_neighbors, params_for_paths):
        self.n_relations = n_relations
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type
        self.use_context = args.use_context
        self.use_path = args.use_path

        if self.use_context:
            self.entity2edges = torch.LongTensor(params_for_neighbors[0]).cuda() if args.cuda \
                else torch.LongTensor(params_for_neighbors[0])
            self.edge2entities = torch.LongTensor(params_for_neighbors[1]).cuda() if args.cuda \
                else torch.LongTensor(params_for_neighbors[1])
            self.edge2relation = torch.LongTensor(params_for_neighbors[2]).cuda() if args.cuda \
                else torch.LongTensor(params_for_neighbors[2])
            self.neighbor_samples = args.neighbor_samples
            self.context_hops = args.context_hops
            self.neighbor_agg = ConcatAggregator
            # if args.neighbor_agg == 'mean':
            #     self.neighbor_agg = MeanAggregator
            # elif args.neighbor_agg == 'concat':
            #     self.neighbor_agg = ConcatAggregator
            # elif args.neighbor_agg == 'cross':
            #     self.neighbor_agg = CrossAggregator

        if self.use_path:
            self.path_type = args.path_type
            if self.path_type == 'embedding':
                self.n_paths = params_for_paths[0]
            elif self.path_type == 'rnn':
                self.max_path_len = args.max_path_len
                self.path_samples = args.path_samples
                self.path_agg = args.path_agg
                self.id2path = torch.LongTensor(params_for_paths[0]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_paths[0])
                self.id2length = torch.LongTensor(params_for_paths[1]).cuda() if args.cuda \
                    else torch.LongTensor(params_for_paths[1])

    def _build_model(self):
        """ define initial relation features """
        if self.use_context or (self.use_path and self.path_type == 'rnn'):
            self._build_relation_feature()

        self.scores = 0.0

        if self.use_context:
            self.aggregators = nn.ModuleList(self._get_neighbor_aggregators()) # define aggregators for each layer

        if self.use_path:
            if self.path_type == 'embedding':
                self.layer = nn.Linear(self.n_paths, self.n_relations)
                nn.init.xavier_uniform_(self.layer.weight)
            elif self.path_type == 'rnn':
                self.rnn = nn.LSTM(input_size=self.relation_dim, hidden_size=self.hidden_dim, batch_first=True)
                self.layer = nn.Linear(self.hidden_dim, self.n_relations)
                self.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch):
        if self.use_context:
            self.entity_pairs = batch['entity_pairs']
            self.train_edges = batch['train_edges']

        if self.use_path:
            if self.path_type == 'embedding':
                self.path_features = batch['path_features']
            elif self.path_type == 'rnn':
                self.path_ids = batch['path_ids']

        self.labels = batch['labels']

        self._call_model()

    def _get_neighbor_aggregators(self, relations, entity_pairs, train_edges):
        aggregators = []    # store all aggregators

        if self.context_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=F.relu))
            # middle layers
            for i in range(self.context_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=F.relu))

            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
            return aggregators

    @staticmethod
    def train_step(model, optimizer, batch):
        model.train()
        optimizer.zero_grad()
        model(batch)
        criterion = nn.CrossEntropyLoss()
        loss = torch.mean(criterion(model.scores, model.labels))
        loss.backward()
        optimizer.step()

        return loss.item()

    @staticmethod
    def test_step(model, batch):
        model.eval()
        with torch.no_grad():
            model(batch)
            # accuracy
            acc = (model.labels == model.scores.argmax(dim=1)).float().tolist()
        return acc, model.scores_normalized.tolist()