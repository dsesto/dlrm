import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch._ops import ops

from tricks.md_embedding_bag import PrEmbeddingBag
from tricks.qr_embedding_bag import QREmbeddingBag


class DLRM_Net(nn.Module):

    def __init__(
        self,
        embedding_size=None,
        layers_embedding=None,
        layers_mlp_bot=None,
        layers_mlp_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
        loss_weights=None,
    ):
        super(DLRM_Net, self).__init__()

        if (
            (embedding_size is not None)
            and (layers_embedding is not None)
            and (layers_mlp_bot is not None)
            and (layers_mlp_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            self.loss_weights = loss_weights or [1.0, 1.0]
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # create operators
            if ndevices <= 1:
                self.embeddings, w_list = self.create_emb(embedding_size, layers_embedding, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.embeddings_per_sample_weights = nn.ParameterList()
                    for w in w_list:
                        self.embeddings_per_sample_weights.append(Parameter(w))
                else:
                    self.embeddings_per_sample_weights = w_list
            self.mlp_bot = self.create_mlp(layers_mlp_bot, sigmoid_bot)
            self.mlp_top = self.create_mlp(layers_mlp_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(self.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def create_mlp(self, layers, sigmoid_layer):
        # build MLP layer by layer
        mlp = nn.ModuleList()
        for i in range(0, layers.size - 1):
            layer_input_size = layers[i]
            layer_output_size = layers[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(layer_input_size), int(layer_output_size), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (layer_output_size + layer_input_size))  # np.sqrt(1 / layer_output_size) # np.sqrt(1 / layer_input_size)
            W = np.random.normal(mean, std_dev, size=(layer_output_size, layer_input_size)).astype(np.float32)
            std_dev = np.sqrt(1 / layer_output_size)  # np.sqrt(2 / (layer_output_size + 1))
            bt = np.random.normal(mean, std_dev, size=layer_output_size).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            mlp.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                mlp.append(nn.Sigmoid())
            else:
                mlp.append(nn.ReLU())

        # approach 1: use ModuleList
        # return mlp
        # approach 2: use Sequential container to wrap all mlp
        return torch.nn.Sequential(*mlp)

    def create_emb(self, embedding_size, layers_embedding, weighted_pooling=None):
        embeddings = nn.ModuleList()
        embeddings_per_sample_weights = []
        for i in range(0, layers_embedding.size):
            feature_vocab_size = layers_embedding[i]

            # construct embedding operator
            if self.qr_flag and feature_vocab_size > self.qr_threshold:
                EE = QREmbeddingBag(
                    feature_vocab_size,
                    embedding_size,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and feature_vocab_size > self.md_threshold:
                base = max(embedding_size)
                _m = embedding_size[i] if feature_vocab_size > self.md_threshold else base
                EE = PrEmbeddingBag(feature_vocab_size, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / feature_vocab_size), high=np.sqrt(1 / feature_vocab_size), size=(feature_vocab_size, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(feature_vocab_size, embedding_size, mode="sum", sparse=True)
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / feature_vocab_size), b=np.sqrt(1 / feature_vocab_size))
                W = np.random.uniform(
                    low=-np.sqrt(1 / feature_vocab_size), high=np.sqrt(1 / feature_vocab_size), size=(feature_vocab_size, embedding_size)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                embeddings_per_sample_weights.append(None)
            else:
                embeddings_per_sample_weights.append(torch.ones(feature_vocab_size, dtype=torch.float32))
            embeddings.append(EE)
        return embeddings, embeddings_per_sample_weights

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, sparse_features_offsets, sparse_features_indices, embeddings, embeddings_per_sample_weights):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(sparse_features_indices):
            sparse_offset_group_batch = sparse_features_offsets[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = embeddings[k]

            if embeddings_per_sample_weights[k] is not None:
                per_sample_weights = embeddings_per_sample_weights[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = embeddings[k]
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )

                ly.append(V)

        # print(ly)
        return ly

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):

        n = len(self.embeddings)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.embeddings[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.embeddings[k].weight
                )
            else:
                return
        self.embeddings = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, sparse_features_offsets, sparse_features_indices):
        return self.sequential_forward(dense_x, sparse_features_offsets, sparse_features_indices)

    def sequential_forward(self, dense_x, sparse_features_offsets, sparse_features_indices):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.mlp_bot)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(sparse_features_offsets, sparse_features_indices, self.embeddings, self.embeddings_per_sample_weights)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.mlp_top)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z
