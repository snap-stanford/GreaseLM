import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import modeling_bert
from transformers import modeling_roberta
from transformers import PretrainedConfig
from transformers.file_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_path,
    hf_bucket_url,
    is_remote_url,
)

from modeling import modeling_gnn
from utils import layers
from utils import utils

logger = logging.getLogger(__name__)


if os.environ.get('INHERIT_BERT', 0):
    ModelClass = modeling_bert.BertModel
else:
    ModelClass = modeling_roberta.RobertaModel
    
print ('ModelClass', ModelClass)


class GreaseLM(nn.Module):

    def __init__(self, args={}, model_name="roberta-large", k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=False, layer_id=-1):
        super().__init__()
        self.lmgnn = LMGNN(args, model_name, k, n_ntype, n_etype,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range, ie_dim=ie_dim, info_exchange=info_exchange, ie_layer_num=ie_layer_num,  sep_ie_layers=sep_ie_layers, layer_id=layer_id)

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        """
        edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        """
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type

    def forward(self, *inputs, cache_output=False, detail=False):
        """
        inputs_ids: (batch_size, num_choice, seq_len)    -> (batch_size * num_choice, seq_len)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        node_scores: [bs, nc, n_node, 1]
        adj_lengths: means the "actual" number of nodes (excluding padding)(batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )

        returns:
        logits: [bs, nc]
        """
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        #Here, merge the batch dimension and the num_choice dimension
        edge_index_orig, edge_type_orig = inputs[-2:]
        _inputs = [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:4]] + [x.reshape(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[4:-2]] + [sum(x,[]) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, edge_index, edge_type = _inputs
        node_scores = torch.zeros_like(node_scores)
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        logits, attn = self.lmgnn(lm_inputs, concept_ids,
                                    node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj,
                                    emb_data=None, cache_output=cache_output)
        # logits: [bs * nc]
        logits = logits.view(bs, nc)
        if not detail:
            return logits, attn
        else:
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig
            # edge_index_orig: list of (batch_size, num_choice). each entry is torch.tensor(2, E)
            # edge_type_orig: list of (batch_size, num_choice). each entry is torch.tensor(E, )

    def get_fake_inputs(self, device="cuda:0"):
        bs = 4
        nc = 5
        seq_len = 100
        input_ids = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, nc, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, nc, seq_len]).to(device)
        output_mask = torch.zeros([bs, nc]).to(device)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(bs, nc, 1).to(device)
        adj_lengths = torch.zeros([bs, nc], dtype=torch.long).fill_(10).to(device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)

        edge_index = [[edge_index] * nc] * bs
        edge_type = [[edge_type] * nc] * bs

        node_type = torch.zeros([bs, nc, n_node], dtype=torch.long).to(device)
        node_type[:, :, 0] = 3
        node_score = torch.zeros([bs, nc, n_node, 1]).to(device)
        node_score[:, :, 1] = 180
        return input_ids, attention_mask, token_type_ids, output_mask, concept_ids, node_type, node_score, adj_lengths, edge_index, edge_type

    def check_outputs(self, logits, attn):
        bs = 4
        nc = 5
        assert logits.size() == (bs, nc)
        n_edges = 3


def test_GreaseLM(device):
    cp_emb = torch.load("data/cpnet/cp_emb.pt")
    model = GreaseLM(pretrained_concept_emb=cp_emb).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(*outputs)


class LMGNN(nn.Module):

    def __init__(self, args={}, model_name="roberta-large", k=5, n_ntype=4, n_etype=38,
                 n_concept=799273, concept_dim=200, concept_in_dim=1024, n_attention_head=2,
                 fc_dim=200, n_fc_layer=0, p_emb=0.2, p_gnn=0.2, p_fc=0.2,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02, ie_dim=200, info_exchange=True, ie_layer_num=1, sep_ie_layers=False, layer_id=-1):
        super().__init__()
        config, _ = ModelClass.config_class.from_pretrained(
            model_name,
            cache_dir=None, return_unused_kwargs=True,
            force_download=False,
            output_hidden_states=True
        )
        self.init_range = init_range

        self.k = k
        self.concept_dim = concept_dim
        self.n_attention_head = n_attention_head
        self.activation = layers.GELU()
        if k >= 0:
            self.concept_emb = layers.CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim, use_contextualized=False, concept_in_dim=concept_in_dim, pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
            self.pooler = layers.MultiheadAttPoolLayer(n_attention_head, config.hidden_size, concept_dim)

        concat_vec_dim = concept_dim * 2 + config.hidden_size
        self.fc = layers.MLP(concat_vec_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

        self.mp, self.loading_info = TextKGMessagePassing.from_pretrained(model_name, output_hidden_states=True, output_loading_info=True, args=args, k=k, n_ntype=n_ntype, n_etype=n_etype, dropout=p_gnn, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers)

        self.layer_id = layer_id
        self.cpnet_vocab_size = n_concept

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs, concept_ids, node_type_ids, node_scores, adj_lengths, special_nodes_mask, adj, emb_data=None, cache_output=False):
        """
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns:
        logits: [bs]
        """
        #LM inputs
        input_ids, attention_mask, token_type_ids, output_mask = inputs

        # GNN inputs
        concept_ids[concept_ids == 0] = self.cpnet_vocab_size + 2
        gnn_input = self.concept_emb(concept_ids - 1, emb_data).to(node_type_ids.device)
        gnn_input[:, 0] = 0
        gnn_input = self.dropout_e(gnn_input) #(batch_size, n_node, dim_node)

        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]

        # Merged core
        outputs, gnn_output = self.mp(input_ids, token_type_ids, attention_mask, output_mask, gnn_input, adj, node_type_ids, node_scores, special_nodes_mask, output_hidden_states=True)
        # outputs: ([bs, seq_len, sent_dim], [bs, sent_dim], ([bs, seq_len, sent_dim] for _ in range(25)))
        # gnn_output: [bs, n_node, dim_node]

        # LM outputs
        all_hidden_states = outputs[-1] # ([bs, seq_len, sent_dim] for _ in range(25))
        hidden_states = all_hidden_states[self.layer_id] # [bs, seq_len, sent_dim]

        sent_vecs = self.mp.pooler(hidden_states) # [bs, sent_dim]

        sent_token_mask = output_mask.clone()
        sent_token_mask[:, 0] = 0

        # GNN outputs
        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) # pool over all KG nodes (excluding the context node)
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)
        # graph_vecs: [bs, node_dim]

        sent_node_mask = special_nodes_mask.clone()
        sent_node_mask[:, 0] = 0

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = torch.cat((graph_vecs, sent_vecs, Z_vecs), 1)
        logits = self.fc(self.dropout_fc(concat))

        return logits, pool_attn

    def get_fake_inputs(self, device="cuda:0"):
        bs = 20
        seq_len = 100
        input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, seq_len]).to(device)

        n_node = 200
        concept_ids = torch.arange(end=n_node).repeat(bs, 1).to(device)
        adj_lengths = torch.zeros([bs], dtype=torch.long).fill_(10).to(device)

        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        adj = (edge_index, edge_type)

        node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        node_type[:, 0] = 3
        node_score = torch.zeros([bs, n_node, 1]).to(device)
        node_score[:, 1] = 180

        return (input_ids, attention_mask, token_type_ids, None), concept_ids, node_type, node_score, adj_lengths, adj

    def check_outputs(self, logits, pool_attn):
        bs = 20
        assert logits.size() == (bs, 1)
        n_edges = 3


def test_LMGNN(device):
    cp_emb = torch.load("data/cpnet/cp_emb.pt")
    model = LMGNN(pretrained_concept_emb=cp_emb).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(*outputs)



class TextKGMessagePassing(ModelClass):

    def __init__(self, config, args={}, k=5, n_ntype=4, n_etype=38, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        super().__init__(config=config)

        self.n_ntype = n_ntype
        self.n_etype = n_etype

        self.hidden_size = concept_dim
        self.emb_node_type = nn.Linear(self.n_ntype, concept_dim // 2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, concept_dim // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, concept_dim // 2)
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(concept_dim // 2, concept_dim // 2)

        self.k = k

        self.Vh = nn.Linear(concept_dim, concept_dim)
        self.Vx = nn.Linear(concept_dim, concept_dim)

        self.activation = layers.GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.encoder = RoBERTaGAT(config, k=k, n_ntype=n_ntype, n_etype=n_etype, hidden_size=concept_dim, dropout=dropout, concept_dim=concept_dim, ie_dim=ie_dim, p_fc=p_fc, info_exchange=info_exchange, ie_layer_num=ie_layer_num, sep_ie_layers=sep_ie_layers)

        self.sent_dim = config.hidden_size

    def forward(self, input_ids, token_type_ids, attention_mask, special_tokens_mask, H, A, node_type, node_score, special_nodes_mask, cache_output=False, position_ids=None, head_mask=None, output_hidden_states=True):
        """
        input_ids: [bs, seq_len]
        token_type_ids: [bs, seq_len]
        attention_mask: [bs, seq_len]
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
            edge_index: [2, n_edges]
            edge_type: [n_edges]
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        # LM inputs
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 1D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if len(attention_mask.size()) == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif len(attention_mask.size()) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError("Attnetion mask should be either 1D or 2D.")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        # GNN inputs
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = modeling_gnn.make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        # Merged core
        encoder_outputs, _X = self.encoder(embedding_output,
                                       extended_attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, special_nodes_mask, output_hidden_states=output_hidden_states)

        # LM outputs
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here

        # GNN outputs
        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return outputs, output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()`` (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the `shortcut name` of a pre-trained model to load from cache or download, e.g.: ``bert-base-uncased``.
              - a string with the `identifier name` of a pre-trained model that was user-uploaded to our S3, e.g.: ``dbmdz/bert-base-german-cased``.
              - a path to a `directory` containing model weights saved using :func:`~transformers.PreTrainedModel.save_pretrained`, e.g.: ``./my_model_directory/``.
              - a path or url to a `tensorflow index checkpoint file` (e.g. `./tf_model/model.ckpt.index`). In this case, ``from_tf`` should be set to True and a configuration object should be provided as ``config`` argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary (resp. with keyword arguments ``config`` and ``state_dict``)

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's ``__init__`` method

            config: (`optional`) one of:
                - an instance of a class derived from :class:`~transformers.PretrainedConfig`, or
                - a string valid as input to :func:`~transformers.PretrainedConfig.from_pretrained()`

                Configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the ``shortcut-name`` string of a pretrained model), or
                    - the model was saved using :func:`~transformers.PreTrainedModel.save_pretrained` and is reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as ``pretrained_model_name_or_path`` and a configuration JSON file named `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from saved weights file.
                This option can be used if you want to create a model from a pretrained configuration but load your own weights.
                In this case though, you should check if using :func:`~transformers.PreTrainedModel.save_pretrained` and :func:`~transformers.PreTrainedModel.from_pretrained` is not a simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model. (e.g. ``output_attention=True``). Behave differently depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwargs`` will be directly passed to the underlying model's ``__init__`` method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the configuration class initialization function (:func:`~transformers.PretrainedConfig.from_pretrained`). Each key of ``kwargs`` that corresponds to a configuration attribute will be used to override said attribute with the supplied ``kwargs`` value. Remaining keys that do not correspond to any configuration attribute will be passed to the underlying model's ``__init__`` function.

        Examples::

            # For example purposes. Not runnable.
            model = BertModel.from_pretrained('bert-base-uncased')    # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)  # Update configuration during loading
            assert model.config.output_attention == True
            # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            config = BertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            model = BertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        use_cdn = kwargs.pop("use_cdn", True)

        k = kwargs["k"]

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or `from_tf` set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            all_keys = list(state_dict.keys())

            # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]

                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                    f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
                    f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
                    f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
                    f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} "
                    f"and are newly initialized: {missing_keys}\n"
                    f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
                )
            else:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\n"
                    f"If your task is similar to the task the model of the ckeckpoint was trained on, "
                    f"you can already use {model.__class__.__name__} for predictions without further training."
                )
            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        model.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
                "all_keys": all_keys,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device:
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model

    def get_fake_inputs(self, device="cuda:0"):
        bs = 20
        seq_len = 100
        input_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        token_type_ids = torch.zeros([bs, seq_len], dtype=torch.long).to(device)
        attention_mask = torch.ones([bs, seq_len]).to(device)

        n_node = 200
        H = torch.zeros([bs, n_node, self.hidden_size]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        A = (edge_index, edge_type)

        node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        node_type[:, 0] = 3
        node_score = torch.zeros([bs, n_node, 1]).to(device)
        node_score[:, 1] = 180
        return input_ids, token_type_ids, attention_mask, H, A,  node_type, node_score

    def check_outputs(self, outputs, gnn_output):
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert gnn_output.size() == (bs, n_node, self.hidden_size)


def test_TextKGMessagePassing(device):
    model = TextKGMessagePassing.from_pretrained("roberta-large", output_hidden_states=True).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(*outputs)


class RoBERTaGAT(modeling_bert.BertEncoder):

    def __init__(self, config, k=5, n_ntype=4, n_etype=38, hidden_size=200, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        super().__init__(config)

        self.k = k
        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
        self.gnn_layers = nn.ModuleList([modeling_gnn.GATConvE(hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])
        self.activation = layers.GELU()
        self.dropout_rate = dropout

        self.sent_dim = config.hidden_size
        self.sep_ie_layers = sep_ie_layers
        if sep_ie_layers:
            self.ie_layers = nn.ModuleList([layers.MLP(self.sent_dim + concept_dim, ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc) for _ in range(k)])
        else:
            self.ie_layer = layers.MLP(self.sent_dim + concept_dim, ie_dim, self.sent_dim + concept_dim, ie_layer_num, p_fc)

        self.concept_dim = concept_dim
        self.num_hidden_layers = config.num_hidden_layers
        self.info_exchange = info_exchange

    def forward(self, hidden_states, attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, special_nodes_mask, output_attentions=False, output_hidden_states=True):
        """
        hidden_states: [bs, seq_len, sent_dim]
        attention_mask: [bs, 1, 1, seq_len]
        head_mask: list of shape [num_hidden_layers]

        _X: [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        edge_index: [2, n_edges]
        edge_type: [n_edges]
        _node_type: [bs * n_nodes]
        _node_feature_extra: [bs * n_nodes, node_dim]
        """
        bs = hidden_states.size(0)
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            # LM
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            if i >= self.num_hidden_layers - self.k:
                # GNN
                gnn_layer_index = i - self.num_hidden_layers + self.k
                _X = self.gnn_layers[gnn_layer_index](_X, edge_index, edge_type, _node_type, _node_feature_extra)
                _X = self.activation(_X)
                _X = F.dropout(_X, self.dropout_rate, training = self.training)

                # Exchange info between LM and GNN hidden states (Modality interaction)
                if self.info_exchange == True or (self.info_exchange == "every-other-layer" and (i - self.num_hidden_layers + self.k) % 2 == 0):
                    X = _X.view(bs, -1, _X.size(1)) # [bs, max_num_nodes, node_dim]
                    context_node_lm_feats = hidden_states[:, 0, :] # [bs, sent_dim]
                    context_node_gnn_feats = X[:, 0, :] # [bs, node_dim]
                    context_node_feats = torch.cat([context_node_lm_feats, context_node_gnn_feats], dim=1)
                    if self.sep_ie_layers:
                        context_node_feats = self.ie_layers[gnn_layer_index](context_node_feats)
                    else:
                        context_node_feats = self.ie_layer(context_node_feats)
                    context_node_lm_feats, context_node_gnn_feats = torch.split(context_node_feats, [context_node_lm_feats.size(1), context_node_gnn_feats.size(1)], dim=1)
                    hidden_states[:, 0, :] = context_node_lm_feats
                    X[:, 0, :] = context_node_gnn_feats
                    _X = X.view_as(_X)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, _X # last-layer hidden state, (all hidden states), (all attentions)

    def get_fake_inputs(self, device="cuda:0"):
        bs = 20
        seq_len = 100
        hidden_states = torch.zeros([bs, seq_len, self.sent_dim]).to(device)
        attention_mask = torch.zeros([bs, 1, 1, seq_len]).to(device)
        head_mask = [None] * self.num_hidden_layers

        n_node = 200
        _X = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        n_edges = 3
        edge_index = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(device)
        edge_type = torch.zeros(n_edges, dtype=torch.long).fill_(2).to(device)
        _node_type = torch.zeros([bs, n_node], dtype=torch.long).to(device)
        _node_type[:, 0] = 3
        _node_type = _node_type.view(-1)
        _node_feature_extra = torch.zeros([bs * n_node, self.concept_dim]).to(device)
        return hidden_states, attention_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra

    def check_outputs(self, outputs, _X):
        bs = 20
        seq_len = 100
        assert outputs[0].size() == (bs, seq_len, self.sent_dim)
        n_node = 200
        assert _X.size() == (bs * n_node, self.concept_dim)


def test_RoBERTaGAT(device):
    config, _ = modeling_roberta.RobertaModel.config_class.from_pretrained(
        "roberta-large",
        cache_dir=None, return_unused_kwargs=True,
        force_download=False,
        output_hidden_states=True
    )
    model = RoBERTaGAT(config, sep_ie_layers=True).to(device)
    inputs = model.get_fake_inputs(device)
    outputs = model(*inputs)
    model.check_outputs(*outputs)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    utils.print_cuda_info()
    free_gpus = utils.select_free_gpus()
    device = torch.device("cuda:{}".format(free_gpus[0]))

    # test_RoBERTaGAT(device)

    # test_TextKGMessagePassing(device)

    # test_LMGNN(device)

    test_GreaseLM(device)
