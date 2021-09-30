# coding=utf-8
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import paddle
import paddle.nn as nn
import numpy as np

from paddle.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2D, LayerNorm
import paddle.nn.initializer as init
from scipy import ndimage
import sys

sys.path.append('../')
import configs

xavier_uniform_ = init.XavierUniform()
normal_ = init.Normal(std=1e-6)
zeros_ = init.Constant(value=0.0)

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2pd(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return paddle.to_tensor(weights, stop_gradient=False)


def swish(x):
    return x * paddle.sigmoid(x)


ACT2FN = {"gelu": paddle.nn.functional.gelu, "relu": paddle.nn.functional.relu, "swish": swish}


class Attention(nn.Layer):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        # In fact, here self.all_head_size==config.hidden_size
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(axis=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = paddle.reshape(x, shape=new_x_shape)
        return paddle.transpose(x, perm=(0, 2, 1, 3))

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = paddle.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size,]
        context_layer = context_layer.reshape(new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Layer):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.fc1.weight)
        xavier_uniform_(self.fc2.weight)
        normal_(self.fc1.bias)
        normal_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def to_2tuple(x):
    if isinstance(x, int):
        return tuple([x] * 2)
    elif isinstance(x, (tuple, list)):
        return tuple(x)
    else:
        raise ValueError("Type of x must be int, tuple or list.")

class Embeddings(nn.Layer):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = to_2tuple(img_size)

        # used for our tasks
        patch_size = to_2tuple(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # make x (bs, 3, h, w) be x (bs, config.hs, h//ps, w//ps)
        self.patch_embeddings = Conv2D(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        # shape (1, n_patchs+1, config.hs), +1 for cls token
        self.position_embeddings = paddle.create_parameter(shape=(1, n_patches+1, config.hidden_size), default_initializer=init.Constant(value=0.0), dtype=paddle.float32)
        self.cls_token = paddle.create_parameter(shape=(1, 1, config.hidden_size), default_initializer=init.Constant(value=0.0), dtype=paddle.float32)

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand((B, -1, -1))

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose((0, 2, 1))
        x = paddle.concat((cls_tokens, x), axis=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Layer):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, epsilon=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, epsilon=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    # 加载预训练权重
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with paddle.no_grad():
            # 因为pytorch和paddle的Linear层本就存在转置关系，故这里就相对官方实现去掉t()
            query_weight = np2pd(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).reshape((self.hidden_size, self.hidden_size))
            key_weight = np2pd(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).reshape((self.hidden_size, self.hidden_size))
            value_weight = np2pd(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).reshape((self.hidden_size, self.hidden_size))
            out_weight = np2pd(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).reshape((self.hidden_size, self.hidden_size))

            query_bias = np2pd(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).flatten(0)
            key_bias = np2pd(weights[pjoin(ROOT, ATTENTION_K, "bias")]).flatten(0)
            value_bias = np2pd(weights[pjoin(ROOT, ATTENTION_V, "bias")]).flatten(0)
            out_bias = np2pd(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).flatten(0)

            self.attn.query.weight.set_value(query_weight)
            self.attn.key.weight.set_value(key_weight)
            self.attn.value.weight.set_value(value_weight)
            self.attn.out.weight.set_value(out_weight)
            self.attn.query.bias.set_value(query_bias)
            self.attn.key.bias.set_value(key_bias)
            self.attn.value.bias.set_value(value_bias)
            self.attn.out.bias.set_value(out_bias)

            mlp_weight_0 = np2pd(weights[pjoin(ROOT, FC_0, "kernel")])
            mlp_weight_1 = np2pd(weights[pjoin(ROOT, FC_1, "kernel")])
            mlp_bias_0 = np2pd(weights[pjoin(ROOT, FC_0, "bias")])
            mlp_bias_1 = np2pd(weights[pjoin(ROOT, FC_1, "bias")])

            self.ffn.fc1.weight.set_value(mlp_weight_0)
            self.ffn.fc2.weight.set_value(mlp_weight_1)
            self.ffn.fc1.bias.set_value(mlp_bias_0)
            self.ffn.fc2.bias.set_value(mlp_bias_1)

            self.attention_norm.weight.set_value(np2pd(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.set_value(np2pd(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.set_value(np2pd(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.set_value(np2pd(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Layer):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.LayerList()
        self.encoder_norm = LayerNorm(config.hidden_size, epsilon=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            # must deepcopy, otherwise if layer is changed, self.layer is also changed
            self.layer.append(copy.deepcopy(layer)) 

    def forward(self, hidden_states):
        attn_weights = []
        hidden_state=[]
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            hidden_state.append(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        hidden_state.pop(-1)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights,hidden_state


class Transformer(nn.Layer):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights,hidden_state = self.encoder(embedding_output)
        return encoded, attn_weights,hidden_state


class VisionTransformer(nn.Layer):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,alpha=0.3):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)
        self.alpha=alpha
    def forward(self, x, labels=None):
        x1, attn_weights1,hidden_state1 = self.transformer(x)

        # x1[:, 0] gets cls token for classification
        logits = self.head(x1[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape((-1, self.num_classes)), labels.flatten(0))

            x2, attn_weights2, hidden_state2 = self.transformer(x)
            newlogits = self.head(x2[:, 0])
            loss2 = loss_fct(newlogits.reshape((-1, self.num_classes)), labels.flatten(0))
            loss += loss2
            
            
            p = nn.functional.log_softmax(logits.reshape((-1, self.num_classes)), axis=-1)
            p_tec = nn.functional.softmax(logits.reshape((-1, self.num_classes)), axis=-1)
            q = nn.functional.log_softmax(newlogits.reshape((-1, self.num_classes)), axis=-1)
            q_tec = nn.functional.softmax(newlogits.reshape((-1, self.num_classes)), axis=-1)
            kl_loss = nn.functional.kl_div(p, q_tec, reduction='none').sum()
            reverse_kl_loss = nn.functional.kl_div(q, p_tec, reduction='none').sum()

            # 这里与论文公式相比，kl_loss的系数1/2融入了self.alpha(0.6->0.3)，不过这里的kl_loss没有取batch_size平均
            # 这里就先按作者官方实现的来，左右可以看做就是一个超参的调整
            loss += self.alpha * (kl_loss + reverse_kl_loss)

            return loss
        else:
            return logits, attn_weights1

    def load_from(self, weights):
        # 因为pytorch和paddle的Linear层本就存在转置关系，故这里就相对官方实现去掉t()
        with paddle.no_grad():
            if self.zero_head:
                zeros_(self.head.weight)
                zeros_(self.head.bias)
            else:
                self.head.weight.set_value(np2pd(weights["head/kernel"]))
                self.head.bias.set_value(np2pd(weights["head/bias"]))

            self.transformer.embeddings.patch_embeddings.weight.set_value(np2pd(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.set_value(np2pd(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.set_value(np2pd(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.set_value(np2pd(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.set_value(np2pd(weights["Transformer/encoder_norm/bias"]))

            posemb = np2pd(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.shape == posemb_new.shape:
                self.transformer.embeddings.position_embeddings.set_value(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.shape, posemb_new.shape))
                ntok_new = posemb_new.shape[1]

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape((gs_old, gs_old, -1))

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.set_value(np2pd(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)



CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}


if __name__=='__main__':
    x = paddle.rand(shape=(2, 3, 384, 384))
    model = VisionTransformer(configs.get_b16_config(), 384, zero_head=True, num_classes=100, alpha=0.3)
    logits, attn_weights1 = model(x)
    print(logits)
    label = paddle.ones((2,), dtype='int64')
    loss = model(x, label)
    print(loss)