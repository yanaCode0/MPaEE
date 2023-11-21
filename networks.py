import copy

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl

from models.vit import VisionTransformer, PatchEmbed, Block,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.convit import ClassAttention
from models.convit import Block as ConBlock


def prompt_triple_class(x, pre_prompt, vit_prompt=None, infer=False, c=0, d=1024, cls_inc=10):
    x_d = x.unsqueeze(1)
    if infer:
        #index_ = torch.range(0, cls_inc-1).long().cuda()
        score = torch.nn.CosineSimilarity(dim=-1)(x_d, pre_prompt)
        score_select = torch.sort(score)
        prompt_chose = pre_prompt.unsqueeze(0).repeat(x.shape[0], 1, 1).gather(1, score_select[1][:,-1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,d))
        prompt_chose = prompt_chose.view(x.size(0), 1, -1)
        return prompt_chose, score_select[0][:, -1], 0, 0, score

    score = torch.nn.CosineSimilarity(dim=-1)(x_d, vit_prompt)
    inter_score = 0
    pre_score=0
    total = score
    cur_task_num = vit_prompt.shape[0]
    if pre_prompt is not None:
        inter_score = torch.nn.CosineSimilarity(dim=-1)(vit_prompt.unsqueeze(dim=1), pre_prompt)
        pre_score = torch.nn.CosineSimilarity(dim=-1)(x_d, pre_prompt)
        total = torch.cat([score, pre_score], 1).view(x.size(0), -1)

    score_select = score.gather(1, c.unsqueeze(-1))
    prompt_chose = vit_prompt.unsqueeze(0).repeat(x.shape[0], 1, 1) * nn.Softmax(-1)(score).unsqueeze(-1)
    prompt_chose = prompt_chose.sum(1)

    prompt_chose = prompt_chose.view(x.size(0), 1, -1)
    return prompt_chose, score_select, pre_score, inter_score, total

def prompt_triple_class2(x, pre_prompt, vit_prompt=None, infer=False, c=0, d=1024, cls_inc=10):
    x_d = x.unsqueeze(1)
    if infer:
        score = torch.nn.CosineSimilarity(dim=-1)(x_d, pre_prompt)
        score_select = torch.sort(score)
        prompt_chose = pre_prompt.unsqueeze(0).repeat(x.shape[0], 1, 1).gather(1, score_select[1][:,-1].unsqueeze(-1).unsqueeze(-1).repeat(1,1,d))
        prompt_chose = prompt_chose.view(x.size(0), 1, -1)
        return prompt_chose, score_select[0][:, -1], 0, 0, score

    score = torch.nn.CosineSimilarity(dim=-1)(x_d, vit_prompt)
    inter_score = 0
    pre_score=0
    total = score
    cur_task_num = vit_prompt.shape[0]
    if pre_prompt is not None:
        inter_score = torch.nn.CosineSimilarity(dim=-1)(vit_prompt.unsqueeze(dim=1), pre_prompt)
        pre_score = torch.nn.CosineSimilarity(dim=-1)(x_d, pre_prompt)
        total = torch.cat([score, pre_score], 1).view(x.size(0), -1)

    score_select = score
    score_select = torch.sort(score)[0][:,-1]
    prompt_chose = vit_prompt.unsqueeze(0).repeat(x.shape[0], 1, 1) * nn.Softmax(-1)(score).unsqueeze(-1)
    prompt_chose = prompt_chose.sum(1)

    prompt_chose = prompt_chose.view(x.size(0), 1, -1)
    return prompt_chose, score_select, pre_score, inter_score, total

class ViT_KPrompts(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn)

    def forward(self, x, instance_tokens=None, returnbeforepool=False, **kwargs):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if instance_tokens is not None:
            instance_tokens = instance_tokens.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = x + self.pos_embed.to(x.dtype)
        if instance_tokens is not None:
            x = torch.cat([x[:,:1,:], instance_tokens, x[:,1:,:]], dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        if returnbeforepool == True:
            return x
        x = self.norm(x)
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        repr_size = None

    model = build_model_with_cfg(
        ViT_KPrompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



class incremental_vitood(pl.LightningModule):
    def __init__(self, num_cls, lr, max_epoch, weight_decay, known_classes, freezep, using_prompt, anchor_energy=-10,
                 lamda=0.1, energy_beta=1):
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

        self.classifiers = nn.Linear(self.image_encoder.embed_dim, self.hparams.num_cls, bias=True)
        self.tabs = ConBlock(dim=self.image_encoder.embed_dim, num_heads=12, mlp_ratio=0.5, qkv_bias=True,
                        qk_scale=None, drop=0.,attn_drop=0., norm_layer=nn.LayerNorm, attention_type=ClassAttention)
        self.task_tokens = copy.deepcopy(self.image_encoder.cls_token)
        self.vitprompt = nn.Linear(self.image_encoder.embed_dim, 100, bias=False)
        self.pre_vitprompt = None

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad_(False)

        if self.hparams.freezep:
            for name, param in self.vitprompt.named_parameters():
                param.requires_grad_(False)

    def forward(self, image):
        if self.hparams.using_prompt:
            image_features = self.image_encoder(image, instance_tokens=self.vitprompt, returnbeforepool=True, )
        else:
            image_features = self.image_encoder(image, returnbeforepool=True)

        B = image_features.shape[0]
        task_token = self.task_tokens.expand(B, -1, -1)
        task_token, attn, v = self.tabs(torch.cat((task_token, image_features), dim=1), mask_heads=None)
        logits = self.classifiers(task_token[:, 0])

        return logits

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.SGD(optparams, momentum=0.9,lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.hparams.max_epoch)
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode='train'):
        _, images, labels = batch
        labels = labels.long()
        labels = labels-self.hparams.known_classes

        if self.hparams.using_prompt:
            image_features = self.image_encoder(images, instance_tokens=self.vitprompt.weight, returnbeforepool=True, )
        else:
            image_features = self.image_encoder(images, returnbeforepool=True)
        B = image_features.shape[0]
        task_token = self.task_tokens.expand(B, -1, -1)
        task_token, attn, v = self.tabs(torch.cat((task_token, image_features), dim=1), mask_heads=None)
        logits = self.classifiers(task_token[:, 0])
        loss = F.cross_entropy(logits, labels)

        #output_div_t = -1.0 * self.hparams.energy_beta * logits
        #output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
        #free_energy = -1.0 * output_logsumexp / self.hparams.energy_beta
        #align_loss = self.hparams.lamda * ((free_energy - self.hparams.anchor_energy) ** 2).mean()
        align_loss = 0

        if self.pre_vitprompt is not None:
            pre_feature = self.image_encoder(images, instance_tokens=self.pre_vitprompt.weight, returnbeforepool=True, )
            kdloss = nn.MSELoss()(pre_feature.detach(), image_features)
        else:
            kdloss = 0

        loss = loss+align_loss+kdloss

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='val')

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        return loss

    def val_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')


class incremental_prompt(pl.LightningModule):
    def __init__(self, num_cls, lr, max_epoch, weight_decay, known_classes, freezep, using_prompt, anchor_energy=-10,
                 lamda=0.1, energy_beta=1, pre_prompt=None, cls_inc=10):
        super().__init__()
        self.save_hyperparameters()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

        self.classifiers = nn.Linear(self.image_encoder.embed_dim, self.hparams.num_cls, bias=True)
        self.tabs = ConBlock(dim=self.image_encoder.embed_dim, num_heads=12, mlp_ratio=0.5, qkv_bias=True,
                        qk_scale=None, drop=0.,attn_drop=0., norm_layer=nn.LayerNorm, attention_type=ClassAttention)
        self.task_tokens = copy.deepcopy(self.image_encoder.cls_token)
        self.vitprompt=nn.Linear(self.image_encoder.embed_dim, 100, bias=False)
        self.vit_prompt = nn.Parameter(torch.FloatTensor(self.hparams.num_cls, self.image_encoder.embed_dim))
        #self.vit_prompt = nn.Parameter(torch.FloatTensor(1, self.image_encoder.embed_dim))
        nn.init.xavier_normal_(self.vit_prompt)
        self.pre_prompt = pre_prompt
        self.pre_vitprompt=None
        self.cls_inc = cls_inc

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad_(False)

        if self.pre_prompt is not None:
            self.pre_prompt.requires_grad_(False)
            #self.pre_prompt = self.pre_prompt.cuda()

        if self.hparams.freezep:
            for name, param in self.vitprompt.named_parameters():
                param.requires_grad_(False)

    def forward(self, image, infer=False, c=None):
        if self.hparams.using_prompt:
            image_features = self.image_encoder(image, instance_tokens=self.vitprompt, returnbeforepool=True, )
        else:
            image_features = self.image_encoder(image, returnbeforepool=True)
        prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features,dim=1), self.pre_prompt, self.vit_prompt, infer=infer, c=c, d=self.image_encoder.embed_dim, cls_inc=self.hparams.num_cls)
        B = image_features.shape[0]
        task_token = self.task_tokens.expand(B, -1, -1)
        task_token, attn, v = self.tabs(torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
        logits = self.classifiers(task_token[:, 0])

        return logits

    def configure_optimizers(self):

        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.SGD(optparams, momentum=0.9,lr=self.hparams.lr,weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.hparams.max_epoch)
        return [optimizer], [scheduler]

    def _calculate_loss(self, batch, mode='train', infer=False):
        _, images, labels = batch
        labels = labels.long()
        labels = labels-self.hparams.known_classes

        if self.pre_prompt is not None:
            self.pre_prompt = self.pre_prompt.cuda()
            #print(self.pre_prompt.requires_grad)

        #images, prompt, s, pre_s, inter_s, total = self.pre_muti(images, mode, infer, labels)
        if self.hparams.using_prompt:
            image_features = self.image_encoder(images, instance_tokens=self.vitprompt.weight, returnbeforepool=True, )
        else:
            image_features = self.image_encoder(images, returnbeforepool=True)
        if mode=='train':
            prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features, dim=1), self.pre_prompt,
                                                                   self.vit_prompt, infer=infer, c=labels,
                                                                   d=self.image_encoder.embed_dim, cls_inc=self.hparams.num_cls)
        else:
            tem = self.vit_prompt
            if self.pre_prompt is not None:
                tem = torch.cat([self.pre_prompt,self.vit_prompt],dim=0)
            prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(image_features, dim=1), tem, infer=infer, c=labels,
                                                                   d=self.image_encoder.embed_dim,cls_inc=self.hparams.num_cls)
        B = image_features.shape[0]
        task_token = self.task_tokens.expand(B, -1, -1)
        task_token, attn, v = self.tabs(torch.cat([task_token, prompt, image_features], dim=1), mask_heads=None)
        #task_token, attn, v = self.tabs(torch.cat([task_token, image_features], dim=1), mask_heads=None)


        logits = self.classifiers(task_token[:, 0])
        loss = F.cross_entropy(logits, labels)


        align_loss = 0

        if self.pre_vitprompt is not None:
            pre_feature = self.image_encoder(images, instance_tokens=self.pre_vitprompt.weight, returnbeforepool=True, )
            kdloss = nn.MSELoss()(pre_feature.detach(), image_features)
        else:
            kdloss = 0

        if self.pre_prompt is not None:
            loss += 0.1 * self.prompt_loss(inter_s, margin=0.5, pn=-1.0)
            #loss += 0.5 * self.prompt_loss(s, margin=(0.7-s.mean()+pre_s.mean(0).max()))
        loss += 0.5 * self.prompt_loss(s, margin=0.7)
        #else:
            #loss += 0.5 * self.prompt_loss(s, margin=0.5)


        loss = loss+align_loss+kdloss

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='val')

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode='train')
        return loss

    def val_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val', infer=True)

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test', infer=True)

    def prompt_loss(self, s, margin=0.6, pn=1.0):
        if isinstance(s,int):
            return 0
        bound = torch.zeros_like(s)
        trip_score = pn * (- s + margin)
        triple_loss = torch.where(trip_score>bound, trip_score, bound)
        return torch.mean(triple_loss)
    
    def pre_train_process(self, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        #pass
        ###print('pre_process')
        sums = torch.zeros_like(self.vit_prompt).cuda()
        count = torch.zeros(self.vit_prompt.shape[0]).cuda()
        with torch.no_grad():
            self.image_encoder.cuda()
            self.image_encoder.eval()
            for _, images, labels in trn_loader:
                labels = labels.long().cuda()
                labels = labels - self.hparams.known_classes
                feat = torch.mean(self.image_encoder(images.cuda(), returnbeforepool=True),dim=1).squeeze(1)
                #feat = torch.mean(self.image_encoder.patch_embed(images.cuda()),dim=1).squeeze(1)

                for i in range(self.vit_prompt.shape[0]):
                    index = torch.nonzero(labels==i).squeeze(dim=1).cuda()
                    count[i] += index.shape[0]
                    if index.shape[0] == 0:
                        tem_sum=torch.zeros(self.vit_prompt.shape[1]).cuda()
                    else:
                        tem_sum = torch.index_select(feat,0,index).sum(dim=0)
                    sums[i,:] += tem_sum

        self.vit_prompt.data = sums / count.unsqueeze(-1)

    def pre_muti(self, x, mode='train', infer=False, labels=None):
        x = self.image_encoder.patch_embed(x)
        x = torch.cat((self.image_encoder.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.image_encoder.pos_embed.to(x.dtype)

        if mode=='train':
            prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(x, dim=1), self.pre_prompt,
                                                                   self.vit_prompt, infer=infer, c=labels,
                                                                   d=self.image_encoder.embed_dim, cls_inc=self.hparams.num_cls)
        else:
            tem = self.vit_prompt
            if self.pre_prompt is not None:
                tem = torch.cat([self.pre_prompt,self.vit_prompt],dim=0)
            prompt, s, pre_s, inter_s, total = prompt_triple_class(torch.mean(x, dim=1), tem, infer=infer, c=labels,
                                                                   d=self.image_encoder.embed_dim,cls_inc=self.hparams.num_cls)
        image_features = torch.cat([x[:,:1,:],prompt, x[:,1:,:]], dim=1)
        return image_features, prompt, s, pre_s, inter_s, total
    

