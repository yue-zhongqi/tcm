from dda_model.backbone import ResNet
from dda_model.util import init_net
from network import AdversarialNetwork
from dda_model.loss import get_alignment_loss, calc_coeff, NMTCritierion
import torch
import torch.nn as nn

class CycleGANDiscriminator():
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.backbone = ResNet(opt.cg_resnet_name)
        self.x_dim = self.backbone.n_features
        self.num_classes = self.opt.cg_num_classes
        self.model_names = ["backbone"]
        self.create_discriminators()
        self.init_networks()
        self.create_optimizer()
        self.iteration = 0

    def create_discriminators(self):
        self.linearA_y = torch.nn.Linear(self.x_dim, self.num_classes)
        self.gvbgA = torch.nn.Linear(self.x_dim, self.num_classes)
        self.model_names.append("linearA_y")
        self.model_names.append("gvbgA")
        self.linearB_y = torch.nn.Linear(self.x_dim, self.num_classes)
        self.gvbgB = torch.nn.Linear(self.x_dim, self.num_classes)
        self.model_names.append("linearB_y")
        self.model_names.append("gvbgB")
        if self.opt.cg_align_feature:
            self.fA_discriminator = AdversarialNetwork(self.x_dim, 1024, 7)
            self.fB_discriminator = AdversarialNetwork(self.x_dim, 1024, 7)
            self.model_names.append("fA_discriminator")
            self.model_names.append("fB_discriminator")
        if self.opt.cg_align_logits:
            self.lA_discriminator = AdversarialNetwork(self.num_classes, 1024, 700.0 * 200)
            self.lB_discriminator = AdversarialNetwork(self.num_classes, 1024, 700.0 * 200)
            self.model_names.append("lA_discriminator")
            self.model_names.append("lB_discriminator")

    def init_networks(self):
        for name in self.model_names:
            if isinstance(name, str):
                parallel = (name == "backbone")
                init_weight = (name != "backbone")
                net = init_net(getattr(self, name), "kaiming", 0.02, self.gpu_ids, parallel, init_weight)
                setattr(self, name, net)
        # Backbone not updating
        for param in self.backbone.parameters():
            param.requires_grad = False

    def create_optimizer(self):
        params = list(self.fA_discriminator.parameters()) + list(self.fB_discriminator.parameters())
        self.optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.001, nesterov=False)

    def set_input(self, source_imgs, source_labels, s2t_mappings, target_imgs, t2s_mappings):
        if source_imgs is not None:
            # During testing, there may be no source images
            self.source_imgs = source_imgs.to(self.device)
        else:
            self.source_imgs = None
        if source_labels is not None:
            # During testing, there may be no source labels
            self.source_labels = source_labels.to(self.device)
        else:
            self.source_labels = None
        if s2t_mappings is not None:
            # During testing, there may be no source cross-domain mappings
            self.s2t_mappings = s2t_mappings.to(self.device)
        else:
            self.s2t_mappings = None
        self.target_imgs = target_imgs.to(self.device)
        self.t2s_mappings = t2s_mappings.to(self.device)

    def get_alignment_inputs(self, x_s, x_t, linear_net, bridge_net):
        x = torch.cat((x_s, x_t), dim=0)
        logits = linear_net(x)
        bridge = bridge_net(x)
        if self.opt.cg_gvbg_weight > 0:
            logits = logits - bridge
        softmax_logits = nn.Softmax(dim=1)(logits)
        return x, softmax_logits, bridge

    def optimize(self):
        self.optimizer.zero_grad()
        x_s = self.backbone(self.source_imgs)
        x_t = self.backbone(self.target_imgs)
        x_s2t = self.backbone(self.s2t_mappings)
        x_t2s = self.backbone(self.t2s_mappings)
        A_x, A_softmax_logits, A_bridge = self.get_alignment_inputs(x_s2t, x_t, self.linearA_y, self.gvbgA)
        B_x, B_softmax_logits, B_bridge = self.get_alignment_inputs(x_t2s, x_s, self.linearB_y, self.gvbgB)
        coeff = calc_coeff(self.iteration, max_iter=10000.0)
        
        loss_transfer = 0
        loss_gvbg = 0
        loss_gvbd = 0
        if self.opt.cg_align_feature:
            Alt, Ame, Ald, Alg = get_alignment_loss(A_x, A_bridge, self.fA_discriminator, coeff,
                                                    self.opt.cg_gvbd_weight, A_softmax_logits, use_logits=False, no_entropy=True)
            Blt, Bme, Bld, Blg = get_alignment_loss(B_x, B_bridge, self.fB_discriminator, coeff,
                                                    self.opt.cg_gvbd_weight, B_softmax_logits, use_logits=False, no_entropy=True)
            loss_transfer += (Alt + Blt)
            loss_gvbg += 0
            loss_gvbd += (Ald + Bld)
        '''
        # Have problems now: No labels
        if self.opt.cg_align_logits:
            Alt, Ame, Ald, Alg = get_alignment_loss(A_softmax_logits, A_bridge, self.lA_discriminator, coeff,
                                                    self.opt.cg_gvbd_weight, A_softmax_logits, use_logits=True)
            Blt, Bme, Bld, Blg = get_alignment_loss(B_softmax_logits, B_bridge, self.lB_discriminator, coeff,
                                                    self.opt.cg_gvbd_weight, B_softmax_logits, use_logits=True)
            loss_transfer += (Alt + Blt)
            loss_gvbg += (Alg + Blg)
            loss_gvbd += (Ald + Bld)
        '''
        self.loss_transfer = loss_transfer
        self.loss_gvbg = loss_gvbg
        self.loss_gvbd = loss_gvbd
        self.loss = 1.0 * loss_transfer + self.opt.cg_gvbd_weight * loss_gvbd + self.opt.cg_gvbg_weight * loss_gvbg
        self.loss.backward(retain_graph=True)
        self.optimizer.step()
        self.iteration += 1