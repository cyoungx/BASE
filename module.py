import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class EDL_loss(nn.Module):
    def __init__(self, arg=None):
        super(EDL_loss, self).__init__()
        self.arg = arg

    def evidential_deep_learning_loss(self, outputs, label):
        one_hot_label = torch.eye(self.arg.num_class).to(outputs.device)
        one_hot_label = one_hot_label[label]

        evidence = F.softplus(outputs)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        loss_edl = torch.sum(one_hot_label * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True).mean() # [B, 1]

        return loss_edl

    def stable_evidential_deep_learning_loss(self, outputs, label):
        one_hot_label = torch.eye(self.arg.num_class).to(outputs.device)[label]
        M = outputs.max(dim=1, keepdim=True).values
        shift_logits = outputs - M
        exp_shift_logits = torch.exp(shift_logits)
        exp_neg_M = torch.exp(-M)
        sum_exp_shift_logits = torch.sum(exp_shift_logits, dim=1, keepdim=True)
        
        log_S_inner = sum_exp_shift_logits + self.arg.num_class * 1 * exp_neg_M
        log_S_stable = M + torch.log(log_S_inner)
        
        log_alpha_inner = exp_shift_logits + 1 * exp_neg_M
        log_alpha_stable = M + torch.log(log_alpha_inner)
        
        log_S_term = log_S_stable.expand(-1, self.arg.num_class) 
        loss_edl = torch.sum(one_hot_label * (log_S_term - log_alpha_stable), dim=1).mean()
        
        return loss_edl

    def get_edl_info(self, logits):
        if logits is None:
            return torch.tensor([1.0])
        evidence = F.softplus(logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        u = self.arg.num_class / S
        edl_info_dict = {'evidence':evidence, 'alpha':alpha, 'S':S, 'u':u}
        return edl_info_dict

    def adaptive_evidence_smoothing(self, alpha, label):
        one_hot_label = torch.eye(self.arg.num_class).to(label.device)[label]
        alpha_y = torch.sum(alpha * one_hot_label, dim=1, keepdim=True)
        c_y_value = 1.0 + 1.0 / (alpha_y + 1e-9)
        c_y_term = c_y_value * one_hot_label
        c_non_target_term = 1.0 * (1.0 - one_hot_label)
        c = c_y_term + c_non_target_term
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_c = torch.sum(c, dim=1, keepdim=True)
        
        lnB_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_c = torch.sum(torch.lgamma(c), dim=1, keepdim=True) - torch.lgamma(S_c)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        loss_ddkl = torch.sum((alpha - c) * (dg1 - dg0), dim=1, keepdim=True) + lnB_alpha - lnB_c
        return loss_ddkl

    def forward(self, outputs_list, label):
        output_j, output_b, output_v = outputs_list[0], outputs_list[1], outputs_list[2]

        loss_edl_j = self.stable_evidential_deep_learning_loss(output_j, label)
        loss_edl_b = self.stable_evidential_deep_learning_loss(output_b, label)
        loss_edl_v = self.stable_evidential_deep_learning_loss(output_v, label)
        loss_edl = loss_edl_j + loss_edl_b + loss_edl_v

        edl_info_dict_j, edl_info_dict_b, edl_info_dict_v = self.get_edl_info(output_j), self.get_edl_info(output_b), self.get_edl_info(output_v)
        edl_info_dict_list = [edl_info_dict_j, edl_info_dict_b, edl_info_dict_v]
        
        if self.arg.flag_loss_aes:
            alpha_list = [edl_info_dict_j['alpha'], edl_info_dict_b['alpha'], edl_info_dict_v['alpha']]
            loss_aes_j = self.adaptive_evidence_smoothing(alpha_list[0], label)
            loss_aes_b = self.adaptive_evidence_smoothing(alpha_list[1], label)
            loss_aes_v = self.adaptive_evidence_smoothing(alpha_list[2], label)
            loss_aes = (loss_aes_j.mean() + loss_aes_b.mean() + loss_aes_v.mean()) / 3
        else:
            loss_aes = torch.tensor(0).cuda()

        info_dict = {'loss_edl': loss_edl,
                     'loss_aes':  loss_aes,
                     'edl_info_dict_list': edl_info_dict_list}
        return info_dict

class Text_Projection(nn.Module):
    def __init__(self, arg=None):
        super().__init__()
        self.arg = arg
        in_channels=768
        out_channels=self.arg.dim
        self.text_projection = nn.Sequential(
            nn.Linear(in_channels, (in_channels + out_channels) // 2),
            nn.GELU(),
            nn.Linear((in_channels + out_channels) // 2, out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, text_feature):
        feature = self.text_projection(text_feature)
        feature = F.normalize(feature, p=2, dim=-1)
        return feature

class LHSG(nn.Module):
    def __init__(self, arg, class_atom_features=None):
        super(LHSG, self).__init__()
        self.arg = arg
        self.D = self.arg.dim
        self.K = 6
        self.d = 64
        self.momentum = 0.9 
        self.num_prototypes = self.arg.num_proto

        self.class_atom_features = nn.Parameter(class_atom_features.clone())
        self.register_buffer("atom_action_init", class_atom_features.detach().clone())

        self.register_buffer("class_prototypes", torch.zeros(self.arg.num_class, self.num_prototypes, self.D))
        self.register_buffer("init_mask", torch.zeros(self.arg.num_class, self.num_prototypes, dtype=torch.bool))

        self.semantic_miner = nn.Sequential(
            nn.Linear(self.D * 2, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.D)
        )

        self.shared_down = nn.Sequential(nn.Linear(self.D, self.d), nn.GELU())
        self.part_generators = nn.ModuleList([nn.Linear(self.d, self.D) for _ in range(self.K)])

    def forward(self, ske_feat, text_feat, label):
        B = ske_feat.size(0)
        device = ske_feat.device
        
        with torch.no_grad():
            norm_ske_feat = F.normalize(ske_feat, dim=-1)
            current_class_protos = self.class_prototypes[label]
            norm_protos = F.normalize(current_class_protos, dim=-1)
            
            sims = torch.einsum('bd, bmd->bm', norm_ske_feat, norm_protos)
            best_proto_idx = torch.argmax(sims, dim=1)

            flat_indices = label * self.num_prototypes + best_proto_idx
            num_total_protos = self.arg.num_class * self.num_prototypes
            
            one_hot = F.one_hot(flat_indices, num_classes=num_total_protos).float() # [B, N]
            count = one_hot.sum(dim=0)
            sum_feat = torch.matmul(one_hot.t(), ske_feat)
            update_mask = count > 0
            valid_indices = torch.nonzero(update_mask).squeeze()
            
            if valid_indices.numel() > 0:
                avg_feat = sum_feat[update_mask] / count[update_mask].unsqueeze(1)
                flat_protos = self.class_prototypes.view(-1, self.D)
                flat_init_mask = self.init_mask.view(-1)
                
                first_time_mask = update_mask & (~flat_init_mask)
                ema_update_mask = update_mask & flat_init_mask
                
                if first_time_mask.any():
                    flat_protos[first_time_mask] = sum_feat[first_time_mask] / count[first_time_mask].unsqueeze(1)
                    flat_init_mask[first_time_mask] = True
                
                if ema_update_mask.any():
                    target_avg_feat = sum_feat[ema_update_mask] / count[ema_update_mask].unsqueeze(1)
                    flat_protos[ema_update_mask] = self.momentum * flat_protos[ema_update_mask] + \
                                                 (1 - self.momentum) * target_avg_feat

        base_proto = self.class_prototypes[label, best_proto_idx]

        f_text = text_feat[label] 
        combined_seed = torch.cat([F.normalize(f_text, dim=-1), F.normalize(base_proto, dim=-1)], dim=-1)
        semantic_compensation = self.semantic_miner(combined_seed)
        t_c_hat = base_proto + semantic_compensation 

        mid_feat = self.shared_down(t_c_hat)
        deltas = [gen(mid_feat) for gen in self.part_generators]
        current_class_atoms = self.class_atom_features[label] 
        parts_tensor = current_class_atoms + torch.stack(deltas, dim=1)

        norm_class_atoms = F.normalize(current_class_atoms, p=2, dim=-1)
        spatial_scores = torch.bmm(ske_feat.unsqueeze(1), norm_class_atoms.transpose(1, 2))
        w_guide = F.softmax(spatial_scores.squeeze(1), dim=-1).unsqueeze(2)
        P_s = torch.sum(w_guide * parts_tensor, dim=1)
        P_s_norm = F.normalize(P_s, p=2, dim=1)
        ske_feat_enhanced = F.normalize(ske_feat + self.arg.lambda_a * P_s_norm, p=2, dim=1)
        
        if self.arg.flag_loss_enh:
            logits_align = torch.matmul(ske_feat_enhanced, F.normalize(text_feat, dim=-1).t())
            loss_enh = F.cross_entropy(logits_align / 0.1, label) 
        else:
            loss_enh = torch.tensor(0., device=device)

        info_dict = {
            'ske_feat_enhanced': ske_feat_enhanced, 
            'prototype': self.class_prototypes,
            'loss_enh': loss_enh}

        return info_dict

class BPOM(nn.Module):
    def __init__(self, arg, text_feat):
        super(BPOM, self).__init__()
        self.arg = arg
        self.D = self.arg.dim

        self.register_buffer('text_feat', text_feat.detach().clone())
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.LayerNorm(self.D),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def generate_semantic_noise(self, batch_size):
        C, P, D = self.text_feat.shape
        device = self.text_feat.device
        
        rand_class_idx = torch.randint(0, C, (batch_size, P), device=device)
        flat_text = self.text_feat.view(C * P, D)
        part_offsets = torch.arange(P, device=device).unsqueeze(0) 
        indices = rand_class_idx * P + part_offsets

        mixed_parts = flat_text[indices.view(-1)].view(batch_size, P, D)
        semantic_noise = torch.mean(mixed_parts, dim=1) 
        semantic_noise = self.text_projection(semantic_noise)

        return semantic_noise

    def forward(self, ske_feat, label, layer, prototype):
        B = ske_feat.size(0)
        C, M, D = prototype.shape
        device = ske_feat.device
        prototype = prototype.detach()
        semantic_noise = self.generate_semantic_noise(B)

        k_neighbors = self.arg.kn
        flat_protos = prototype.view(C * M, D)
        dist_matrix = torch.cdist(F.normalize(semantic_noise, p=2, dim=1), flat_protos, p=2)
        topk_dist, topk_idx = torch.topk(dist_matrix, k=k_neighbors, dim=1, largest=False)
        repulsion_weights = F.softmax(-topk_dist / 0.1, dim=1) # [B, K]
        
        target_center = torch.zeros_like(semantic_noise)
        for i in range(k_neighbors):
            target_proto = flat_protos[topk_idx[:, i]]
            target_center += repulsion_weights[:, i].unsqueeze(1) * target_proto

        d_min = topk_dist[:, 0:1]
        gamma_base = self.arg.gamma_base
        alpha = gamma_base * torch.exp(-d_min) 
        shift_vector = (target_center - semantic_noise) * alpha
        pood_feat_temp = semantic_noise + shift_vector
        pood_feat_norm = F.normalize(pood_feat_temp, p=2, dim=1)

        local_density = torch.mean(topk_dist, dim=1, keepdim=True)
        eta_base = self.arg.eta_base
        adaptive_margin = local_density * eta_base

        dist_pood_all = torch.cdist(pood_feat_norm, flat_protos, p=2)
        min_dist_val, min_idx = torch.min(dist_pood_all, dim=1, keepdim=True)
        
        violation = (adaptive_margin - min_dist_val).clamp(min=0)
        nearest_proto = flat_protos[min_idx.squeeze()]
        repulsion_dir = F.normalize(pood_feat_norm - nearest_proto, p=2, dim=1)
        
        pood_feat = pood_feat_temp + repulsion_dir * violation
        pood_feat_norm = F.normalize(pood_feat, p=2, dim=1)
        
        for param in layer.parameters(): param.requires_grad = False
        pood_logits = layer(pood_feat_norm)
        for param in layer.parameters(): param.requires_grad = True

        pood_alpha = F.softplus(pood_logits) + 1
        pood_S = torch.sum(pood_alpha, dim=1, keepdim=True)
        pood_u = self.arg.num_class / pood_S
        
        return {
            'pood_feat': pood_feat,
            'pood_alpha': pood_alpha,
            'pood_u': pood_u,
        }

class CrossModalInteraction(nn.Module):
    def __init__(self, arg, text_model, label_mapping=None):
        super().__init__()
        self.arg = arg
        self.output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        feature_dim = self.arg.dim
        
        if self.arg.flag_cmi_lhsg:
            class_atom_features = torch.Tensor(np.load(self.arg.text_class_atom_path)).cuda()
            class_atom_features = text_model(class_atom_features)
            idx = torch.tensor(label_mapping, dtype=torch.long)
            class_atom_features = class_atom_features[idx]
            self.lhsg_j = LHSG(self.arg, class_atom_features).cuda(self.output_device)
            self.lhsg_b = LHSG(self.arg, class_atom_features).cuda(self.output_device)
            self.lhsg_v = LHSG(self.arg, class_atom_features).cuda(self.output_device)

        if self.arg.flag_cmi_bpom:
            class_atom_features = torch.Tensor(np.load(self.arg.text_class_atom_path)).cuda()
            class_atom_features = text_model(class_atom_features)
            idx = torch.tensor(label_mapping, dtype=torch.long)
            class_atom_features = class_atom_features[idx]
            self.bpom_j = BPOM(self.arg, class_atom_features).cuda(self.output_device)
            self.bpom_b = BPOM(self.arg, class_atom_features).cuda(self.output_device)
            self.bpom_v = BPOM(self.arg, class_atom_features).cuda(self.output_device)
    
    def bcl(self, ske_feat_list, label, enh_ske_feat_list, pood_feat_list, id_u_list, prototypes_list):
        feat_j, feat_b, feat_v = ske_feat_list[0], ske_feat_list[1], ske_feat_list[2]
        enh_feat_j, enh_feat_b, enh_feat_v = enh_ske_feat_list[0], enh_ske_feat_list[1], enh_ske_feat_list[2]
        pood_feat_j, pood_feat_b, pood_feat_v = pood_feat_list[0], pood_feat_list[1], pood_feat_list[2]
        
        sample_weights = []
        with torch.no_grad():
            for i in range(3):
                curr_feat = ske_feat_list[i]
                curr_u = id_u_list[i].view(-1, 1) 
                curr_proto = prototypes_list[i]
                
                flat_protos = curr_proto.view(-1, self.arg.dim)
                dist_to_all = torch.cdist(curr_feat, flat_protos, p=2)

                class_mask = torch.zeros(label.size(0), self.arg.num_class, device=curr_feat.device)
                class_mask.scatter_(1, label.unsqueeze(1), 1.0)
                proto_mask = class_mask.repeat_interleave(self.arg.num_proto, dim=1).bool()

                d_intra, _ = torch.min(dist_to_all.masked_fill(~proto_mask, 1e6), dim=1, keepdim=True)
                d_inter, _ = torch.min(dist_to_all.masked_fill(proto_mask, 1e6), dim=1, keepdim=True)
                
                b_score = torch.sigmoid(d_intra / (d_inter + 1e-6)) * curr_u
                sample_weights.append(1.0 + b_score.detach())

            cat_sample_weight = torch.cat(sample_weights, dim=0)

        cat_id_supcon_feat = torch.cat([feat_j, feat_b, feat_v], dim=0)
        cat_id_supcon_label = torch.cat([label, label, label], dim=0)
        
        # ID
        sim_id_supcon = torch.mm(cat_id_supcon_feat, cat_id_supcon_feat.T) / self.arg.temp_id
        sim_id_supcon_mask = torch.ones_like(sim_id_supcon).bool()
        sim_id_supcon_mask.fill_diagonal_(False)
        neg_inf_approx = torch.finfo(sim_id_supcon.dtype).min
        sim_id_supcon = sim_id_supcon.masked_fill(~sim_id_supcon_mask, neg_inf_approx)
        sim_id_supcon_pos_mask = cat_id_supcon_label.unsqueeze(1) == cat_id_supcon_label.unsqueeze(0)
        sim_id_supcon_pos_mask.fill_diagonal_(False)
        
        # ID_eID
        cat_id_selfcon_feat = torch.cat([enh_feat_j, enh_feat_b, enh_feat_v], dim=0)
        sim_id_selfcon = torch.mm(cat_id_supcon_feat, cat_id_selfcon_feat.T) / self.arg.temp_eid
        sim_id_selfcon_pos_mask = torch.eye(sim_id_selfcon.size(0), sim_id_selfcon.size(1), dtype=torch.bool, device=sim_id_selfcon.device)

        # ID_pOOD
        cat_pood_feat = torch.cat([pood_feat_j, pood_feat_b, pood_feat_v], dim=0)
        sim_id_pood = torch.mm(cat_id_supcon_feat, cat_pood_feat.T) / self.arg.temp_pood
        sim_id_pood_pos_mask = torch.zeros_like(sim_id_pood).bool()

        sim = torch.cat([sim_id_supcon, sim_id_selfcon, sim_id_pood], dim=1)
        pos_mask = torch.cat([sim_id_supcon_pos_mask, sim_id_selfcon_pos_mask, sim_id_pood_pos_mask], dim=1)
        
        log_prob = F.log_softmax(sim, dim=1)
        num_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
        loss_row = -(log_prob * pos_mask).sum(dim=1, keepdim=True) / num_pos
        loss_scl = (loss_row * cat_sample_weight).mean()
        
        return loss_scl

    def forward(self, ske_logits_list, ske_feat_list, label, ske_edl_info_dict_list, text_dict, layer_list):
        feature_j, feature_b, feature_v = ske_feat_list[0], ske_feat_list[1], ske_feat_list[2]
        feature_j_l2, feature_b_l2, feature_v_l2 = F.normalize(feature_j, p=2, dim=1), F.normalize(feature_b, p=2, dim=1), F.normalize(feature_v, p=2, dim=1)
        ske_feat_l2_list = [feature_j_l2, feature_b_l2, feature_v_l2]
        
        edl_info_dict_j, edl_info_dict_b, edl_info_dict_v = ske_edl_info_dict_list[0], ske_edl_info_dict_list[1], ske_edl_info_dict_list[2]
        ske_evidence_list = [edl_info_dict_j['evidence'], edl_info_dict_b['evidence'], edl_info_dict_v['evidence']]
        ske_alpha_list = [edl_info_dict_j['alpha'], edl_info_dict_b['alpha'], edl_info_dict_v['alpha']]
        ske_S_list = [edl_info_dict_j['S'], edl_info_dict_b['S'], edl_info_dict_v['S']]
        ske_u_list = [edl_info_dict_j['u'], edl_info_dict_b['u'], edl_info_dict_v['u']]

        B, _ = feature_j_l2.shape
        
        text_gpt4_feature = text_dict['text_gpt4_feature']
        text_feat_l2 = F.normalize(text_gpt4_feature, p=2, dim=2)
            
        enh_feature_j = torch.zeros((1,2), dtype=torch.float32).cuda()
        enh_feature_b = torch.zeros((1,2), dtype=torch.float32).cuda()
        enh_feature_v = torch.zeros((1,2), dtype=torch.float32).cuda()

        if self.arg.flag_cmi_lhsg:
            lhsg_j_dict = self.lhsg_j(feature_j_l2, text_feat_l2.squeeze(0), label)
            lhsg_b_dict = self.lhsg_b(feature_b_l2, text_feat_l2.squeeze(0), label)
            lhsg_v_dict = self.lhsg_v(feature_v_l2, text_feat_l2.squeeze(0), label)
            enh_feature_j, enh_feature_b, enh_feature_v = lhsg_j_dict['ske_feat_enhanced'], lhsg_b_dict['ske_feat_enhanced'], lhsg_v_dict['ske_feat_enhanced']
            loss_enh_j, loss_enh_b, loss_enh_v = lhsg_j_dict['loss_enh'], lhsg_b_dict['loss_enh'], lhsg_v_dict['loss_enh']
            loss_enh = loss_enh_j + loss_enh_b + loss_enh_v
        else:
            loss_enh = torch.tensor(0.).cuda()

        enh_feature_j_l2 = F.normalize(enh_feature_j, p=2, dim=1)
        enh_feature_b_l2 = F.normalize(enh_feature_b, p=2, dim=1)
        enh_feature_v_l2 = F.normalize(enh_feature_v, p=2, dim=1)

        # pOOD 
        pood_feat_j = torch.randn((B,256), dtype=torch.float32).cuda()
        pood_feat_b = torch.randn((B,256), dtype=torch.float32).cuda()
        pood_feat_v = torch.randn((B,256), dtype=torch.float32).cuda()
        
        if self.arg.flag_cmi_bpom:
            bpom_j_dict = self.bpom_j(feature_j_l2, label, layer_list[0], lhsg_j_dict['prototype'])
            bpom_b_dict = self.bpom_b(feature_b_l2, label, layer_list[1], lhsg_b_dict['prototype'])
            bpom_v_dict = self.bpom_v(feature_v_l2, label, layer_list[2], lhsg_v_dict['prototype'])
            pood_feat_j, pood_feat_b, pood_feat_v = bpom_j_dict['pood_feat'], bpom_b_dict['pood_feat'], bpom_v_dict['pood_feat']
        
        pood_feature_j_l2 = F.normalize(pood_feat_j, p=2, dim=1)
        pood_feature_b_l2 = F.normalize(pood_feat_b, p=2, dim=1)
        pood_feature_v_l2 = F.normalize(pood_feat_v, p=2, dim=1)
        pood_feat_l2_list = [pood_feature_j_l2, pood_feature_b_l2, pood_feature_v_l2]
        
        # enhanced features
        enh_ske_feat_l2_list = [enh_feature_j_l2, enh_feature_b_l2, enh_feature_v_l2]

        if self.arg.flag_loss_bcl:
            prototypes_list = [lhsg_j_dict['prototype'], lhsg_b_dict['prototype'], lhsg_v_dict['prototype']]
            loss_bcl = self.bcl(ske_feat_l2_list, label, enh_ske_feat_l2_list, pood_feat_l2_list, ske_u_list, prototypes_list)
        else:
            loss_bcl = torch.tensor(0.).cuda()
        
        info_dict = {'loss_enh':loss_enh,
                     'loss_bcl': loss_bcl,
                     'pood_feature_j_l2': pood_feature_j_l2,
                     'pood_feature_b_l2': pood_feature_b_l2,
                     'pood_feature_v_l2': pood_feature_v_l2,
                     }
        
        return info_dict