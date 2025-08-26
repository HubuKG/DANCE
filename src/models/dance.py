# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse.linalg import svds, eigsh
from common.abstract_recommender import GeneralRecommender


class DANCE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PGL, self).__init__(config, dataset)
        self.mode = config['mode']

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.lambda1 = config['reg_weight']
        
        self.mm_image_weight = config['mm_image_weight']

        self.n_nodes = self.n_users + self.n_items

        self.sub_graph, self.mm_adj = None, None
        self.v_preference, self.t_preference = None, None
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path,'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
        self.dropoutf = nn.Dropout(config['dropout'])
        self.num_heads = config['num_heads']
        self.temp = config['temp']
        self.attention_layer = MultiHeadAttentionLayer(input_dim=4 * self.embedding_dim, num_heads=self.num_heads, device=self.device)
        

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for (i, j), val in data_dict.items():
            A[i, j] = val
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
            degree_len = int(self.edge_values.size(0) * 0.3)
            degree_idx = torch.multinomial(self.edge_values, degree_len)
            keep_indices = self.edge_indices[:, degree_idx]
            keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
            all_values = torch.cat((keep_values, keep_values))
            keep_indices[1] += self.n_users
            all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
            self.sub_graph = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * c_inv_sqrt[indices[1]]
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values


    def forward(self, adj):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)  
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)  

        image_feats, text_feats = F.normalize(image_feats), F.normalize(text_feats)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)   
        

        h = item_embeds 
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)  

        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):

            row, col = adj._indices()
            node1 = ego_embeddings[row]
            node2 = ego_embeddings[col]
            attention_weights = self.attention_layer(node1, node2)
            attention_weights = attention_weights.squeeze()

            new_values = adj._values() * attention_weights
            new_adj = torch.sparse.FloatTensor(adj._indices(), new_values, adj.shape).to(self.device)

            ego_embeddings = torch.sparse.mm(new_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        return -torch.mean(maxi)

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.exp((view1 * view2).sum(dim=-1) / temperature)
        ttl_score = torch.exp(torch.matmul(view1, view2.T) / temperature).sum(dim=1)
        return -torch.mean(torch.log(pos_score / ttl_score))

    def bidirectional_infonce(self,u_emb, i_emb, temperature=0.2):

        # Dot products (u → i)
        pos_score_ui = torch.sum(u_emb * i_emb, dim=-1)  # [B]
        pos_score_ui = torch.exp(pos_score_ui / temperature)
        total_score_ui = torch.exp(torch.matmul(u_emb, i_emb.T) / temperature).sum(dim=1)
        loss_ui = -torch.log(pos_score_ui / total_score_ui).mean()

        # Dot products (i → u)
        pos_score_iu = torch.sum(i_emb * u_emb, dim=-1)
        pos_score_iu = torch.exp(pos_score_iu / temperature)
        total_score_iu = torch.exp(torch.matmul(i_emb, u_emb.T) / temperature).sum(dim=1)
        loss_iu = -torch.log(pos_score_iu / total_score_iu).mean()

        # Final CL loss
        cl_loss = (loss_ui + loss_iu) / 2
        return cl_loss

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        ua_embeddings, ia_embeddings = self.forward(self.sub_graph)
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings) 

        reg_embedding_loss_v = (
            (self.v_preference[users] ** 2).mean()
            if self.v_preference is not None
            else 0.0
        )
        reg_embedding_loss_t = (
            (self.t_preference[users] ** 2).mean()
            if self.t_preference is not None
            else 0.0
        )
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)


        # dropout view
        u_view1 = F.normalize(self.dropoutf(u_g_embeddings), dim=-1)
        u_view2 = F.normalize(self.dropoutf(u_g_embeddings), dim=-1)
        i_view1 = F.normalize(self.dropoutf(pos_i_g_embeddings), dim=-1)
        i_view2 = F.normalize(self.dropoutf(pos_i_g_embeddings), dim=-1)

        # intra-view (self-supervised) contrastive
        cl_u = self.InfoNCE(u_view1, u_view2, temperature=self.temp)
        cl_i = self.InfoNCE(i_view1, i_view2, temperature=self.temp)

        # cross-view contrastive
        cl_cross = self.bidirectional_infonce(u_view1, i_view1, temperature=self.temp)

        cl_loss = (cl_u + cl_i + cl_cross) / 3

        total_loss = batch_mf_loss + self.lambda1 * reg_loss + self.lambda2 * cl_loss

        return total_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.T)
        return scores


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads, device):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_heads)
        ])
        self.device = device
        self.to(device)

    def forward(self, node1, node2):
        concat_features = torch.cat([node1, node2], dim=1)  # [E, 2*D]
        attention_scores = [torch.sigmoid(att(concat_features)) for att in self.attentions]  # List of [E, 1]
        attention_scores = torch.stack(attention_scores, dim=0)  # [H, E, 1]
        attention_scores = attention_scores.mean(dim=0).squeeze()  # [E]
        return attention_scores  #  [E]
