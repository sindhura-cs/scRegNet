import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):
        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T,negative_slope=self.alpha)
        return e

    def forward(self,x,adj):
        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense()>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass

        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data = F.normalize(output_data,p=2,dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data


class scTransNet_GCN(nn.Module):
    def __init__(self,input_dim,args,gene_dim,device):
        super(scTransNet_GCN, self).__init__()
        self.args = args
        self.device = device

        self.convs = torch.nn.ModuleList()
        
        for i in range(self.args.gnn_num_layers):
            hidden_dim = self.args.gnn_hidden_dims[i]
            self.convs.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]

            if i==0:
                self.layers.append(nn.Linear(input_dim+gene_dim,hidden_dim_mlp))
            else:
                self.layers.append(nn.Linear(input_dim,hidden_dim_mlp))
            
            input_dim = hidden_dim_mlp

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2*input_dim, 2)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()
        
    def encode(self,x,adj):
        for i, conv in enumerate(self.convs):
            x = conv(x,adj)
            if i < len(self.convs) - 1:
                x = F.relu(x) 
                p = self.args.dropout
                x = F.dropout(x, p, training=self.training)
        return x 
        
    def decode(self,tf_embed,target_embed):
        if self.args.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)
            return prob
        elif self.args.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))
        
    def forward(self, x, adj, train_sample, llm_emb):
        embed = self.encode(x,adj)
        embed = torch.cat((llm_emb, embed), dim=1)
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                tf_embed = F.dropout(tf_embed, p)
        
        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                target_embed = F.dropout(target_embed, p)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred
    
    def get_embedding(self):
        return self.tf_ouput, self.target_output


class scTransNet_SAGE(nn.Module):
    def __init__(self,input_dim,args,gene_dim,device):
        super(scTransNet_SAGE, self).__init__()
        self.args = args
        self.device = device

        self.convs = torch.nn.ModuleList()
        
        for i in range(self.args.gnn_num_layers):
            hidden_dim = self.args.gnn_hidden_dims[i]
            self.convs.append(SAGEConv(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]

            if i==0:
                self.layers.append(nn.Linear(input_dim+gene_dim,hidden_dim_mlp))
            else:
                self.layers.append(nn.Linear(input_dim,hidden_dim_mlp))
            
            input_dim = hidden_dim_mlp

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2*input_dim, 2)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()

    def encode(self,x,adj):
        for i, conv in enumerate(self.convs):
            x = conv(x,adj)
            if i < len(self.convs) - 1:
                x = F.relu(x) 
                p = self.args.dropout
                x = F.dropout(x, p, training=self.training)
        return x 
        
    def decode(self,tf_embed,target_embed):
        if self.args.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)
            return prob
        elif self.args.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))
        
    def forward(self, x, adj, train_sample, llm_emb):
        embed = self.encode(x,adj)
        embed = torch.cat((llm_emb, embed), dim=1)
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                tf_embed = F.dropout(tf_embed, p)
        
        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                target_embed = F.dropout(target_embed, p)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred
    
    def get_embedding(self):
        return self.tf_ouput, self.target_output


class scTransNet_GAT(nn.Module):
    def __init__(self,input_dim,args,gene_dim,device):
        super(scTransNet_GAT, self).__init__()
        self.args = args
        self.device = device
        self.reduction = self.args.reduction

        self.convs = []
        gnn_num_layers = self.args.gnn_num_layers
        for i in range(gnn_num_layers):
            num_head = self.args.num_heads[i]
            hidden_dim = self.args.gnn_hidden_dims[i]

            conv_layer = [AttentionLayer(input_dim,hidden_dim,self.args.alpha) for _ in range(num_head)]
            self.convs.append(conv_layer)
            for j, attention in enumerate(conv_layer):
                self.add_module(f'ConvLayer{i}_AttentionHead{j}',attention)
            input_dim = hidden_dim

            if self.reduction == 'concate' and i<gnn_num_layers-1:
                input_dim = num_head*hidden_dim
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.args.mlp_num_layers):
            hidden_dim_mlp = self.args.mlp_hidden_dims[i]

            if i==0:
                self.layers.append(nn.Linear(input_dim+gene_dim,hidden_dim_mlp))
            else:
                self.layers.append(nn.Linear(input_dim,hidden_dim_mlp))
            
            input_dim = hidden_dim_mlp

        if self.args.type == 'MLP':
            self.linear = nn.Linear(2*input_dim, 2)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            for attention in conv:
                attention.reset_parameters()
        
        for layer in self.layers:
            layer.reset_parameters()

    def encode(self,x,adj):
        for i, conv in enumerate(self.convs):
            if i == len(self.convs) - 1:
                out = torch.mean(torch.stack([att(x, adj) for att in conv]),dim=0)
                return out
            elif self.reduction =='concate':
                x = torch.cat([att(x, adj) for att in conv], dim=1)
            elif self.reduction =='mean':
                x = torch.mean(torch.stack([att(x, adj) for att in conv]), dim=0)
            else:
                raise TypeError
            
            x = F.elu(x)
        
    def decode(self,tf_embed,target_embed):
        if self.args.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)
            return prob
        elif self.args.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)
            return prob
        elif self.args.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))
        
    def forward(self, x, adj, train_sample, llm_emb):
        embed = self.encode(x,adj)
        embed = torch.cat((llm_emb, embed), dim=1)
        tf_embed = target_embed = embed

        for i, layer in enumerate(self.layers):
            tf_embed = layer(tf_embed)
            tf_embed = F.leaky_relu(tf_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                tf_embed = F.dropout(tf_embed, p)
        
        for i, layer in enumerate(self.layers):
            target_embed = layer(target_embed)
            target_embed = F.leaky_relu(target_embed)
            if i < len(self.layers) - 1:
                p = self.args.dropout
                target_embed = F.dropout(target_embed, p)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred
    
    def get_embedding(self):
        return self.tf_ouput, self.target_output