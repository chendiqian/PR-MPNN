import torch


class BindModel(torch.nn.Module):
    def __init__(self, inner_model, outer_model):
        super(BindModel, self).__init__()
        self.inner_model = inner_model
        self.outer_model = outer_model

    def forward(self, new_data, data):
        intermediate_node_emb = self.inner_model(new_data)
        pred = self.outer_model(data, intermediate_node_emb)
        return pred

    def reset_parameters(self):
        self.inner_model.reset_parameters()
        self.outer_model.reset_parameters()
