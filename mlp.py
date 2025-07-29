import torch
from engine import Engine
import diffusion.gaussian_diffusion as gd
import torch.optim as optim


class Client(torch.nn.Module):
    def __init__(self, config):
        super(Client, self).__init__()
        self.config = config
        self.num_items_train = config['num_items_train']
        self.latent_dim = config['latent_dim']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items_train, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        vector = self.embedding_item(item_indices)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def cold_predict(self, item_embeddings):
        logits = self.affine_output(item_embeddings)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass

class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        super(MLPEngine, self).__init__(config)
        self.client_model = Client(config)
        # Build Diffusion
        if config['mean_type'] == 'x0':
            mean_type = gd.ModelMeanType.START_X
        elif config['mean_type'] == 'eps':
            mean_type = gd.ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % config['mean_type'])
        diffusion = gd.GaussianDiffusion(config, mean_type, self.device).to(self.device)

        diff_num = sum([param.nelement() for param in diffusion.get_model().parameters()])
        print("Number of server parameters:", diff_num)
        self.server_diffusion = diffusion

        if config['use_cuda'] is True:
            self.client_model.cuda()
            self.server_diffusion.cuda()

