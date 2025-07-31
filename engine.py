import torch

from diffusion.DNN import xavier_normal_initialization
from diffdataset import EmbeddingDataset
import numpy as np
import copy
from torch.utils.data import DataLoader

from data import UserItemRatingDataset
from utils import *
from torch.distributions.laplace import Laplace

class Engine(object):
    def __init__(self, config):
        self.config = config  # model configuration

        # self.server_opt = torch.optim.Adam(self.server_model.parameters(), lr=config['lr_server'], weight_decay=config['l2_regularization'])

        self.server_model_param = {}
        self.client_model_params = {}
        self.client_crit = torch.nn.BCELoss()
        self.server_crit = torch.nn.MSELoss()
        
        self.device = torch.device("cuda:0" if config['use_cuda'] else "cpu")
    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers):
        """train a batch and return an updated model."""
        # load batch data.
        _, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        reg_item_embedding = copy.deepcopy(self.server_model_param['global_item_rep'])

        if self.config['use_cuda'] is True:
            items, ratings = items.cuda(), ratings.cuda()
            reg_item_embedding = reg_item_embedding.cuda()

        optimizer, optimizer_i = optimizers
        # update score function.
        optimizer.zero_grad()
        ratings_pred = model_client(items)
        loss = self.client_crit(ratings_pred.view(-1), ratings)
        loss.backward()
        optimizer.step()

        # update item embedding.
        optimizer_i.zero_grad()
        ratings_pred = model_client(items)
        loss_i = self.client_crit(ratings_pred.view(-1), ratings)
        loss_i.backward()
        optimizer_i.step()
        return model_client

    def aggregate_clients_params(self, round_user_params, item_content, round_id):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        t = 0
        for user in round_user_params.keys():
            # load a user's parameters.
            user_params = round_user_params[user]
            # print(user_params)
            if t == 0:
                self.server_model_param = copy.deepcopy(user_params)
            else:
                for key in user_params.keys():
                    self.server_model_param[key].data += user_params[key].data
            t += 1
        for key in self.server_model_param.keys():
            self.server_model_param[key].data = self.server_model_param[key].data / len(round_user_params)

        item_content = torch.tensor(item_content)
        target = self.server_model_param['embedding_item.weight'].data
        if self.config['use_cuda'] is True:
            item_content = item_content.cuda()
            target = target.cuda()

        # dataload
        dataset = EmbeddingDataset(target, item_content)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
        )

        self.server_diffusion.model.train()

        for epoch in range(self.config['server_epoch']):
            total_loss = 0.
            for batch_idx, batch in enumerate(dataloader):
                target, guidance = batch['target'].to(self.device), batch['guidance'].to(self.device)
                loss = self.server_diffusion.training_losses(target, guidance, self.config['reweight'])
                total_loss += loss

        # store the global item representation learned by server model.
        self.server_diffusion.model.eval()
        with torch.no_grad():
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
            )
            diffusion_item = self.server_diffusion.predict(dataloader)
            global_item_rep = diffusion_item

        self.server_model_param['global_item_rep'] = global_item_rep


    def fed_train_a_round(self, user_ids, all_train_data, round_id, item_content):
        """train a round."""
        # sample users participating in single round.
        if self.config['clients_sample_ratio'] <= 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = np.random.choice(user_ids, num_participants, replace=False)
        else:
            participants = np.random.choice(user_ids, self.config['clients_sample_num'], replace=False)

        # initialize server parameters for the first round.
        if round_id == 0:
            item_content = torch.tensor(item_content)
            if self.config['use_cuda'] is True:
                item_content = item_content.cuda()
            # self.server_model.eval()
            # with torch.no_grad():
                # global_item_rep = self.server_diffusion.predict(item_content)
            global_item_rep = item_content
            self.server_model_param['global_item_rep'] = global_item_rep

        # store users' model parameters of current round.
        round_participant_params = {}
        # perform model update for each participated user.
        for user in participants:
            # copy the client model architecture from self.client_model
            model_client = copy.deepcopy(self.client_model)
            if round_id != 0:
                user_param_dict = copy.deepcopy(self.client_model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            optimizer = torch.optim.SGD(model_client.affine_output.parameters(),
                                        lr=self.config['lr_client'],
                                        weight_decay=self.config['l2_regularization'])  # MLP optimizer
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr_client'] * self.config['num_items_train'] * self.config['lr_eta'],
                                          weight_decay=self.config['l2_regularization'])  # Item optimizer
            optimizers = [optimizer, optimizer_i]

            # load current user's training data and instance a train loader.
            user_train_data = all_train_data[user]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client = self.fed_train_single_batch(model_client, batch, optimizers)
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' local parameters for personalization.
            self.client_model_params[user] = {}
            for key in client_param.keys():
                if key != 'embedding_item.weight':
                    # self.client_model_params[user][key] = copy.deepcopy(client_param[key]).data.cpu()
                    self.client_model_params[user][key] = copy.deepcopy(client_param[key]).data
            # store client models' local parameters for global update.
            round_participant_params[user] = {}
            # round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(client_param['embedding_item.weight']).data.cpu()
            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(client_param['embedding_item.weight']).data
            if self.config['dp'] != 0:
                round_participant_params[user]['embedding_item.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_item.weight'].shape).sample().cuda()

        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params, item_content, round_id)


    def fed_evaluate(self, evaluate_data, item_content, item_ids_map, round_id, t):
        """evaluate all client models' performance using testing data."""
        item_content = torch.tensor(item_content)
        if self.config['use_cuda'] is True:
            item_content = item_content.cuda()

        # obtain cold-start items' latent representation via server model.
        current_model = copy.deepcopy(self.server_diffusion)
        current_model.eval()
        with torch.no_grad():
            dataset = EmbeddingDataset(item_content, item_content)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
            )
            item_rep = current_model.predict(dataloader)

        # obtain cola-start items' prediction for each user.
        user_ids = evaluate_data['uid'].unique()
        user_item_preds = {}
        for user in user_ids:
            user_model = copy.deepcopy(self.client_model)
            user_param_dict = copy.deepcopy(self.client_model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                cold_pred = user_model.cold_predict(item_rep)
                user_item_preds[user] = cold_pred.view(-1)

        # compute the evaluation metrics.
        recall, precision, ndcg = compute_metrics(evaluate_data, user_item_preds, item_ids_map, self.config['recall_k'])
        return recall, precision, ndcg
