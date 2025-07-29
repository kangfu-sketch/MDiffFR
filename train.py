import os
import sys
import numpy as np
import datetime
import argparse
from mlp import MLPEngine
from utils import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='fedcs')
parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
parser.add_argument('--clients_sample_num', type=int, default=0)
parser.add_argument('--num_round', type=int, default=100)
parser.add_argument('--local_epoch', type=int, default=1)
parser.add_argument('--server_epoch', type=int, default=1)
parser.add_argument('--lr_eta', type=int, default=80)
parser.add_argument('--reg', type=float, default=10.0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr_client', type=float, default=1e-1)
parser.add_argument('--lr_server', type=float, default=1e-4)
parser.add_argument('--dataset', type=str, default='Food')
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_items_train', type=int)
parser.add_argument('--num_items_vali', type=int)
parser.add_argument('--num_items_test', type=int)
parser.add_argument('--content_dim', type=int)
parser.add_argument('--latent_dim', type=int, default=64)
parser.add_argument('--num_negative', type=int, default=5)
parser.add_argument('--client_model_layers', type=str, default='200')
parser.add_argument('--recall_k', type=str, default='20, 50, 100')
parser.add_argument('--l2_regularization', type=float, default=0.)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--device_id', type=int, default=3)
parser.add_argument("--start_server_num", type=int, default=1)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--step_increment", type=int, default=0)
parser.add_argument("--time_sample_strategy", type=str, default="uniform", help="uniform, importance")
parser.add_argument('--dp', type=float, default=0.)
"""         Diffusion Parameters        """
parser.add_argument('--wd2', type=float, default=0.001, help='weight decay for MLP')
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--out_dims', type=str, default='[]', help='the hidden dims for the decoder')
# params for the MLP
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--mlp_dims', type=str, default='[64, 32]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--time_emb_size', type=int, default=16, help='timestep embedding size')
parser.add_argument('--guidance_emb_size', type=int, default=64, help='timestep embedding size')
parser.add_argument('--mlp_act_func', type=str, default='tanh', help='default=tanh. the activation function for MLP')
parser.add_argument('--optimizer2', type=str, default='SGD', help='optimizer for MLP: Adam, AdamW, SGD, Adagrad, Momentum')
# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=10, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001)
parser.add_argument('--noise_max', type=float, default=0.02)
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')
args = parser.parse_args()


# Model.
config = vars(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_id"])


if len(config['recall_k']) > 1:
    config['recall_k'] = [int(item) for item in config['recall_k'].split(',')]
else:
    config['recall_k'] = [int(config['recall_k'])]

if len(config['client_model_layers']) > 1:
    config['client_model_layers'] = [config['latent_dim']].extend([int(item) for item in config['client_model_layers'].split(',')])
else:
    config['client_model_layers'] = [config['latent_dim']].extend(int(config['client_model_layers']))

if config['dataset'] in ['KU', 'Food', 'Dance', 'Movie']:
    df1 = pd.read_csv(f'../data/{config["dataset"]}/train.csv')  # 替换为你的文件路径
    df2 = pd.read_csv(f'../data/{config["dataset"]}/test.csv')  # 替换为你的文件路径
    df3 = pd.read_csv(f'../data/{config["dataset"]}/vali.csv')  # 替换为你的文件路径
    # 统计每一列的不重复元素个数
    train_counts1 = df1.nunique()
    test_counts2 = df2.nunique()
    vali_counts3 = df3.nunique()
    config['num_users'] = train_counts1['uid']
    config['num_items_train'] = train_counts1['iid']
    config['num_items_vali'] = vali_counts3['iid']
    config['num_items_test'] = test_counts2['iid']
    config['content_dim'] = 64
else:
    pass

engine = MLPEngine(config)

# Logging.
path = 'log/'
current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logname = os.path.join(path, current_time+'.txt')
initLogging(logname)

# Load Data
dataset_dir = "./data/" + config['dataset']
data_dict = load_data(dataset_dir, config['content_dim'])
# train, validation and test data with (uid, iid) dataframe format.
train_data = data_dict['train']
vali_data = data_dict['vali']
test_data = data_dict['test']
# train, validation, test cold-start item information, including item raw feature and reindex item id dict {ori_id: reindex_id}
train_item_content = data_dict['train_item_content']
user_ids = data_dict['user_ids']
vali_item_content = data_dict['vali_item_content']
vali_item_ids_map = data_dict['vali_item_ids_map']
test_item_content = data_dict['test_item_content']
test_item_ids_map = data_dict['test_item_ids_map']

vali_recalls = []
vali_precisions = []
vali_ndcgs = []
test_recalls = []
test_precisions = []
test_ndcgs = []
best_recall = 0
final_test_round = 0
for round in range(config['num_round']):
    # break
    logging.info('-' * 80)
    logging.info('-' * 80)
    logging.info('Round {} starts !'.format(round))

    all_train_data = negative_sampling(train_data, config['num_negative'])
    logging.info('-' * 80)
    logging.info('Training phase!')
    engine.fed_train_a_round(user_ids, all_train_data, round, train_item_content)

    logging.info('-' * 80)
    logging.info('Testing phase!')
    test_recall, test_precision, test_ndcg = engine.fed_evaluate(test_data, test_item_content, test_item_ids_map, round, t='test')
    logging.info('Recall@{} = {:.6f}, Recall@{} = {:.6f}, Recall@{} = {:.6f}'.format(
        config['recall_k'][0], test_recall[0],
        config['recall_k'][1], test_recall[1],
        config['recall_k'][2], test_recall[2]
    ))
    logging.info('Precision@{} = {:.6f}, Precision@{} = {:.6f}, Precision@{} = {:.6f}'.format(
        config['recall_k'][0], test_precision[0],
        config['recall_k'][1], test_precision[1],
        config['recall_k'][2], test_precision[2]
    ))
    logging.info('NDCG@{} = {:.6f}, NDCG@{} = {:.6f}, NDCG@{} = {:.6f}'.format(
        config['recall_k'][0], test_ndcg[0],
        config['recall_k'][1], test_ndcg[1],
        config['recall_k'][2], test_ndcg[2]
    ))
    test_recalls.append(test_recall)
    test_precisions.append(test_precision)
    test_ndcgs.append(test_ndcg)

    logging.info('-' * 80)
    logging.info('Validating phase!')
    vali_recall, vali_precision, vali_ndcg = engine.fed_evaluate(vali_data, vali_item_content, vali_item_ids_map, round, t='vali')
    logging.info('Recall@{} = {:.6f}, Recall@{} = {:.6f}, Recall@{} = {:.6f}'.format(
        config['recall_k'][0], vali_recall[0],
        config['recall_k'][1], vali_recall[1],
        config['recall_k'][2], vali_recall[2]
    ))
    logging.info('Precision@{} = {:.6f}, Precision@{} = {:.6f}, Precision@{} = {:.6f}'.format(
        config['recall_k'][0], vali_precision[0],
        config['recall_k'][1], vali_precision[1],
        config['recall_k'][2], vali_precision[2]
    ))
    logging.info('NDCG@{} = {:.6f}, NDCG@{} = {:.6f}, NDCG@{} = {:.6f}'.format(
        config['recall_k'][0], vali_ndcg[0],
        config['recall_k'][1], vali_ndcg[1],
        config['recall_k'][2], vali_ndcg[2]
    ))
    logging.info('')
    vali_recalls.append(vali_recall)
    vali_precisions.append(vali_precision)
    vali_ndcgs.append(vali_ndcg)

    if np.sum(vali_recall) >= np.sum(best_recall):
        best_recall = vali_recall
        final_test_round = round
        logging.info("Best result has been updated.")

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
str = current_time + '-' + 'latent_dim: ' + str(config['latent_dim']) + '-' + 'lr_client: ' + str(config['lr_client']) \
      + '-' + 'lr_server: ' + str(config['lr_server']) + '-' + 'local_epoch: ' + str(config['local_epoch']) + '-' + \
      'server_epoch: ' + str(config['server_epoch']) + '-' + 'client_model_layers: ' + str(config['client_model_layers']) + '-' \
      'clients_sample_ratio: ' + str(config['clients_sample_ratio']) + '-' + 'num_round: ' + str(config['num_round']) \
      + '-' + 'negatives: ' + str(config['num_negative']) + '-' + 'lr_eta: ' + str(config['lr_eta']) + '-' + \
      'batch_size: ' + str(config['batch_size']) + '-' + 'Recall: ' + str(test_recalls[final_test_round]) + '-' \
      + 'Precision: ' + str(test_precisions[final_test_round]) + '-' + 'NDCG: ' + str(test_ndcgs[final_test_round]) + '-' \
      + 'best_round: ' + str(final_test_round) + '-' + 'recall_k: ' + str(config['recall_k']) + '-' + \
      'optimizer: ' + config['optimizer'] + '-' + 'l2_regularization: ' + str(config['l2_regularization']) + '-' + \
      'reg: ' + str(config['reg'])
file_name = "sh_result/"+config['dataset']+".txt"
with open(file_name, 'a') as file:
    file.write(str + '\n')

logging.info('fedcs')
logging.info('recall_list: {}'.format(test_recalls))
logging.info('precision_list: {}'.format(test_precisions))
logging.info('ndcg_list: {}'.format(test_ndcgs))
logging.info('clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr_client: {}, lr_server: {}, local_epoch: {},'
             'server_epoch: {}, client_model_layers: {}, recall_k: {}, dataset: {}, '
             'factor: {}, negatives: {}, reg: {}'.format(config['clients_sample_ratio'], config['lr_eta'],
                                                         config['batch_size'], config['lr_client'], config['lr_server'],
                                                         config['local_epoch'], config['server_epoch'],
                                                         config['client_model_layers'],
                                                         config['recall_k'], config['dataset'], config['latent_dim'],
                                                         config['num_negative'], config['reg']))
logging.info('Best test recall: {}, precision: {}, ndcg: {} at round {}'.format(test_recalls[final_test_round],
                                                                                test_precisions[final_test_round],
                                                                                test_ndcgs[final_test_round],
                                                                                final_test_round))