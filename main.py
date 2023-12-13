import os
import os.path as osp
import MLdataset
import argparse
import time
from model import ModelFirst, ModelSecond
import utils
from utils import AverageMeter
import evaluation
import torch
import numpy as np
from loss import LossFirstStage, LossSecondStage
from torch import nn
from torch.optim import AdamW
import copy


def initialize(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)


def train_first(loader, model, loss_fn, opt, sche, epoch, logger, v):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data = data[v].to(device)
        label = label.to(device)
        inc_V_ind = inc_V_ind.float().to(device)
        inc_L_ind = inc_L_ind.float().to(device)

        pred, _, (mu, std) = model(data, inc_V_ind[:, v].unsqueeze(1))
        loss = loss_fn(pred, label, mu, std, inc_V_ind[:, v].unsqueeze(1), inc_L_ind)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        losses.update(loss.item())

        if sche is not None:
            sche.step()

        batch_time.update(time.time() - end)
        end = time.time()

    if logger is not None:
        logger.info('Epoch:[{0}]\t'
                    'Time {batch_time.avg:.3f}\t'
                    'Data {data_time.avg:.3f}\t'
                    'Loss {losses.avg:.3f}\t'.format(epoch, batch_time=batch_time, data_time=data_time, losses=losses))

    return losses, model


def test_first(loader, model, epoch, logger, v, mode='val'):
    batch_time = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
            data = data[v].to(device)
            inc_V_ind = inc_V_ind.float().to(device)
            pred, _, (_, _) = model(data, inc_V_ind[: ,v].unsqueeze(1))

            pred = pred.cpu()

            total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(total_labels) > 0 else label.numpy()
            total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(total_preds) > 0 else \
                pred.detach().numpy()

            batch_time.update(time.time() - end)
            end = time.time()
    total_labels = torch.tensor(total_labels)
    total_preds = torch.tensor(total_preds)

    evaluation_results = [nn.BCELoss()(total_preds, total_labels)]
    if logger is not None:
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'CE {ce:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ce=evaluation_results[0],
        ))

    return evaluation_results


def train_second(loader, model_first, model_second, loss_fn, opt, sche, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for model in model_first:
        model.eval()
    model_second.train()
    end = time.time()

    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)

        inc_V_ind = inc_V_ind.float().to(device)
        inc_L_ind = inc_L_ind.float().to(device)
        data = [v_data.to(device) for v_data in data]
        data = [model_first[v](data[v], inc_V_ind[: , v].unsqueeze(1), 0)[1].detach() for v in range(len(data))]
        label = label.to(device)

        pred, rec_r = model_second(data, inc_V_ind)
        loss = loss_fn(pred, label, inc_V_ind, inc_L_ind, data, rec_r)
        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_second.parameters(), 1)
        opt.step()

        losses.update(loss.item())
        if sche is not None:
            sche.step()

        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}\t'.format(
        epoch, batch_time=batch_time, data_time=data_time, losses=losses))

    return losses, model_second


def test_second(loader, model_first, model_second, epoch, logger, mode='val'):
    batch_time = AverageMeter()
    total_labels = []
    total_preds = []

    for model in model_first:
        model.eval()
    model_second.eval()
    end = time.time()

    with torch.no_grad():
        for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
            # data_time.update(time.time() - end)
            inc_V_ind = inc_V_ind.float().to(device)
            data = [v_data.to(device) for v_data in data]
            data = [model_first[v](data[v], inc_V_ind[:, v].unsqueeze(1), 0)[1].detach() for v in range(len(data))]

            pred, _ = model_second(data, inc_V_ind)
            pred = pred.cpu()

            total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(total_labels) > 0 else label.numpy()
            total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(total_preds) > 0 else \
                pred.detach().numpy()

            batch_time.update(time.time() - end)
            end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)

    # if mode == 'val' or mode == 'train':
    if mode == 'train':
        evaluation_results = [evaluation.compute_average_precision(total_preds, total_labels)]
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ap=evaluation_results[0],
        ))
    else:
        evaluation_results = evaluation.do_metric(total_preds, total_labels)  # compute auc is very slow
        logger.info('Epoch:[{0}]\t'
                    'Mode:{mode}\t'
                    'Time {batch_time.avg:.3f}\t'
                    'AP {ap:.4f}\t'
                    'HL {hl:.4f}\t'
                    'RL {rl:.4f}\t'
                    'AUC {auc:.4f}\t'.format(
            epoch, mode=mode, batch_time=batch_time,
            ap=evaluation_results[0],
            hl=evaluation_results[1],
            rl=evaluation_results[2],
            auc=evaluation_results[3]
        ))

    return evaluation_results


def main(args, file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view_MaskRatios_' + str(
        args.mask_view_ratio) + '_LabelMaskRatio_' +
                              str(args.mask_label_ratio) + '_TraindataRatio_' +
                              str(args.training_sample_ratio) + '.mat')

    folds_num = args.folds_num
    folds_results = [AverageMeter() for _ in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir, args.name + args.dataset + '_V_' + str(
            args.mask_view_ratio) + '_L_' +
                           str(args.mask_label_ratio) + '_T_' +
                           str(args.training_sample_ratio) + '_beta_' +
                           str(args.beta) + '_dropout_' +
                           str(args.dropout) + '_d_model_' +
                           str(args.d_z) + '.txt')
    else:
        logfile = None
    logger = utils.setLogger(logfile)

    for fold_idx in range(folds_num):
        fold_idx = fold_idx
        train_dataloder, train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,
                                                                    training_ratio=args.training_sample_ratio,
                                                                    fold_idx=fold_idx,
                                                                    mode='train',
                                                                    batch_size=args.batch_size,
                                                                    shuffle=False,
                                                                    num_workers=4)
        test_dataloder, test_dataset = MLdataset.getIncDataloader(data_path,
                                                                  fold_data_path,
                                                                  training_ratio=args.training_sample_ratio,
                                                                  val_ratio=0.15,
                                                                  fold_idx=fold_idx,
                                                                  mode='test',
                                                                  batch_size=args.batch_size,
                                                                  num_workers=4)
        val_dataloder, val_dataset = MLdataset.getIncDataloader(data_path,
                                                                fold_data_path,
                                                                training_ratio=args.training_sample_ratio,
                                                                fold_idx=fold_idx,
                                                                mode='val',
                                                                batch_size=args.batch_size,
                                                                num_workers=4)
        d_list = train_dataset.d_list  # dimension list
        n_view = len(d_list)
        n_cls = train_dataset.classes_num

        model_first = [ModelFirst(d_list[v], n_cls, args.theta, 0) for v in range(n_view)]
        model_second = ModelSecond(d_list, args.d_emb_second, args.n_enc_layer_second, args.n_dec_layer_second, n_cls, args.theta, args.dropout)

        for v in range(n_view):
            model_first[v] = model_first[v].to(device)
        model_second = model_second.to(device)

        if fold_idx == 0:
            print(f'number of model_first: {n_view}')
            print(f'The model_first has {sum(sum(p.numel() for p in model_first[v].parameters() if p.requires_grad) for v in range(n_view)):,} trainable parameters')
            print(f'The model_second has {sum(p.numel() for p in model_second.parameters() if p.requires_grad):,} trainable parameters')

        for v in range(n_view):
            initialize(model_first[v])
        initialize(model_second)

        loss_first = LossFirstStage(args.beta).to(device)
        loss_second = LossSecondStage(args.alpha, args.gamma).to(device)

        optimizer_first = [AdamW(model_first[v].parameters(), lr=args.lr, weight_decay=args.weight_decay_first) for v in range(n_view)]
        optimizer_second = AdamW(model_second.parameters(), lr=args.lr, weight_decay=args.weight_decay_second)

        scheduler = None

        logger.info('train_data_num:' + str(len(train_dataset)) + '  test_data_num:' + str(len(test_dataset)) +
                    '   fold_idx:' + str(fold_idx))
        print(args)

        best_model_first_dict = [{'model': model_first[v].state_dict(), 'epoch': 0} for v in range(n_view)]

        if args.load_first:
            for v in range(n_view):
                model_first[v].load_state_dict(torch.load(args.weights_dir+'model_first'+str(v)+'.pt'))
        else:
            for v in range(n_view):
                val_metric_list_first = []
                loss_list_first = []

                print('model_first ' + str(v) + ' start training...')
                best_result_first = 1e9
                best_epoch = 0
                model = model_first[v]
                for epoch in range(args.epochs):
                    loss_v, model = train_first(train_dataloder, model, loss_first, optimizer_first[v], scheduler, epoch, logger, v)
                    train_metric = test_first(train_dataloder, model, epoch, logger, v, mode='train')
                    val_metric = test_first(val_dataloder, model, epoch, logger, v, mode='val')
                    loss_list_first.append(loss_v.avg)

                    val_metric = val_metric[0]
                    train_metric = train_metric[0]

                    val_metric_list_first.append(val_metric)

                    if val_metric < best_result_first:
                        best_result_first = val_metric
                        best_epoch = epoch
                        best_model_first_dict[v]['model'] = copy.deepcopy(model.state_dict())
                        best_model_first_dict[v]['epoch'] = epoch

                    if train_metric < val_metric and (epoch - best_epoch > args.patience_first):
                        print('View', v, ' Training stopped: epoch=%d' %(epoch))
                        break

                if args.save_first:
                    torch.save(best_model_first_dict[v]['model'], args.weights_dir+'model_first'+str(v)+'.pt')

                if args.save_curve:
                    np.save(osp.join(args.curve_dir, args.dataset + '_V_' + str(args.mask_view_ratio) + '_L_' + str(
                        args.mask_label_ratio)) + '_' + str(fold_idx) + '_first' + str(v) + '.npy',
                            np.array(list(zip(val_metric_list_first, loss_list_first))))
            for v in range(n_view):
                model_first[v].load_state_dict(best_model_first_dict[v]['model'])

        best_result = 0
        val_metric_list_second = []
        loss_list_second = []
        best_epoch = 0
        best_model_second_dict = {'model': model_second.state_dict(), 'epoch': 0}

        for epoch in range(args.epochs):

            train_losses_second, model_second = train_second(train_dataloder, model_first, model_second, loss_second, optimizer_second,
                                                       scheduler, epoch, logger)
            _ = test_second(train_dataloder, model_first, model_second, epoch, logger, mode='train')
            val_metric = test_second(val_dataloder, model_first, model_second, epoch, logger, mode='val')

            loss_list_second.append(train_losses_second.avg)

            # val_metric = val_metric[0]
            val_metric = val_metric[0] * 0.2 + val_metric[1] * 0.2 + val_metric[2] * 0.2 + val_metric[3] * 0.4

            val_metric_list_second.append(val_metric)
            if val_metric > best_result:
                best_result = val_metric
                best_model_second_dict['model'] = copy.deepcopy(model_second.state_dict())
                best_model_second_dict['epoch'] = epoch
                best_epoch = epoch

            if epoch > 150 and (epoch - best_epoch > args.patience_second):
                print('Training stopped: epoch=%d' % (epoch))
                break

        if args.save_curve:
            np.save(osp.join(args.curve_dir, args.dataset + '_V_' + str(args.mask_view_ratio) + '_L_' + str(
                args.mask_label_ratio)) + '_' + str(fold_idx) + '_' + str(args.alpha) + '_' + str(args.beta) + '_' +
                    str(args.gamma) + '_' + 'second' + '.npy',
                    np.array(list(zip(val_metric_list_second, loss_list_second))))

        model_second.load_state_dict(best_model_second_dict['model'])
        test_result_second = test_second(test_dataloder, model_first, model_second, -1, logger, mode='test')

        logger.info(
            'final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(
                fold_idx, best_epoch, test_result_second[0], test_result_second[1], test_result_second[2], test_result_second[3]))

        for i in range(9):
            folds_results[i].update(test_result_second[i])

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP  HL  RL  AUCme  one_error  coverage  macAUC  macro_f1  micro_f1  alpha  beta  gamma\n')
    # generate string-result of 9 metrics and two parameters
    res_list = [str(round(res.avg, 4)) + '+' + str(round(res.std, 4)) for res in folds_results]
    res_list.extend([str(args.alpha), str(args.beta), str(args.gamma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'records'))
    parser.add_argument('--root-dir', type=str, metavar='PATH', default='./data/')
    parser.add_argument('--dataset', type=str, default='corel5k')  # mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=1, type=int)
    parser.add_argument('--weights_dir', type=str, metavar='PATH', default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--name', type=str, default='final_')
    parser.add_argument('--save_first', type=bool, default=False)  # if True, save fc after training
    parser.add_argument('--load_first', type=bool, default=False)  # if True, won't train fc
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay_first', type=float, default=0.01)
    parser.add_argument('--weight_decay_second', type=float, default=1)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--d_emb_second', type=int, default=1024)
    parser.add_argument('--n_block_second', type=int, default=3)
    parser.add_argument('--n_enc_layer_second', type=int, default=2)
    parser.add_argument('--n_dec_layer_second', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--patience_first', type=int, default=50)
    parser.add_argument('--patience_second', type=int, default=60)
    parser.add_argument('--theta', type=float, default=0.8)

    args = parser.parse_args()

    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    assert torch.cuda.is_available()
    device = 'cuda:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # hyperparams
    lr_list = [0.0001]
    alpha_list = [45]  # 45 500 100 200 100
    beta_list = [1e-2] # 1e-2 for all
    gamma_list = [15]  # 15 100 60 30 100
    d_emb_list = [512] # 512 for corel5k 768 for others

    args.datasets = ['corel5k']  # corel5k pascal07 espgame mirflickr iaprtc12
    args.load_first = False  # if True, only train one fold
    args.save_first = False

    if args.load_first:
        args.folds_num = 1

    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for d_emb in d_emb_list:
                        args.d_emb_second = d_emb
                        for dataset in args.datasets:
                            args.dataset = dataset
                            file_path = osp.join(args.records_dir, args.name + args.dataset + '_ViewMask_' + str(
                                args.mask_view_ratio) + '_LabelMask_' + str(args.mask_label_ratio) + '_Training_' +
                                                 str(args.training_sample_ratio) + '.txt')
                            main(args, file_path)
