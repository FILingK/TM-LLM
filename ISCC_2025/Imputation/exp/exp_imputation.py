from data_provider.data_factory import data_provider
from data_provider.data_factory_gan import data_provider_gan
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, ndcg
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from models.adversarial_learning import Generator, Discriminator, weights_init_normal

warnings.filterwarnings('ignore')

class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()
    def forward(self, pred, true):
        x_loss = torch.mean(100 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        return x_loss


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

        self.pred_len = 0
        self.seq_len = self.args.seq_len
        self.generator = Generator(self.args.seq_len, self.args.dec_in)
        self.discriminator = Discriminator(self.args.seq_len, self.args.dec_in)
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_data_gan(self, flag):
        data_set_gan, data_loader_gan = data_provider_gan(self.args, flag)
        return data_set_gan, data_loader_gan

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pre_vali(self, vali_data, vali_loader, adversarial_loss):
        total_loss = []
        self.generator.eval()
        self.discriminator.eval()
        with torch.no_grad():
            for i, batch_x in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                # random mask
                B, T, N = batch_x.shape

                # Initialize the mask
                mask = torch.zeros((B, T, N)).to(self.device)

                # Process each column (feature) separately
                for col in range(N):
                    # Generate random masks for each column
                    random_indices = torch.randperm(T * B)[:int((1-self.args.mask_rate) * T * B)]
                    flat_mask = mask[:, :, col].flatten()
                    flat_mask[random_indices] = 1
                    mask[:, :, col] = flat_mask.reshape(B, T)

                # Generate initial input, setting masked values to 0
                inp = batch_x.masked_fill(mask == 0, 0)
                # Assign values to masked entries: assign the average of the 10% unmasked values in each column to the 90% masked values.
                for col in range(N):
                    column_values = batch_x[:, :, col]
                    column_mask = mask[:, :, col]

                    unmasked_values = column_values[column_mask == 1]

                    if len(unmasked_values) > 0:
                        mean_value = unmasked_values.mean()
                    else:
                        mean_value = 0

                    inp[:, :, col][mask[:, :, col] == 0] = mean_value

                # Standardize
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev

                means1 = inp.mean(1, keepdim=True).detach()
                inp = inp - means1
                stdev1 = torch.sqrt(
                    torch.var(inp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                inp /= stdev1


                valid = Tensor(batch_x.size(0), 1).fill_(1.0)
                fake = Tensor(batch_x.size(0), 1).fill_(0.0)
                # -----------------
                #  Vali Generator
                # -----------------
                output = self.generator(inp)
                g_loss = adversarial_loss(self.discriminator(output), valid)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                real_loss = adversarial_loss(self.discriminator(batch_x), valid)
                fake_loss = adversarial_loss(self.discriminator(output.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                x_loss = torch.mean(self.args.Lambda * torch.abs(output - batch_x))
                g_loss += x_loss
                total_loss.append(x_loss)
        total_loss = np.average([loss.cpu().item() for loss in total_loss])
        self.generator.train()
        self.discriminator.train()
        return total_loss

    def pre_train(self, setting):
        train_data, train_loader = self._get_data_gan(flag='train')
        vali_data, vali_loader = self._get_data_gan(flag='val')
        test_data, test_loader = self._get_data_gan(flag='test')
        adversarial_loss = torch.nn.BCELoss()
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            adversarial_loss.cuda()

        path = os.path.join('./pre_checkpoints/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer_G = optim.Adam(self.generator.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.args.learning_rate, betas=(0.5, 0.999))

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []
            train_gloss = []
            train_gxloss = []
            self.model.train()
            epoch_time = time.time()
            print(f'len_pre_train:{len(train_loader)}')
            for i, batch_x in tqdm(enumerate(train_loader)):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                # random mask
                B, T, N = batch_x.shape

                # Initialize the mask
                mask = torch.zeros((B, T, N)).to(self.device)

                # Process each column (feature) separately
                for col in range(N):
                    # Generate random masks for each column
                    random_indices = torch.randperm(T * B)[:int((1 - self.args.mask_rate) * T * B)]
                    flat_mask = mask[:, :, col].flatten()
                    flat_mask[random_indices] = 1
                    mask[:, :, col] = flat_mask.reshape(B, T)

                # Generate initial input, setting masked values to 0
                inp = batch_x.masked_fill(mask == 0, 0)
                # Assign values to masked entries: assign the average of the 10% unmasked values in each column to the 90% masked values.
                for col in range(N):
                    column_values = batch_x[:, :, col]
                    column_mask = mask[:, :, col]

                    unmasked_values = column_values[column_mask == 1]

                    if len(unmasked_values) > 0:
                        mean_value = unmasked_values.mean()
                    else:
                        mean_value = 0

                    inp[:, :, col][mask[:, :, col] == 0] = mean_value

                # Standardize
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev

                means1 = inp.mean(1, keepdim=True).detach()
                inp = inp - means1
                stdev1 = torch.sqrt(
                    torch.var(inp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                inp /= stdev1


                # Adversarial ground truths
                valid = Tensor(batch_x.size(0), 1).fill_(1.0)
                fake = Tensor(batch_x.size(0), 1).fill_(0.0)
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # 生成输出
                output = self.generator(inp)
                g_loss = adversarial_loss(self.discriminator(output), valid)
                x_loss = torch.mean(self.args.Lambda * torch.abs(output - batch_x))
                gx_loss = g_loss + x_loss
                gx_loss.backward()
                # g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # --------------------
                optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(self.discriminator(batch_x), valid)
                fake_loss = adversarial_loss(self.discriminator(output.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()

                optimizer_D.step()

                train_loss.append(d_loss.item())
                train_gloss.append(g_loss.item())
                train_gxloss.append(x_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | pre_train_loss: {2:.7f}".format(i + 1, epoch + 1, d_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                torch.cuda.empty_cache()
            print("pre_train: Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_gloss = np.average(train_gloss)
            train_gxloss = np.average(train_gxloss)
            vali_loss = self.pre_vali(vali_data, vali_loader, adversarial_loss)
            test_loss = self.pre_vali(test_data, test_loader, adversarial_loss)
            torch.cuda.empty_cache()
            print("Epoch: {0}, Steps: {1} | pre_Train d_Loss: {2:.7f} g_Loss: {3:.7f} x_Loss: {4:.7f} "
                  " pre_Vali Loss: {5:.7f} pre_Test Loss: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, train_gloss, train_gxloss, vali_loss, test_loss))
            early_stopping(vali_loss, self.generator, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # torch.save(self.generator.state_dict(), path + '/' + 'checkpoint.pth')
            adjust_learning_rate(optimizer_G, epoch + 1, self.args)
            adjust_learning_rate(optimizer_D, epoch + 1, self.args)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.generator.load_state_dict(torch.load(best_model_path))

        return self.generator

    def vali(self, vali_data, vali_loader, criterion, generator):
        total_loss = []
        self.model.eval()
        generator.eval()
        with torch.no_grad():
            print(f'len_vali:{len(vali_loader)}')
            for i, batch_x in tqdm(enumerate(vali_loader)):
                # batch_x = self.zero_out_columns(batch_x)
                batch_x = batch_x.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape

                # Initialize the mask
                mask = torch.zeros((B, T, N)).to(self.device)

                # Process each column (feature) separately
                for col in range(N):
                    # Generate random masks for each column
                    random_indices = torch.randperm(T * B)[:int((1 - self.args.mask_rate) * T * B)]
                    flat_mask = mask[:, :, col].flatten()
                    flat_mask[random_indices] = 1
                    mask[:, :, col] = flat_mask.reshape(B, T)

                # Generate initial input, setting masked values to 0
                inp = batch_x.masked_fill(mask == 0, 0)
                # Assign values to masked entries: assign the average of the 10% unmasked values in each column to the 90% masked values.
                for col in range(N):
                    column_values = batch_x[:, :, col]
                    column_mask = mask[:, :, col]

                    unmasked_values = column_values[column_mask == 1]

                    if len(unmasked_values) > 0:
                        mean_value = unmasked_values.mean()
                    else:
                        mean_value = 0

                    inp[:, :, col][mask[:, :, col] == 0] = mean_value

                # Standardize
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev

                means1 = inp.mean(1, keepdim=True).detach()
                inp = inp - means1
                stdev1 = torch.sqrt(
                    torch.var(inp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                inp /= stdev1


                inp = generator(inp).detach()
                inp = torch.where(mask == 1, batch_x, inp)


                outputs = self.model(inp, None, None, None, mask)

                # with torch.cuda.amp.autocast():
                #     outputs = self.model(inp, None, None, None, mask)
                outputs = outputs.detach()

                batch_x = batch_x * \
                          (stdev[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))
                batch_x = batch_x + \
                          (means[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))

                outputs = outputs * \
                          (stdev1[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))
                outputs = outputs + \
                          (means1[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))

                # pred = outputs.detach().cpu()
                pred = outputs.cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        scaler = torch.cuda.amp.GradScaler()
        # 进行预训练
        generator = self.pre_train(setting)
        generator.eval()
        path0 = os.path.join('./pre_checkpoints/', setting)
        best_model_path = path0 + '/' + 'checkpoint.pth'
        self.generator.load_state_dict(torch.load(best_model_path))
        self.generator.eval()
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = SMAPE()

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f'len_train:{len(train_loader)}')
            for i, batch_x in tqdm(enumerate(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                # batch_x = self.zero_out_columns(batch_x)
                batch_x = batch_x.float().to(self.device)
                B, T, N = batch_x.shape

                # Initialize the mask
                mask = torch.zeros((B, T, N)).to(self.device)

                # Process each column (feature) separately
                for col in range(N):
                    # Generate random masks for each column
                    random_indices = torch.randperm(T * B)[:int((1 - self.args.mask_rate) * T * B)]
                    flat_mask = mask[:, :, col].flatten()
                    flat_mask[random_indices] = 1
                    mask[:, :, col] = flat_mask.reshape(B, T)

                # Generate initial input, setting masked values to 0
                inp = batch_x.masked_fill(mask == 0, 0)
                # Assign values to masked entries: assign the average of the 10% unmasked values in each column to the 90% masked values.
                for col in range(N):
                    column_values = batch_x[:, :, col]
                    column_mask = mask[:, :, col]

                    unmasked_values = column_values[column_mask == 1]

                    if len(unmasked_values) > 0:
                        mean_value = unmasked_values.mean()
                    else:
                        mean_value = 0

                    inp[:, :, col][mask[:, :, col] == 0] = mean_value

                # Standardize
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev

                means1 = inp.mean(1, keepdim=True).detach()
                inp = inp - means1
                stdev1 = torch.sqrt(
                    torch.var(inp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                inp /= stdev1

                self.generator.cuda()
                inp = self.generator(inp).detach()
                inp = torch.where(mask == 1, batch_x, inp)

                outputs = self.model(inp, None, None, None, mask)

                batch_x = batch_x * \
                          (stdev[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))
                batch_x = batch_x + \
                          (means[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))

                outputs = outputs * \
                          (stdev1[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))
                outputs = outputs + \
                          (means1[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))


                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                # scaler.scale(loss).backward()
                # scaler.step(model_optim)
                # scaler.update()
                torch.cuda.empty_cache()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, self.generator)
            test_loss = self.vali(test_data, test_loader, criterion, self.generator)
            torch.cuda.empty_cache()
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []

        path0 = os.path.join('./pre_checkpoints/', setting)
        best_model_path = path0 + '/' + 'checkpoint.pth'
        self.generator.load_state_dict(torch.load(best_model_path))
        self.generator.eval()
        self.model.eval()
        with torch.no_grad():
            print(f'len_test:{len(test_loader)}')
            for i, batch_x in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)

                B, T, N = batch_x.shape

                # Initialize the mask
                mask = torch.zeros((B, T, N)).to(self.device)

                # Process each column (feature) separately
                for col in range(N):
                    # Generate random masks for each column
                    random_indices = torch.randperm(T * B)[:int((1 - self.args.mask_rate) * T * B)]
                    flat_mask = mask[:, :, col].flatten()
                    flat_mask[random_indices] = 1
                    mask[:, :, col] = flat_mask.reshape(B, T)

                # Generate initial input, setting masked values to 0
                inp = batch_x.masked_fill(mask == 0, 0)
                # Assign values to masked entries: assign the average of the 10% unmasked values in each column to the 90% masked values.
                for col in range(N):
                    column_values = batch_x[:, :, col]
                    column_mask = mask[:, :, col]

                    unmasked_values = column_values[column_mask == 1]

                    if len(unmasked_values) > 0:
                        mean_value = unmasked_values.mean()
                    else:
                        mean_value = 0

                    inp[:, :, col][mask[:, :, col] == 0] = mean_value

                # Standardize
                means = batch_x.mean(1, keepdim=True).detach()
                batch_x = batch_x - means
                stdev = torch.sqrt(
                    torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                batch_x /= stdev

                means1 = inp.mean(1, keepdim=True).detach()
                inp = inp - means1
                stdev1 = torch.sqrt(
                    torch.var(inp, dim=1, keepdim=True, unbiased=False) + 1e-5)
                inp /= stdev1

                inp = self.generator(inp).detach()
                inp = torch.where(mask == 1, batch_x, inp)

                outputs = self.model(inp, None, None, None, mask)
                outputs = outputs.detach()

                # with torch.cuda.amp.autocast():
                #     outputs = self.model(inp, None, None, None, mask)

                # eval
                batch_x = batch_x * \
                          (stdev[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))
                batch_x = batch_x + \
                          (means[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))

                outputs = outputs * \
                          (stdev1[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))
                outputs = outputs + \
                          (means1[:, 0, :].unsqueeze(1).repeat(
                              1, self.pred_len + self.seq_len, 1))


                outputs = outputs.cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())


        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('test shape:', preds.shape, trues.shape)


        nmae, nrmse, kl, mspe = metric(preds[masks == 0], trues[masks == 0])
        ndcg_v = ndcg(preds, trues, masks)
        print('nmae:{}, nrmse:{} kl:{} ndcg:{}'.format(nmae, nrmse, kl, ndcg_v))
        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write('nmae:{}, nrmse:{} kl:{} ndcg:{}'.format(nmae, nrmse, kl, ndcg_v))
        f.write('\n')
        f.write('\n')
        f.close()

        return
