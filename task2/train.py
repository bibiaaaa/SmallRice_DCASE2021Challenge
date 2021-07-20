import sys
import csv
import json
import yaml
import datetime
import os
from pathlib import Path
import fcntl

from ignite.contrib.handlers import (
    CustomPeriodicEvent,
    LRScheduler,
    ProgressBar,
    create_lr_scheduler_with_warmup,
)
from sklearn import metrics
from scipy.stats import hmean
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.metrics import Loss
from ignite.utils import apply_to_tensor, convert_tensor
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import dataset
import losses
import models
import utils

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


class Runner(object):
    def __init__(self, config_file, datenow, basepath, **kwargs) -> None:
        super().__init__()
        self.config_parameters = utils.parse_config_or_kwargs(
            config_file, **kwargs)
        torch.manual_seed(self.config_parameters['seed'])
        np.random.seed(self.config_parameters['seed'])
        self.basepath = basepath
        if not os.path.exists(self.basepath):
            os.makedirs(self.basepath)
        with open(os.path.join(self.basepath, 'config.yaml'), 'w') as f:
            yaml.dump(self.config_parameters, f)

    def train_cnn(self, machine_type):
        self.outputpath = os.path.join(self.basepath, machine_type)
        os.makedirs(os.path.join(self.outputpath, 'models'))
        outputfile = os.path.join(self.outputpath, 'log')
        self.logger = utils.getfile_outlogger(outputfile)

        train_df = pd.read_csv(self.config_parameters['train_df_path'],
                               sep='\t')
        eval_source_df = pd.read_csv(
            self.config_parameters['eval_source_df_path'], sep='\t')
        eval_target_df = pd.read_csv(
            self.config_parameters['eval_target_df_path'], sep='\t')

        train_loader = dataset.get_trainloader(
            self.config_parameters['train_hdf5_path'],
            train_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['train_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['train_batch_size'],
            shuffle=True)
        eval_source_loader = dataset.get_evalloader(
            self.config_parameters['eval_source_hdf5_path'],
            eval_source_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['eval_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['eval_batch_size'])
        eval_target_loader = dataset.get_evalloader(
            self.config_parameters['eval_target_hdf5_path'],
            eval_target_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['eval_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['eval_batch_size'])

        model = getattr(models, self.config_parameters['model']['structure'])(
            **self.config_parameters['model']['args']).to(DEVICE)
        optimizer = getattr(torch.optim, self.config_parameters['optimizer'])(
            model.parameters(), **self.config_parameters['optimizer_args'])
        step_scheduler = MultiStepLR(optimizer,
                                     milestones=[30, 60, 90],
                                     gamma=0.5)
        scheduler = LRScheduler(step_scheduler)

        criterion = getattr(losses,
                            self.config_parameters['criterion'])().to(DEVICE)

        self.logger.info(utils.pretty_dict(self.config_parameters))
        self.logger.info(model)

        # trainer and evaluator
        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            input_wavs, _, _, sections = batch
            input_wavs = input_wavs.to(DEVICE)
            sections = sections.to(DEVICE)
            res = {}
            embeddings, outputs = model(
                input_wavs,
                timemask=self.config_parameters['timemask'],
                timeshift=self.config_parameters['timeshift'],
                medianfilter=self.config_parameters['medianfilter'])
            loss = criterion(outputs, sections)
            res['CE_Loss'] = loss.item()
            loss.backward()
            optimizer.step()
            return res

        self.sum_loss = {}

        def sum_training_loss(engine):
            for k, v in engine.state.output.items():
                if k not in self.sum_loss:
                    self.sum_loss[k] = []
                self.sum_loss[k].append(v)

        def log_training_avg_loss(engine):
            avg_loss = {}
            for k, v in self.sum_loss.items():
                avg_loss[k] = np.mean(v)
            self.sum_loss = {}
            for k, v in avg_loss.items():
                self.logger.info(f"Epoch: {engine.state.epoch} {k} : {v}")

        def save_train_model(engine):
            torch.save(
                model.state_dict(),
                os.path.join(self.outputpath, 'models',
                             str(trainer.state.epoch) + '.pth'))

        trainer = Engine(train_step)

        def validation(dataloader):
            eval_result = []
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(dataloader):
                    input_wavs, targets, domain, section = batch
                    input_wavs = input_wavs.to(DEVICE)
                    for index, input_wav in enumerate(input_wavs):
                        _, outputs = model(
                            input_wav,
                            mixup=None,
                            timemask=self.config_parameters['timemask'],
                            timeshift=self.config_parameters['timeshift'],
                            medianfilter=self.config_parameters['medianfilter']
                        )
                        outputs = softmax(outputs, dim=1)
                        outputs = outputs[:, section[index]]
                        eplison = torch.zeros_like(
                            outputs) + sys.float_info.epsilon
                        anomaly_score = (
                            torch.log(torch.maximum(1.0 - outputs, eplison)) -
                            torch.log(torch.max(outputs, eplison))).mean()
                        eval_result.append([
                            anomaly_score, targets[index].item(),
                            domain[index], section[index]
                        ])
            return eval_result

        auc_results = {}
        omega_results = {}

        best_results = {}
        best_results['omega'] = {'omega': 0., 'epoch': 0}

        block_results = []
        line_results = []
        line_results.append(['epoch', 'section', 'domain', 'auc', 'p_auc'])

        def log_validation_results(trainer):
            eval_results = []
            eval_results.extend(validation(eval_source_loader))
            eval_results.extend(validation(eval_target_loader))
            eval_df = pd.DataFrame(
                eval_results,
                columns=['anomaly_score', 'target', 'domain', 'section'])
            epoch = trainer.state.epoch
            auc_results[epoch] = {}
            cal_auc = []
            cal_pauc = []
            block_results.append(['epoch', epoch])
            block_results.append(['section', 'domain', 'auc', 'p_auc'])
            for section in self.config_parameters['eval_sections']:
                auc_results[epoch][section] = {}
                for domain in self.config_parameters['domain']:
                    auc_results[epoch][section][domain] = {}
                    result = eval_df[(eval_df.section == section)
                                     & (eval_df.domain == domain)]
                    y_true = result.target.values
                    y_pred = result.anomaly_score.values
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
                    auc_results[epoch][section][domain]['auc'] = auc
                    auc_results[epoch][section][domain]['p_auc'] = p_auc
                    block_results.append([section, domain, auc, p_auc])
                    line_results.append([epoch, section, domain, auc, p_auc])
                    self.logger.info(
                        f"Validation Results in section {section} domain {domain} - Epoch: {epoch} AUC:{auc:.4f} P_AUC:{p_auc:.4f}"
                    )
                    cal_auc.append(auc)
                    cal_pauc.append(p_auc)
            auc_mean = np.array(cal_auc).mean()
            auc_hmean = hmean(np.array(cal_auc))
            pauc_mean = np.array(cal_pauc).mean()
            pauc_hmean = hmean(np.array(cal_pauc))
            cal_omega = cal_auc + cal_pauc
            omega = hmean(np.array(cal_omega))
            omega_results[epoch] = omega
            if best_results['omega']['omega'] < omega:
                best_results['omega']['omega'] = omega
                best_results['omega']['epoch'] = epoch
                best_results['omega']['auc'] = auc_results[epoch]
                best_results['omega']['auc_mean'] = auc_mean
                best_results['omega']['auc_hmean'] = auc_hmean
                best_results['omega']['pauc_mean'] = pauc_mean
                best_results['omega']['pauc_hmean'] = pauc_hmean

            block_results.append(['Arithmetic mean', '', auc_mean, pauc_mean])
            block_results.append(['omega', omega])
            self.logger.info(f"Epoch: {epoch} Omega: {omega:.4f}")
            # save results
            utils.save_dict(auc_results,
                            os.path.join(self.outputpath, 'auc_result.json'))
            utils.save_dict(omega_results,
                            os.path.join(self.outputpath, 'omega_result.json'))
            utils.save_dict(best_results,
                            os.path.join(self.outputpath, 'best_result'))
            utils.save_csv(
                block_results,
                os.path.join(self.basepath, machine_type + '_block.csv'))
            utils.save_csv(
                line_results,
                os.path.join(self.outputpath, machine_type + '_line.csv'))

        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  sum_training_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  log_training_avg_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_train_model)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  log_validation_results)

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})
        trainer.run(train_loader,
                    max_epochs=self.config_parameters['max_epochs'])

    def train_autoencoder(self, machine_type):
        self.outputpath = os.path.join(self.basepath, machine_type)
        os.makedirs(os.path.join(self.outputpath, 'models'))
        outputfile = os.path.join(self.outputpath, 'log')
        self.logger = utils.getfile_outlogger(outputfile)

        train_df = pd.read_csv(self.config_parameters['train_df_path'],
                               sep='\t')
        eval_source_df = pd.read_csv(
            self.config_parameters['eval_source_df_path'], sep='\t')
        eval_target_df = pd.read_csv(
            self.config_parameters['eval_target_df_path'], sep='\t')

        train_loader = dataset.get_trainloader(
            self.config_parameters['train_hdf5_path'],
            train_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['train_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['train_batch_size'],
            shuffle=True)
        eval_source_loader = dataset.get_evalloader(
            self.config_parameters['eval_source_hdf5_path'],
            eval_source_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['eval_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['eval_batch_size'])
        eval_target_loader = dataset.get_evalloader(
            self.config_parameters['eval_target_hdf5_path'],
            eval_target_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['eval_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['eval_batch_size'])

        # prepare model, optimzer, criterion
        model = getattr(models, self.config_parameters['model']['structure'])(
            **self.config_parameters['model']['args']).to(DEVICE)
        optimizer = getattr(torch.optim, self.config_parameters['optimizer'])(
            model.parameters(), **self.config_parameters['optimizer_args'])
        criterion = getattr(losses,
                            self.config_parameters['criterion'])().to(DEVICE)

        self.logger.info(utils.pretty_dict(self.config_parameters))
        self.logger.info(model)

        # trainer and evaluator
        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            input_wavs, _, _, _ = batch
            input_wavs = input_wavs.to(DEVICE)
            outputs, inputs = model(input_wavs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            return loss.item()

        self.sum_loss = []

        def sum_training_loss(engine):
            self.sum_loss.append(engine.state.output)

        def log_training_avg_loss(engine):
            avg_loss = np.mean(self.sum_loss)
            self.sum_loss = []
            self.logger.info(f"Epoch: {engine.state.epoch} Loss: {avg_loss}")

        def save_train_model(engine):
            torch.save(
                model.state_dict(),
                os.path.join(self.outputpath, 'models',
                             str(trainer.state.epoch) + '.pth'))

        trainer = Engine(train_step)

        def validation(dataloader):
            eval_result = []
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(dataloader):
                    input_wavs, targets, domain, section = batch
                    input_wavs = input_wavs.to(DEVICE)
                    for index, input_wav in enumerate(input_wavs):
                        outputs, inputs = model(input_wav)
                        anomaly_score = (outputs - inputs).pow(2).mean().item()
                        eval_result.append([
                            anomaly_score, targets[index].item(),
                            domain[index], section[index]
                        ])
            return eval_result

        auc_results = {}
        omega_results = {}

        best_results = {}
        for section in self.config_parameters['eval_sections']:
            best_results[section] = {}
            for domain in self.config_parameters['domain']:
                best_results[section][domain] = {
                    'auc_epoch': 0,
                    'auc': 0.,
                    'p_auc_epoch': 0,
                    'p_auc': 0.
                }
        best_results['omega'] = {'omega': 0., 'epoch': 0}

        block_results = []
        line_results = []
        line_results.append(['epoch', 'section', 'domain', 'auc', 'p_auc'])

        def log_validation_results(trainer):
            eval_results = []
            eval_results.extend(validation(eval_source_loader))
            eval_results.extend(validation(eval_target_loader))
            eval_df = pd.DataFrame(
                eval_results,
                columns=['anomaly_score', 'target', 'domain', 'section'])
            epoch = trainer.state.epoch
            auc_results[epoch] = {}
            cal_auc = []
            cal_pauc = []
            block_results.append(['epoch', epoch])
            block_results.append(['section', 'domain', 'auc', 'p_auc'])
            for section in self.config_parameters['eval_sections']:
                auc_results[epoch][section] = {}
                for domain in self.config_parameters['domain']:
                    auc_results[epoch][section][domain] = {}
                    result = eval_df[(eval_df.section == section)
                                     & (eval_df.domain == domain)]
                    y_true = result.target.values
                    y_pred = result.anomaly_score.values
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
                    auc_results[epoch][section][domain]['auc'] = auc
                    auc_results[epoch][section][domain]['p_auc'] = p_auc
                    if auc > best_results[section][domain]['auc']:
                        best_results[section][domain]['auc'] = auc
                        best_results[section][domain]['auc_epoch'] = epoch

                    if p_auc > best_results[section][domain]['p_auc']:
                        best_results[section][domain]['p_auc'] = p_auc
                        best_results[section][domain]['p_auc_epoch'] = epoch
                    block_results.append([section, domain, auc, p_auc])
                    line_results.append([epoch, section, domain, auc, p_auc])
                    self.logger.info(
                        f"Validation Results in section {section} domain {domain} - Epoch: {epoch} AUC:{auc:.4f} P_AUC:{p_auc:.4f}"
                    )
                    cal_auc.append(auc)
                    cal_pauc.append(p_auc)
            auc_mean = np.array(cal_auc).mean()
            pauc_mean = np.array(cal_pauc).mean()
            cal_omega = cal_auc + cal_pauc
            omega = hmean(np.array(cal_omega))
            omega_results[epoch] = omega
            if best_results['omega']['omega'] < omega:
                best_results['omega']['omega'] = omega
                best_results['omega']['epoch'] = epoch
                best_results['omega']['auc'] = auc_results[epoch]
            block_results.append(['Arithmetic mean', '', auc_mean, pauc_mean])
            block_results.append(['omega', omega])
            self.logger.info(f"Epoch: {epoch} Omega: {omega:.4f}")
            # save results
            utils.save_dict(auc_results,
                            os.path.join(self.outputpath, 'auc_result.json'))
            utils.save_dict(omega_results,
                            os.path.join(self.outputpath, 'omega_result.json'))
            utils.save_dict(best_results,
                            os.path.join(self.outputpath, 'best_result'))
            utils.save_csv(
                block_results,
                os.path.join(self.basepath, machine_type + '_block.csv'))
            utils.save_csv(
                line_results,
                os.path.join(self.outputpath, machine_type + '_line.csv'))

        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  sum_training_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  log_training_avg_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_train_model)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  log_validation_results)

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})
        trainer.run(train_loader,
                    max_epochs=self.config_parameters['max_epochs'])

    def train_cl(self, machine_type):
        self.outputpath = os.path.join(self.basepath, machine_type)
        os.makedirs(os.path.join(self.outputpath, 'models'))
        outputfile = os.path.join(self.outputpath, 'log')
        self.logger = utils.getfile_outlogger(outputfile)

        train_df = pd.read_csv(self.config_parameters['train_df_path'],
                               sep='\t')
        eval_source_df = pd.read_csv(
            self.config_parameters['eval_source_df_path'], sep='\t')
        eval_target_df = pd.read_csv(
            self.config_parameters['eval_target_df_path'], sep='\t')

        train_loader = dataset.get_trainloader(
            self.config_parameters['train_hdf5_path'],
            train_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['train_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['train_batch_size'],
            shuffle=True)
        eval_source_loader = dataset.get_evalloader(
            self.config_parameters['eval_source_hdf5_path'],
            eval_source_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['eval_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['eval_batch_size'])
        eval_target_loader = dataset.get_evalloader(
            self.config_parameters['eval_target_hdf5_path'],
            eval_target_df,
            machine_type,
            self.config_parameters['domain'],
            self.config_parameters['eval_sections'],
            self.config_parameters['acoustic_paras'],
            num_workers=self.config_parameters['num_workers'],
            batch_size=self.config_parameters['eval_batch_size'])

        model = getattr(models, self.config_parameters['model']['structure'])(
            **self.config_parameters['model']['args']).to(DEVICE)
        optimizer = getattr(torch.optim, self.config_parameters['optimizer'])(
            model.parameters(), **self.config_parameters['optimizer_args'])
        step_scheduler = MultiStepLR(optimizer,
                                     milestones=[30, 60, 90],
                                     gamma=0.5)
        scheduler = LRScheduler(step_scheduler)

        cnn_criterion = getattr(
            losses, self.config_parameters['cnn_criterion'])().to(DEVICE)
        ae_criterion = getattr(
            losses, self.config_parameters['ae_criterion'])().to(DEVICE)
        cl_criterion = getattr(
            losses, self.config_parameters['cl_criterion'])().to(DEVICE)

        def train_step(engine, batch):
            model.train()
            optimizer.zero_grad()
            input_wavs, _, _, sections = batch
            input_wavs = input_wavs.to(DEVICE)
            sections = sections.to(DEVICE)
            res = {}
            ae_input, ae_hidden_output, cnn_hidden_output, ae_output, cnn_output = model(
                input_wavs,
                medianfilter=self.config_parameters['medianfilter'])
            ae_loss = ae_criterion(ae_output, ae_input)
            cnn_loss = cnn_criterion(cnn_output, sections)
            cl_loss = cl_criterion(
                torch.stack([ae_hidden_output, cnn_hidden_output], dim=1))
            loss = ae_loss + cnn_loss + 0.0005 * cl_loss
            res['AE Loss'] = ae_loss.item()
            res['CNN Loss'] = cnn_loss.item()
            res['CL Loss'] = cl_loss.item()
            res['Total Loss'] = loss.item()
            loss.backward()
            optimizer.step()
            return res

        self.sum_loss = {}

        def sum_training_loss(engine):
            for k, v in engine.state.output.items():
                if k not in self.sum_loss:
                    self.sum_loss[k] = []
                self.sum_loss[k].append(v)

        def log_training_avg_loss(engine):
            avg_loss = {}
            for k, v in self.sum_loss.items():
                avg_loss[k] = np.mean(v)
            self.sum_loss = {}
            for k, v in avg_loss.items():
                self.logger.info(f"Epoch: {engine.state.epoch} {k} : {v}")

        def save_train_model(engine):
            torch.save(
                model.state_dict(),
                os.path.join(self.outputpath, 'models',
                             str(trainer.state.epoch) + '.pth'))

        trainer = Engine(train_step)

        def validation(dataloader):
            eval_result = []
            model.eval()
            with torch.no_grad():
                for _, batch in enumerate(dataloader):
                    input_wavs, targets, domain, section = batch
                    input_wavs = input_wavs.to(DEVICE)
                    for index, input_wav in enumerate(input_wavs):
                        outputs = model(input_wav,
                                        medianfilter=self.
                                        config_parameters['medianfilter'])
                        outputs = softmax(outputs, dim=1)
                        outputs = outputs[:, section[index]]
                        eplison = torch.zeros_like(
                            outputs) + sys.float_info.epsilon
                        anomaly_score = (
                            torch.log(torch.maximum(1.0 - outputs, eplison)) -
                            torch.log(torch.max(outputs, eplison))).mean()
                        eval_result.append([
                            anomaly_score, targets[index].item(),
                            domain[index], section[index]
                        ])
            return eval_result

        auc_results = {}
        omega_results = {}

        best_results = {}
        best_results['omega'] = {'omega': 0., 'epoch': 0}

        block_results = []
        line_results = []
        line_results.append(['epoch', 'section', 'domain', 'auc', 'p_auc'])

        def log_validation_results(trainer):
            eval_results = []
            eval_results.extend(validation(eval_source_loader))
            eval_results.extend(validation(eval_target_loader))
            eval_df = pd.DataFrame(
                eval_results,
                columns=['anomaly_score', 'target', 'domain', 'section'])
            epoch = trainer.state.epoch
            auc_results[epoch] = {}
            cal_auc = []
            cal_pauc = []
            block_results.append(['epoch', epoch])
            block_results.append(['section', 'domain', 'auc', 'p_auc'])
            for section in self.config_parameters['eval_sections']:
                auc_results[epoch][section] = {}
                for domain in self.config_parameters['domain']:
                    auc_results[epoch][section][domain] = {}
                    result = eval_df[(eval_df.section == section)
                                     & (eval_df.domain == domain)]
                    y_true = result.target.values
                    y_pred = result.anomaly_score.values
                    auc = metrics.roc_auc_score(y_true, y_pred)
                    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
                    auc_results[epoch][section][domain]['auc'] = auc
                    auc_results[epoch][section][domain]['p_auc'] = p_auc
                    block_results.append([section, domain, auc, p_auc])
                    line_results.append([epoch, section, domain, auc, p_auc])
                    self.logger.info(
                        f"Validation Results in section {section} domain {domain} - Epoch: {epoch} AUC:{auc:.4f} P_AUC:{p_auc:.4f}"
                    )
                    cal_auc.append(auc)
                    cal_pauc.append(p_auc)
            auc_mean = np.array(cal_auc).mean()
            auc_hmean = hmean(np.array(cal_auc))
            pauc_mean = np.array(cal_pauc).mean()
            pauc_hmean = hmean(np.array(cal_pauc))
            cal_omega = cal_auc + cal_pauc
            omega = hmean(np.array(cal_omega))
            omega_results[epoch] = omega
            if best_results['omega']['omega'] < omega:
                best_results['omega']['omega'] = omega
                best_results['omega']['epoch'] = epoch
                best_results['omega']['auc'] = auc_results[epoch]
                best_results['omega']['auc_mean'] = auc_mean
                best_results['omega']['auc_hmean'] = auc_hmean
                best_results['omega']['pauc_mean'] = pauc_mean
                best_results['omega']['pauc_hmean'] = pauc_hmean

            block_results.append(['Arithmetic mean', '', auc_mean, pauc_mean])
            block_results.append(['omega', omega])
            self.logger.info(f"Epoch: {epoch} Omega: {omega:.4f}")
            # save results
            utils.save_dict(auc_results,
                            os.path.join(self.outputpath, 'auc_result.json'))
            utils.save_dict(omega_results,
                            os.path.join(self.outputpath, 'omega_result.json'))
            utils.save_dict(best_results,
                            os.path.join(self.outputpath, 'best_result'))
            utils.save_csv(
                block_results,
                os.path.join(self.basepath, machine_type + '_block.csv'))
            utils.save_csv(
                line_results,
                os.path.join(self.outputpath, machine_type + '_line.csv'))

        trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)
        trainer.add_event_handler(Events.ITERATION_COMPLETED,
                                  sum_training_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  log_training_avg_loss)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, save_train_model)
        trainer.add_event_handler(Events.EPOCH_COMPLETED,
                                  log_validation_results)

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})
        trainer.run(train_loader,
                    max_epochs=self.config_parameters['max_epochs'])


if __name__ == '__main__':
    from fire import Fire
    Fire(Runner)
