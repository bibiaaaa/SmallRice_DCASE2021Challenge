import datetime
from pathlib import Path
import uuid
from typing import Union, List, Tuple

from fire import Fire
from ignite.contrib.metrics import ROC_AUC
from ignite.contrib.handlers import ProgressBar, create_lr_scheduler_with_warmup
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    EarlyStopping,
    global_step_from_engine,
)
from ignite.metrics import Loss, Precision, Recall, RunningAverage, Accuracy
import numpy as np
import torch
import tqdm
import yaml

import dataset
import models
import utils
import metrics
import losses

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)

# Some defaults for non-specified arguments in yaml
DEFAULT_TRAIN_ARGS = {
    'outputpath': 'experiments',
    'feature_args': {
        'n_mels': 64,
        'hop_size': 160,
        'sr': 16000,
        'win_size': 512,
        'f_min': 0
    },
    'train_weak_data': 'data/weak_train.tsv',
    'train_syn_data': 'data/synthetic_train.tsv',
    'cv_syn_data': 'data/synthetic_validation.tsv',
    'cv_syn_duration': 'data/synthetic_validation_duration.tsv',
    'test_data': 'data/validation.tsv',
    'unlabeled_data': 'data/unlabeled_train.tsv',
    'sampler': 'random',  # Can also be "balanced"
    'loss': 'BCELoss',  # default is BCEloss.
    'loss_args': {},
    'student': 'CDur',
    'student_args': {},
    'usesyn': False,  # Default do not use synthethic dataset for training
    'useunb': False,
    'usepred': False,  #Using predicted labels from another model
    'consistency_criterion':
    'BCELoss',  # default use bce for consistency training
    'batch_sizes': (32, 32, 64),
    'unlabeled_aug': {
        'mask_length': 16000,
        'prob': 0.5,
        'gain_min': -20.,
        'gain_max': 10.,
    },
    'warmup_iters': 20,
    'max_grad_norm': 1.0,
    'mixup': None,
    'epoch_length': None,
    'mixup_rate':
    1.0,  # When to mixup samples, 1.0 means if mixup > 0, then do always mixup
    'num_workers': 2,  # Number of dataset loaders
    'step_size': 1000,  # When to reduce LR
    'spectransforms': {},  #Default no augmentation
    'wavtransforms': {},
    'early_stop': 15,
    'save':
    'best',  # if saving is done just every epoch, Otherwise any value is saving at test
    'epochs': 200,
    'n_saved': 1,
    'optimizer': 'AdamW',
    'optimizer_args': {
        'lr': 0.001,
    },
}


def parse_config_or_kwargs(config_file,
                           default_args=DEFAULT_TRAIN_ARGS,
                           **override_kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    arguments = dict(yaml_config, **override_kwargs)
    # In case some arguments were not passed, replace with default ones
    for key, value in default_args.items():
        arguments.setdefault(key, value)
    return arguments


def has_and_exists(item: str, adict: dict):
    return item in adict and adict[item]


def transfer_to_device(batch):
    return (x.to(DEVICE, non_blocking=True)
            if isinstance(x, torch.Tensor) else x for x in batch)


class Runner(object):
    def __init__(self, seed=42):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train_sed(self,
                  config: str = 'configs/train_supervised_baseline.yaml',
                  **override_kwargs):
        config_parameters = parse_config_or_kwargs(config, **override_kwargs)
        outputdir = Path(
            Path(config_parameters['outputpath']) / Path(config).stem /
            config_parameters['student'],
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m')}_{uuid.uuid1().hex}"
        )
        outputdir.mkdir(exist_ok=True, parents=True)
        logger = utils.getlogger(outputdir / 'train.log')
        logger.info(f"Storing experiment in {outputdir}")
        logger.info(f"Running on device {DEVICE}")
        for k, v in config_parameters.items():
            logger.info(f"{k} : {v}")

        train_weak_df, train_syn_df, cv_syn_df, unlabeled_df = (
            utils.read_labels(input_data)
            for input_data in (config_parameters['train_weak_data'],
                               config_parameters['train_syn_data'],
                               config_parameters['cv_syn_data'],
                               config_parameters['unlabeled_data']))

        encoder = utils.LabelEncoder(
            train_weak_df['event_labels'].sum())  #.sum does np.concatenate
        train_weak_df, cv_weak_df = utils.split_train_cv(train_weak_df, 0.9)
        logger.info(encoder)
        weak_bs, syn_bs, unlabel_bs = config_parameters['batch_sizes']

        train_weak_ds = dataset.WeakHDF5Dataset(train_weak_df, encoder)
        train_syn_ds = dataset.StrongHDF5Dataset(
            train_syn_df,
            encoder,
            smooth_ramp=config_parameters['strong_ramp'],
            **config_parameters['feature_args'])
        unlabeled_ds = dataset.UnlabeledHDF5Dataset(unlabeled_df)

        cv_weak_ds = dataset.WeakHDF5Dataset(cv_weak_df, encoder)
        cv_syn_ds = dataset.StrongHDF5Dataset(
            cv_syn_df, encoder, **config_parameters['feature_args'])

        cvdataloader = dataset.getdataloader(torch.utils.data.ConcatDataset(
            [cv_syn_ds, cv_weak_ds]),
                                             batch_size=4,
                                             num_workers=2)
        weak_sampler = torch.utils.data.RandomSampler(train_weak_ds)
        syn_sampler = torch.utils.data.RandomSampler(train_syn_ds)
        unb_sampler = torch.utils.data.RandomSampler(unlabeled_ds)

        if config_parameters['sampler'] == 'balanced':
            weak_sampler = dataset.BalancedSampler(
                train_weak_df['event_labels'], encoder)
            syn_sampler = dataset.BalancedSampler(train_syn_df['event_label'],
                                                  encoder)
        logger.info(f"Using sampler {config_parameters['sampler']}")
        lam = 1.  # weight for unsupervised UDA
        mixup_alpha = config_parameters['mixup']
        mixup_coef = 2 if mixup_alpha else 1
        mixup_rate = config_parameters['mixup_rate']

        traindata = {
            'weak':
            dataset.getdataloader(
                train_weak_ds,
                batch_size=weak_bs * mixup_coef,
                num_workers=config_parameters['num_workers'],
                sampler=weak_sampler,
            )
        }
        if has_and_exists('usesyn', config_parameters):
            logger.info("Using Synthetic data")
            traindata['syn'] = dataset.getdataloader(
                train_syn_ds,
                batch_size=syn_bs,
                num_workers=config_parameters['num_workers'],
                sampler=syn_sampler)
        if has_and_exists('useunb', config_parameters):
            logger.info("Using Unlabaled data")
            traindata['unlabel'] = dataset.getdataloader(
                unlabeled_ds,
                batch_size=unlabel_bs,
                num_workers=config_parameters['num_workers'],
                sampler=unb_sampler)
            unlabeled_augment = models.UnlabeledAugmentor(
                **config_parameters['unlabeled_aug'])

        traindataloader = dataset.getmultidatasetdataloader(**traindata)
        logger.info(f"Using training data: {list(traindata.keys())}")

        spectransforms = utils.parse_spectransforms(
            config_parameters['spectransforms'])

        student = getattr(models, config_parameters['student'])(
            spectransforms=spectransforms, **config_parameters['student_args'])
        student_optimizer = getattr(
            torch.optim,
            config_parameters['optimizer'],
        )(student.parameters(), **config_parameters['optimizer_args'])

        if has_and_exists('pretrained', config_parameters):
            logger.info(
                f"Loading pretrained model {config_parameters['pretrained']}")
            pretrained_model = torch.load(config_parameters['pretrained'],
                                          map_location='cpu')['model']
            student = utils.load_pretrained(student, pretrained_model)
        student = student.train().to(DEVICE)

        criterion = getattr(
            losses,
            config_parameters['loss'])(**config_parameters['loss_args'])

        do_strong_consistency = False
        if has_and_exists('strong_consistency', config_parameters):
            do_strong_consistency = True
        if has_and_exists('consistency_criterion', config_parameters):
            consistency_criterion = getattr(
                torch.nn, config_parameters['consistency_criterion'])()
            logger.info(f"Using Consistency Criterion {consistency_criterion}")

        def _mixup_weights(size, alpha):
            return torch.tensor(np.random.beta(alpha, alpha, size=size),
                                device=DEVICE,
                                dtype=torch.float32)

        def compute_UDA_loss(engine,
                             unsup_x,
                             unsup_aug_x,
                             unsup_frame_aug_x=None):
            # Unsupervised part
            with torch.no_grad():
                unsup_orig_clip_pred, unsup_orig_frame_pred = student(unsup_x)
                unsup_orig_clip_pred = unsup_orig_clip_pred.detach()
                unsup_orig_frame_pred = unsup_orig_frame_pred.detach()

            unsup_aug_clip_pred, unsup_aug_frame_pred = student(unsup_aug_x)
            if unsup_frame_aug_x != None:
                pass
            consistency_loss = consistency_criterion(unsup_aug_clip_pred,
                                                     unsup_orig_clip_pred)
            if do_strong_consistency:
                consistency_loss_strong = consistency_criterion(
                    unsup_aug_frame_pred, unsup_orig_frame_pred)
                consistency_loss += consistency_loss_strong

            return consistency_loss

        def _train_batch(engine, batch):
            student.train()
            with torch.enable_grad():
                student_optimizer.zero_grad(set_to_none=True)
                weak_data, target, _ = transfer_to_device(batch['weak'])

                mixup_lambda = None
                do_mixup = bool(torch.empty(1).uniform_(0, 1) < mixup_rate)
                if mixup_alpha and do_mixup:
                    mixup_lambda = _mixup_weights(
                        len(weak_data) // 2, mixup_alpha)
                    target = utils.mixup(target, mixup_lambda)
                #Mixup is done in the model
                clip_pred, frame_pred = student(weak_data, mixup=mixup_lambda)
                loss = criterion(clip_pred, target)
                if 'syn' in batch:
                    syn_data, stronk_targets, weak_targets, _ = transfer_to_device(
                        batch['syn'])
                    if mixup_alpha and do_mixup:
                        mixup_lambda = _mixup_weights(
                            len(syn_data) // 2, mixup_alpha)
                        stronk_targets = utils.mixup(stronk_targets,
                                                     mixup_lambda)
                        weak_targets = utils.mixup(weak_targets, mixup_lambda)
                    clip_pred, frame_pred = student(syn_data, mixup_lambda)
                    # loss += criterion(clip_pred, weak_targets)
                    loss += criterion(frame_pred, stronk_targets)
                if 'unlabel' in batch:
                    # Compute transformations on CPU
                    unb_data, _ = batch['unlabel']
                    aug_unb_data = unlabeled_augment(unb_data.clone())
                    unb_data, aug_unb_data = transfer_to_device(
                        (unb_data, aug_unb_data))
                    uda_loss = compute_UDA_loss(engine, unb_data, aug_unb_data)
                    loss = loss + lam * uda_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), config_parameters['max_grad_norm'])
                student_optimizer.step()
                return loss.item()

        def _inference(engine, batch):
            student.eval()
            output = {'strong': None, 'weak': None, 'intersection': None}
            with torch.no_grad():
                data, *targets, filenames = transfer_to_device(batch)
                clip_out, frame_out = student(data)
                # Stronk labels
                if len(targets) == 2:
                    output['strong'] = (frame_out, targets[0])
                    output['intersection'] = (frame_out, filenames)
                else:
                    output['weak'] = (clip_out, targets[0])
                return output

        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)

        def _get_lr():
            for param_group in student_optimizer.param_groups:
                return param_group['lr']

        def compute_metrics(engine):
            inferencer = inference_engine
            inferencer.run(cvdataloader)
            results = inferencer.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
            ] + [f"{metric} {results[metric]:<5.2f}"
                 for metric in results] + [f"LR {_get_lr():.4f}"]
            logger.info(" ".join(output_str_list))

        checkpoint_saver = Checkpoint(
            {
                'model': student,
                'optimizer': student_optimizer,
                'config': utils._DictWrapper(
                    config_parameters
                ),  # Just to have state_dict and load_state_dict methods
                'encoder': encoder,
            },
            DiskSaver(str(outputdir)),
            n_saved=1,
            global_step_transform=global_step_from_engine(train_engine),
            filename_prefix='best',
            score_function=Checkpoint.get_default_score_fn('Obj_Metric', 1.0),
            score_name='Obj_Metric',
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            student_optimizer,
            step_size=config_parameters['step_size'],
            gamma=0.5)
        scheduler = create_lr_scheduler_with_warmup(
            scheduler,
            warmup_start_value=0.0,
            warmup_duration=config_parameters['warmup_iters'])
        train_engine.add_event_handler(Events.ITERATION_STARTED, scheduler)

        earlystop_handler = EarlyStopping(
            patience=config_parameters['early_stop'],
            score_function=Checkpoint.get_default_score_fn('Obj_Metric', 1.0),
            trainer=train_engine)
        inference_engine.add_event_handler(Events.COMPLETED, earlystop_handler)

        ProgressBar().attach(train_engine,
                             output_transform=lambda x: {'loss': x})
        ProgressBar().attach(inference_engine)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics)
        inference_engine.add_event_handler(Events.COMPLETED, checkpoint_saver)

        def transform_to_05_output(output):
            if output is not None:
                return torch.round(output[0]), output[1]
            return None

        def f1_score(label):
            pre = losses.Precision(
                output_transform=lambda x: transform_to_05_output(x[label]),
                device=DEVICE,
                average=False)
            rec = losses.Recall(
                output_transform=lambda x: transform_to_05_output(x[label]),
                device=DEVICE,
                average=False)
            return (pre * rec * 2 / (pre + rec)).mean()

        def loss_with_label(label):
            return losses.Loss(criterion, output_transform=lambda x: x[label])

        def mAP_with_label(label):
            return losses.MeanAveragePrecision(
                output_transform=lambda x: x[label])

        def dprime_with_label(label):
            return losses.Dprime(output_transform=lambda x: x[label])

        def intersection_f1():
            # Note that this takes a while...
            return losses.Intersection_F1(
                encoder=encoder,
                validation_data_file=config_parameters['cv_syn_data'],
                validation_duration_file=config_parameters['cv_syn_duration'],
                output_transform=lambda x: x['intersection'])

        evaluation_metrics = {
            'Weak_F1': f1_score('weak'),
            'Strong_F1': f1_score('strong'),
            'F1': (f1_score('weak') + f1_score('strong')) / 2,
            'Total_loss':
            (loss_with_label('weak') + loss_with_label('strong')) / 2,
            'Weak_mAP': mAP_with_label('weak'),
            'Weak_Dprime': dprime_with_label('weak'),
            # 'Obj_Metric': (dprime_with_label('weak') + f1_score('weak')) / 2 +
            # 2 * intersection_f1(),
            'Obj_Metric': f1_score('weak') + intersection_f1(),
        }
        for name, metric in evaluation_metrics.items():
            metric.attach(inference_engine, name)

        epoch_length = len(train_weak_df) // weak_bs if not config_parameters[
            'epoch_length'] else config_parameters['epoch_length']

        train_engine.run(
            traindataloader,
            max_epochs=config_parameters['epochs'],
            epoch_length=epoch_length,
        )

        return outputdir

    def run(self, config, **override_kwargs):
        outputdir = self.train_sed(config, **override_kwargs)
        self.evaluate(outputdir, **override_kwargs)

    def _forward_eval(self,
                      exp_path: Path,
                      logger,
                      validation_data=None,
                      **kwargs):
        logger.info(f"Running evaluation for {exp_path}")
        logger.info(f"Running on {DEVICE}")
        exp_path = Path(exp_path)
        if exp_path.is_file():
            trained_dump = torch.load(exp_path, map_location='cpu')
            exp_path = exp_path.parent  # Set upper dir as result dir
        else:
            trained_dump = torch.load(next(exp_path.glob('*checkpoint*')),
                                      map_location='cpu')
        trained_model_params = trained_dump['model']

        encoder, config = trained_dump['encoder'], trained_dump['config']
        logger.info(encoder)
        if has_and_exists('sepsed', config):
            sepsed_params = config['sepsed']
            sed_model = getattr(
                models,
                sepsed_params['sed_model'])(**sepsed_params['sed_model_args'])

            model = getattr(models, sepsed_params['sep_model'])(
                sed_model=sed_model).to(DEVICE).eval()
        else:
            model = getattr(models, config['student'])(
                len(encoder), **config['student_args']).to(DEVICE).eval()
        model.load_state_dict(trained_model_params)
        validation_data = config[
            'test_data'] if validation_data == None else validation_data
        logger.info(f"Validating on {validation_data}")
        validation_agg_df = utils.read_labels(validation_data).dropna()
        ground_truth_df = utils.read_labels(
            validation_data, aggregate=False).dropna().reset_index(drop=True)
        eval_dataloader = dataset.getdataloader(dataset.StrongHDF5Dataset(
            validation_agg_df, encoder),
                                                batch_size=1,
                                                num_workers=3,
                                                shuffle=False)
        weak_true, weak_pred = [], []
        frame_preds, filenames = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(eval_dataloader):
                data, strong_target, weak_target, filename = transfer_to_device(
                    batch)
                clip_out, frame_out = model(data)
                weak_pred.append(clip_out.cpu().numpy())
                weak_true.append(weak_target.cpu().numpy())
                frame_preds.append(frame_out.detach().cpu().squeeze(0))
                filenames.append(filename[0])
        return weak_true, weak_pred, frame_preds, filenames, encoder, ground_truth_df, validation_data

    def _forward_submit(self, exp_path, eval_data):
        exp_path = Path(exp_path)
        data_df = utils.read_labels(eval_data)
        if exp_path.is_file():
            trained_dump = torch.load(exp_path, map_location='cpu')
            exp_path = exp_path.parent  # Set upper dir as result dir
        else:
            trained_dump = torch.load(next(exp_path.glob('*checkpoint*')),
                                      map_location='cpu')
        trained_model_params = trained_dump['model']

        encoder, config = trained_dump['encoder'], trained_dump['config']
        if has_and_exists('sepsed', config):
            sepsed_params = config['sepsed']
            sed_model = getattr(
                models,
                sepsed_params['sed_model'])(**sepsed_params['sed_model_args'])

            model = getattr(models, sepsed_params['sep_model'])(
                sed_model=sed_model).to(DEVICE).eval()
        else:
            model = getattr(models, config['student'])(
                len(encoder), **config['student_args']).to(DEVICE).eval()
        model.load_state_dict(trained_model_params)
        dataloader = dataset.getdataloader(
            dataset.UnlabeledHDF5Dataset(data_df),
            batch_size=1,
            num_workers=3,
            shuffle=False)
        frame_preds = []
        filenames = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                data, filename = transfer_to_device(batch)
                clip_out, frame_out = model(data)
                filenames.append(filename[0])
                frame_preds.append(frame_out.detach().cpu().squeeze(0))
        return frame_preds, filenames, encoder

    def submit(
        self,
        *exppaths: List[Path],
        outputdir="submissions/",
        eval_data="data/eval.tsv",
        time_resolution=0.01,
        adaptive_factor: int = 3,
        median_filter=None,
    ):
        logger = utils.getlogger()
        outputdir = Path(outputdir)
        outputdir.mkdir(parents=True, exist_ok=True)
        frame_pred_all = []
        for i, exp_path in enumerate(exppaths):
            frame_preds, filenames, encoder = self._forward_submit(
                exp_path, eval_data)
            frame_pred_all.append(frame_preds)
        buf = frame_pred_all[0]
        for single_out in frame_pred_all[1:]:
            for i in range(len(single_out)):
                buf[i] += single_out[i]
        for i in range(len(buf)):
            buf[i] /= len(exppaths)
        frame_preds = buf
        if median_filter == 'adaptive':
            logger.debug(
                f"Using adaptive post-processing with factor {adaptive_factor}"
            )
            frame_preds = utils.adaptive_median_filter(frame_preds,
                                                       encoder,
                                                       factor=adaptive_factor)
        else:
            frame_preds = utils.median_filter(frame_preds, median_filter)
        test_thresholds = np.arange(1 / (50 * 2), 1, 1 / 50)
        thresholded_predictions = utils.frame_preds_to_chunk_preds(
            frame_preds,
            filenames,
            encoder,
            thresholds=test_thresholds,
            frame_resolution=time_resolution,
        )
        logger.info(f"Dumping results to {outputdir}")
        outputdir.mkdir(exist_ok=True, parents=True)
        for k in thresholded_predictions.keys():
            thresholded_predictions[k]['filename'] = thresholded_predictions[
                k]['filename'].apply(lambda x: Path(x).name)
            thresholded_predictions[k].to_csv(
                outputdir / f"{k:.3f}.tsv",
                sep="\t",
                index=False,
            )


    def evaluate_ensemble(
        self,
        *exppaths: List[Path],
        validation_data: Path = None,
        validation_duration: Path = 'data/validation_duration.tsv',
        outputdir="evaluation_ensemble",
        n_thresholds: int = 50,
        time_resolution: float = 0.01,
        median_filter: Union[int, str] = None,
        adaptive_factor: float = 3,
    ):
        logger = utils.getlogger()
        # Das ist hier richtig hingepfuscht
        outputdir = Path(outputdir)
        outputdir.mkdir(parents=True, exist_ok=True)
        weak_true_all, weak_pred_all, frame_pred_all, filenames_all = [], [], [], []
        for i, exp_path in enumerate(exppaths):
            weak_true, weak_pred, frame_preds, filenames, encoder, ground_truth_df, ground_truth_data = self._forward_eval(
                exp_path, logger=logger, validation_data=validation_data)
            weak_pred_all.append(np.concatenate(weak_pred))
            frame_pred_all.append(frame_preds)
            if i == 0:
                weak_true_all.append(np.concatenate(weak_true))
        buf = frame_pred_all[0]
        for single_out in frame_pred_all[1:]:
            for i in range(len(single_out)):
                buf[i] += single_out[i]
        for i in range(len(buf)):
            buf[i] /= len(exppaths)
        frame_preds = buf
        if median_filter == 'adaptive':
            frame_preds = utils.adaptive_median_filter(frame_preds,
                                                       encoder,
                                                       factor=adaptive_factor)
        else:
            frame_preds = utils.median_filter(frame_preds, median_filter)

        weak_pred = np.stack(weak_pred_all, -1).mean(-1)

        weak_true = np.concatenate(weak_true_all)
        test_thresholds = np.arange(1 / (n_thresholds * 2), 1,
                                    1 / n_thresholds)

        auc, mAP, dprime = metrics.clip_level_metrics(weak_true, weak_pred)
        logger.info(
            f"Clip-level Eval: Auc: {auc:.3f} mAP: {mAP:.3f} dprime: {dprime:.3f}"
        )
        # Prediction for simple thresholding
        thresholded_pred_05 = utils.frame_preds_to_chunk_preds(
            frame_preds,
            filenames,
            encoder,
            thresholds=(0.5, ),
            frame_resolution=time_resolution,
        )

        thresholded_pred_05[0.5].to_csv(outputdir / 'predictions_05.txt',
                                        index=False,
                                        sep='\t')
        event_res, segment_res = metrics.compute_sed_eval_metrics(
            thresholded_pred_05[0.5], ground_truth_df)
        logger.info(event_res)
        logger.info(segment_res)
        with open(outputdir / 'event_results.txt', 'w') as wp:
            print(event_res, file=wp)

        thresholded_predictions = utils.frame_preds_to_chunk_preds(
            frame_preds,
            filenames,
            encoder,
            thresholds=test_thresholds,
            frame_resolution=time_resolution,
        )
        intersection_f1_macro = metrics.compute_per_intersection_macro_f1(
            thresholded_predictions, ground_truth_data, validation_duration)
        logger.info(f"Intersection F1 macro: {100*intersection_f1_macro:.2f}")

        psds_score_scenario1 = metrics.compute_psds_from_operating_points(
            thresholded_predictions,
            ground_truth_data,
            validation_duration,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
            save_dir=outputdir / 'scenario1')
        logger.info(f"PSDS Score Scenario 1: {psds_score_scenario1:.4f}")

        psds_score_scenario2 = metrics.compute_psds_from_operating_points(
            thresholded_predictions,
            ground_truth_data,
            validation_duration,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=outputdir / 'scenario2')
        logger.info(f"PSDS Score Scenario 2: {psds_score_scenario2:.4f}")
        best_test_result = max(psds_score_scenario1, psds_score_scenario2)
        logger.info(f"Best Test result {best_test_result:.4f}")
        with open(outputdir / 'results.txt', 'w') as wp:
            event_metrics = event_res.results_class_wise_average_metrics(
            )['f_measure']
            segment_metrics = segment_res.results_class_wise_average_metrics(
            )['f_measure']
            print(f"Auc: {auc:.3f} mAP: {mAP:.3f} dprime: {dprime:.3f}",
                  file=wp)
            print(
                f"Event-F1: {event_metrics['f_measure']}\nEvent-Precision: {event_metrics['precision']}\nEvent-Rec: {event_metrics['recall']}",
                file=wp)
            print(
                f"Segment-F1: {segment_metrics['f_measure']}\nSegment-Precision: {segment_metrics['precision']}\nSegment-Rec: {segment_metrics['recall']}",
                file=wp)
            print(f"Intersection-F1: {intersection_f1_macro}", file=wp)
            print(f"PSDS-Scenario 1: {psds_score_scenario1:.4f}", file=wp)
            print(f"PSDS-Scenario 2: {psds_score_scenario2:.4f}", file=wp)
            print(f"PSDS-Best: {best_test_result:.4f}", file=wp)

    def evaluate(self,
                 exp_path: Path,
                 validation_data: Path = None,
                 validation_duration: Path = 'data/validation_duration.tsv',
                 n_thresholds: int = 50,
                 time_resolution:float = 0.01,
                 median_filter: Union[int, str] = 1 ,
                 adaptive_factor:float = 3, # If adaptive median filtering is used
                 **kwargs
                 ):
        logger = utils.getlogger()
        exp_path = Path(exp_path)
        #Overwrite validation data
        weak_true, weak_pred, frame_preds, filenames, encoder, ground_truth_df, validation_data = self._forward_eval(
            exp_path, logger=logger, validation_data=validation_data)
        test_thresholds = np.arange(1 / (n_thresholds * 2), 1,
                                    1 / n_thresholds)
        if median_filter == 'adaptive':
            frame_preds = utils.adaptive_median_filter(frame_preds, encoder, factor=adaptive_factor)
        else:
            frame_preds = utils.median_filter(frame_preds, median_filter)


        auc, mAP, dprime = metrics.clip_level_metrics(
            np.concatenate(weak_true), np.concatenate(weak_pred))
        logger.info(
            f"Clip-level Eval: Auc: {auc:.3f} mAP: {mAP:.3f} dprime: {dprime:.3f}"
        )
        # Prediction for simple thresholding
        thresholded_pred_05 = utils.frame_preds_to_chunk_preds(
            frame_preds,
            filenames,
            encoder,
            thresholds=(0.5, ),
            frame_resolution=time_resolution)
        thresholded_pred_05[0.5].to_csv(exp_path / 'predictions_05.txt',
                           index=False,
                           sep='\t')
        event_res, segment_res = metrics.compute_sed_eval_metrics(
            thresholded_pred_05[0.5], ground_truth_df)
        logger.info(event_res)
        logger.info(segment_res)
        with open(exp_path / 'event_results.txt', 'w') as wp:
            print(event_res, file=wp)
        thresholded_predictions = utils.frame_preds_to_chunk_preds(frame_preds,
                                               filenames,
                                               encoder,
                                               thresholds=test_thresholds,
                                               frame_resolution=time_resolution,
                                               )
        intersection_f1_macro = metrics.compute_per_intersection_macro_f1(
            thresholded_predictions, validation_data, validation_duration)
        logger.info(f"Intersection F1 macro: {100*intersection_f1_macro:.2f}")

        psds_score_scenario1 = metrics.compute_psds_from_operating_points(
            thresholded_predictions,
            validation_data,
            validation_duration,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            alpha_ct=0,
            alpha_st=1,
            save_dir=exp_path / 'scenario1')
        logger.info(f"PSDS Score Scenario 1: {psds_score_scenario1:.4f}")

        psds_score_scenario2 = metrics.compute_psds_from_operating_points(
            thresholded_predictions,
            validation_data,
            validation_duration,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            save_dir=exp_path / 'scenario2')
        logger.info(f"PSDS Score Scenario 2: {psds_score_scenario2:.4f}")
        best_test_result = max(psds_score_scenario1, psds_score_scenario2)
        logger.info(f"Best Test result {best_test_result:.4f}")
        with open(exp_path / 'results.txt', 'w') as wp:
            event_metrics = event_res.results_class_wise_average_metrics(
            )['f_measure']
            segment_metrics = segment_res.results_class_wise_average_metrics(
            )['f_measure']
            print(f"Auc: {auc:.3f} mAP: {mAP:.3f} dprime: {dprime:.3f}",
                  file=wp)
            print(
                f"Event-F1: {event_metrics['f_measure']}\nEvent-Precision: {event_metrics['precision']}\nEvent-Rec: {event_metrics['recall']}",
                file=wp)
            print(
                f"Segment-F1: {segment_metrics['f_measure']}\nSegment-Precision: {segment_metrics['precision']}\nSegment-Rec: {segment_metrics['recall']}",
                file=wp)
            print(f"Intersection-F1: {intersection_f1_macro}", file=wp)
            print(f"PSDS-Scenario 1: {psds_score_scenario1:.4f}", file=wp)
            print(f"PSDS-Scenario 2: {psds_score_scenario2:.4f}", file=wp)
            print(f"PSDS-Best: {best_test_result:.4f}", file=wp)






if __name__ == "__main__":
    Fire(Runner)
