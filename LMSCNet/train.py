import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sys

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from LMSCNet.common.seed import seed_all
from LMSCNet.common.config import CFG
from LMSCNet.common.dataset import get_dataset
from LMSCNet.common.model import get_model
from LMSCNet.common.logger import get_logger
from LMSCNet.common.optimizer import build_optimizer, build_scheduler
from LMSCNet.common.io_tools import dict_to
from LMSCNet.common.metrics import Metrics
import LMSCNet.common.checkpoint as checkpoint


def parse_args():
  parser = argparse.ArgumentParser(description='LMSCNet training')
  parser.add_argument(
    '--cfg',
    dest='config_file',
    default='',
    metavar='FILE',
    help='path to config file',
    type=str,
  )
  parser.add_argument(
    '--dset_root',
    dest='dataset_root',
    default=None,
    metavar='DATASET',
    help='path to dataset root folder',
    type=str,
  )
  args = parser.parse_args()
  return args


def train(model, optimizer, scheduler, dataset, _cfg, start_epoch, logger, tbwriter):
  """
  Train a model using the PyTorch Module API.
  Inputs:
  - model: A PyTorch Module giving the model to train.
  - optimizer: An Optimizer object we will use to train the model
  - scheduler: Scheduler for learning rate decay if used
  - dataset: The dataset to load files
  - _cfg: The configuration dictionary read from config file
  - start_epoch: The epoch at which start the training (checkpoint)
  - logger: The logger to save info
  - tbwriter: The tensorboard writer to save plots
  Returns: Nothing, but prints model accuracies during training.
  """

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  dtype = torch.float32  # Tensor type to be used

  # Moving optimizer and model to used device
  model = model.to(device=device)
  for state in optimizer.state.values():
    for k, v in state.items():
      if isinstance(v, torch.Tensor):
        state[k] = v.to(device)

  dset = dataset['train']

  nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']
  nbr_iterations = len(dset)  # number of iterations depends on batchs size

  # Defining metrics class and initializing them..
  metrics = Metrics(dset.dataset.nbr_classes, nbr_iterations, model.get_scales())
  metrics.reset_evaluator()
  metrics.losses_track.set_validation_losses(model.get_validation_loss_keys())
  metrics.losses_track.set_train_losses(model.get_train_loss_keys())

  for epoch in range(start_epoch, nbr_epochs+1):

    logger.info('=> =============== Epoch [{}/{}] ==============='.format(epoch, nbr_epochs))
    logger.info('=> Reminder - Output of routine on {}'.format(_cfg._dict['OUTPUT']['OUTPUT_PATH']))

    # Print learning rate
    # for param_group in optimizer.param_groups:
    logger.info('=> Learning rate: {}'.format(scheduler.get_lr()[0]))

    model.train()  # put model to training mode

    # for t, (data, indices) in enumerate(dataset['train']):
    for t, (data, indices) in enumerate(dset):

      data = dict_to(data, device, dtype)

      scores = model(data)

      loss = model.compute_loss(scores, data)

      # Zero out the gradients.
      optimizer.zero_grad()
      # Backward pass: gradient of loss wr. each model parameter.
      loss['total'].backward()
      # update parameters of model by gradients.
      optimizer.step()

      if _cfg._dict['SCHEDULER']['FREQUENCY'] == 'iteration':
        scheduler.step()

      for l_key in loss:
        tbwriter.add_scalar('train_loss_batch/{}'.format(l_key), loss[l_key].item(), len(dset) * (epoch-1) + t)
      # Updating batch losses to then get mean for epoch loss
      metrics.losses_track.update_train_losses(loss)

      if (t + 1) % _cfg._dict['TRAIN']['SUMMARY_PERIOD'] == 0:
        loss_print = '=> Epoch [{}/{}], Iteration [{}/{}], Learn Rate: {}, Train Losses: '\
          .format(epoch, nbr_epochs, t+1, len(dset), scheduler.get_lr()[0])
        for key in loss.keys(): loss_print += '{} = {:.6f},  '.format(key, loss[key])
        logger.info(loss_print[:-3])

      metrics.add_batch(prediction=scores, target=model.get_target(data))

    for l_key in metrics.losses_track.train_losses:
      tbwriter.add_scalar('train_loss_epoch/{}'.format(l_key),
                          metrics.losses_track.train_losses[l_key].item()/metrics.losses_track.train_iteration_counts,
                          epoch - 1)
    tbwriter.add_scalar('lr/lr', scheduler.get_lr()[0], epoch - 1)

    epoch_loss = metrics.losses_track.train_losses['total']/metrics.losses_track.train_iteration_counts

    for scale in metrics.evaluator.keys():
      tbwriter.add_scalar('train_performance/{}/mIoU'.format(scale), metrics.get_semantics_mIoU(scale).item(), epoch-1)
      tbwriter.add_scalar('train_performance/{}/IoU'.format(scale), metrics.get_occupancy_IoU(scale).item(), epoch-1)
      # tbwriter.add_scalar('train_performance/{}/Precision'.format(scale), metrics.get_occupancy_Precision(scale).item(), epoch-1)
      # tbwriter.add_scalar('train_performance/{}/Recall'.format(scale), metrics.get_occupancy_Recall(scale).item(), epoch-1)
      # tbwriter.add_scalar('train_performance/{}/F1'.format(scale), metrics.get_occupancy_F1(scale).item(), epoch-1)

    logger.info('=> [Epoch {} - Total Train Loss = {}]'.format(epoch, epoch_loss))
    for scale in metrics.evaluator.keys():
      loss_scale = metrics.losses_track.train_losses['semantic_{}'.format(scale)].item()/metrics.losses_track.train_iteration_counts
      logger.info('=> [Epoch {} - Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                  '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'
                  .format(epoch, scale, loss_scale,
                          metrics.get_semantics_mIoU(scale).item(),
                          metrics.get_occupancy_IoU(scale).item(),
                          metrics.get_occupancy_Precision(scale).item(),
                          metrics.get_occupancy_Recall(scale).item(),
                          metrics.get_occupancy_F1(scale).item()))

    logger.info('=> Epoch {} - Training set class-wise IoU:'.format(epoch))
    for i in range(1, metrics.nbr_classes):
      class_name  = dset.dataset.dataset_config['labels'][dset.dataset.dataset_config['learning_map_inv'][i]]
      class_score = metrics.evaluator['1_1'].getIoU()[1][i]
      logger.info('    => IoU {}: {:.6f}'.format(class_name, class_score))

    # Reset evaluator for validation...
    metrics.reset_evaluator()

    checkpoint_info = validate(model, dataset['val'], _cfg, epoch, logger, tbwriter, metrics)

    # Reset evaluator and losses for next epoch...
    metrics.reset_evaluator()
    metrics.losses_track.restart_train_losses()
    metrics.losses_track.restart_validation_losses()

    if _cfg._dict['SCHEDULER']['FREQUENCY'] == 'epoch':
      scheduler.step()

    # Save checkpoints
    for k in checkpoint_info.keys():
      checkpoint_path = os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'chkpt', k)
      _cfg._dict['STATUS'][checkpoint_info[k]] = checkpoint_path
      checkpoint.save(checkpoint_path, model, optimizer, scheduler, epoch, _cfg._dict)

    # Save checkpoint if current epoch matches checkpoint period
    if epoch % _cfg._dict['TRAIN']['CHECKPOINT_PERIOD'] == 0:
      checkpoint_path = os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'chkpt', str(epoch).zfill(2))
      checkpoint.save(checkpoint_path, model, optimizer, scheduler, epoch, _cfg._dict)

    # Update config file
    _cfg.update_config(resume=True)

  return metrics.best_metric_record


def validate(model, dset, _cfg, epoch, logger, tbwriter, metrics):

  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  dtype = torch.float32  # Tensor type to be used

  nbr_epochs = _cfg._dict['TRAIN']['EPOCHS']

  logger.info('=> Passing the network on the validation set...')

  model.eval()

  with torch.no_grad():

    for t, (data, indices) in enumerate(dset):

      data = dict_to(data, device, dtype)

      scores = model(data)

      loss = model.compute_loss(scores, data)

      for l_key in loss:
        tbwriter.add_scalar('validation_loss_batch/{}'.format(l_key), loss[l_key].item(), len(dset) * (epoch-1) + t)
      # Updating batch losses to then get mean for epoch loss
      metrics.losses_track.update_validaiton_losses(loss)

      if (t + 1) % _cfg._dict['VAL']['SUMMARY_PERIOD'] == 0:
        loss_print = '=> Epoch [{}/{}], Iteration [{}/{}], Train Losses: '.format(epoch, nbr_epochs, t+1, len(dset))
        for key in loss.keys(): loss_print += '{} = {:.6f},  '.format(key, loss[key])
        logger.info(loss_print[:-3])

      metrics.add_batch(prediction=scores, target=model.get_target(data))

    for l_key in metrics.losses_track.validation_losses:
      tbwriter.add_scalar('validation_loss_epoch/{}'.format(l_key),
                          metrics.losses_track.validation_losses[l_key].item()/metrics.losses_track.validation_iteration_counts,
                          epoch - 1)

    epoch_loss = metrics.losses_track.validation_losses['total']/metrics.losses_track.validation_iteration_counts

    for scale in metrics.evaluator.keys():
      tbwriter.add_scalar('validation_performance/{}/mIoU'.format(scale), metrics.get_semantics_mIoU(scale).item(), epoch-1)
      tbwriter.add_scalar('validation_performance/{}/IoU'.format(scale), metrics.get_occupancy_IoU(scale).item(), epoch-1)
      # tbwriter.add_scalar('validation_performance/{}/Precision'.format(scale), metrics.get_occupancy_Precision(scale).item(), epoch-1)
      # tbwriter.add_scalar('validation_performance/{}/Recall'.format(scale), metrics.get_occupancy_Recall(scale).item(), epoch-1)
      # tbwriter.add_scalar('validation_performance/{}/F1'.format(scale), metrics.get_occupancy_F1(scale).item(), epoch-1)

    logger.info('=> [Epoch {} - Total Validation Loss = {}]'.format(epoch, epoch_loss))
    for scale in metrics.evaluator.keys():
      loss_scale = metrics.losses_track.validation_losses['semantic_{}'.format(scale)].item()/metrics.losses_track.train_iteration_counts
      logger.info('=> [Epoch {} - Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                  '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'
                  .format(epoch, scale, loss_scale,
                          metrics.get_semantics_mIoU(scale).item(),
                          metrics.get_occupancy_IoU(scale).item(),
                          metrics.get_occupancy_Precision(scale).item(),
                          metrics.get_occupancy_Recall(scale).item(),
                          metrics.get_occupancy_F1(scale).item()))

    logger.info('=> Epoch {} - Validation set class-wise IoU:'.format(epoch))
    for i in range(1, metrics.nbr_classes):
      class_name  = dset.dataset.dataset_config['labels'][dset.dataset.dataset_config['learning_map_inv'][i]]
      class_score = metrics.evaluator['1_1'].getIoU()[1][i]
      logger.info('    => {}: {:.6f}'.format(class_name, class_score))

    checkpoint_info = {}

    if epoch_loss < _cfg._dict['OUTPUT']['BEST_LOSS']:
      logger.info('=> Best loss on validation set encountered: ({} < {})'.
                  format(epoch_loss, _cfg._dict['OUTPUT']['BEST_LOSS']))
      _cfg._dict['OUTPUT']['BEST_LOSS'] = epoch_loss.item()
      checkpoint_info['best-loss'] = 'BEST_LOSS'

    mIoU_1_1 = metrics.get_semantics_mIoU('1_1')
    IoU_1_1  = metrics.get_occupancy_IoU('1_1')
    if mIoU_1_1 > _cfg._dict['OUTPUT']['BEST_METRIC']:
      logger.info('=> Best metric on validation set encountered: ({} > {})'.
                  format(mIoU_1_1, _cfg._dict['OUTPUT']['BEST_METRIC']))
      _cfg._dict['OUTPUT']['BEST_METRIC'] = mIoU_1_1.item()
      checkpoint_info['best-metric'] = 'BEST_METRIC'
      metrics.update_best_metric_record(mIoU_1_1, IoU_1_1, epoch_loss.item(), epoch)

    checkpoint_info['last'] = 'LAST'

    return checkpoint_info


def main():

  # https://github.com/pytorch/pytorch/issues/27588
  torch.backends.cudnn.enabled = False

  seed_all(0)

  args = parse_args()

  train_f = args.config_file
  dataset_f = args.dataset_root

  # Read train configuration file
  _cfg = CFG()
  _cfg.from_config_yaml(train_f)

  # Replace dataset path in config file by the one passed by argument
  if dataset_f is not None:
    _cfg._dict['DATASET']['ROOT_DIR'] = dataset_f

  # Create writer for Tensorboard
  tbwriter = SummaryWriter(log_dir=os.path.join(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'metrics'))

  # Setting the logger to print statements and also save them into logs file
  logger = get_logger(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'logs_train.log')

  logger.info('============ Training routine: "%s" ============\n' % train_f)
  dataset = get_dataset(_cfg)

  logger.info('=> Loading network architecture...')
  model = get_model(_cfg, dataset['train'].dataset)
  if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model = model.module

  logger.info('=> Loading optimizer...')
  optimizer = build_optimizer(_cfg, model)
  scheduler = build_scheduler(_cfg, optimizer)

  model, optimizer, scheduler, epoch = checkpoint.load(model, optimizer, scheduler, _cfg._dict['STATUS']['RESUME'],
                                                       _cfg._dict['STATUS']['LAST'], logger)

  best_record = train(model, optimizer, scheduler, dataset, _cfg, epoch, logger, tbwriter)

  logger.info('=> ============ Network trained - all epochs passed... ============')

  logger.info('=> [Best performance: Epoch {} - mIoU = {} - IoU {}]'.format(best_record['epoch'], best_record['mIoU'], best_record['IoU']))

  logger.info('=> Writing config file in output folder - deleting from config files folder')
  _cfg.finish_config()
  logger.info('=> Training routine completed...')

  exit()


if __name__ == '__main__':
  main()