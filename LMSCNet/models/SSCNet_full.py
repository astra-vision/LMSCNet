import torch.nn as nn
import torch.nn.functional as F
import torch


class SSCNet_full(nn.Module):
  '''
  # Class coded from caffe model https://github.com/shurans/sscnet/blob/master/test/demo.txt
  '''

  def __init__(self, class_num):
    '''
    SSCNet architecture
    :param N: number of classes to be predicted (i.e. 12 for NYUv2)
    '''
    super().__init__()

    self.nbr_classes = class_num

    self.conv1_1 =  nn.Conv3d(1, 16, kernel_size=7, padding=3, stride=2, dilation=1)  # conv(16, 7, 2, 1)

    self.reduction2_1 = nn.Conv3d(16, 32, kernel_size=1, padding=0, stride=1, dilation=1)  # conv(32, 1, 1, 1)

    self.conv2_1 =  nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(32, 3, 1, 1)
    self.conv2_2 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(32, 3, 1, 1)

    self.pool2 = nn.MaxPool3d(2)  # pooling

    self.reduction3_1 = nn.Conv3d(64, 64, kernel_size=1, padding=0, stride=1, dilation=1)  # conv(64, 1, 1, 1)

    self.conv3_1 = nn.Conv3d(32, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)
    self.conv3_2 = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)

    self.conv3_3 = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)
    self.conv3_4 = nn.Conv3d(64, 64, kernel_size=3, padding=1, stride=1, dilation=1)  # conv(64, 3, 1, 1)

    self.conv3_5 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)
    self.conv3_6 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)

    self.conv3_7 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)
    self.conv3_8 = nn.Conv3d(64, 64, kernel_size=3, padding=2, stride=1, dilation=2)  # dilated(64, 3, 1, 2)

    self.conv4_1 = nn.Conv3d(192, 128, kernel_size=1, padding=0, stride=1, dilation=1)  # conv(128, 1, 1, 1)
    self.conv4_2 = nn.Conv3d(128, 128, kernel_size=1, padding=0, stride=1, dilation=1)  # conv(128, 1, 1, 1)

    self.deconv_classes = nn.ConvTranspose3d(128, self.nbr_classes, kernel_size=4, padding=0, stride=4)

    return

  def forward(self, x):

    input = x['3D_OCCUPANCY'].permute(0, 1, 3, 2, 4) # Reshaping [bs, H, W, D]

    out = F.relu(self.conv1_1(input))
    out_add_1 = self.reduction2_1(out)
    out = F.relu((self.conv2_1(out)))
    out = F.relu(out_add_1 + self.conv2_2(out))

    out = self.pool2(out)

    out = F.relu(self.conv3_1(out))
    out_add_2 = self.reduction3_1(out)
    out = F.relu(out_add_2 + self.conv3_2(out))

    out_add_3 = self.conv3_3(out)
    out = self.conv3_4(F.relu(out_add_3))
    out_res_1 = F.relu(out_add_3 + out)

    out_add_4 = self.conv3_5(out_res_1)
    out = self.conv3_6(F.relu(out_add_4))
    out_res_2 = F.relu(out_add_4 + out)

    out_add_5 = self.conv3_7(out_res_2)
    out = self.conv3_8(F.relu(out_add_5))
    out_res_3 = F.relu(out_add_5 + out)

    out = torch.cat((out_res_3, out_res_2, out_res_1), 1)

    out = F.relu(self.conv4_1(out))
    out = F.relu(self.conv4_2(out))

    out = self.deconv_classes(out)

    out = out.permute(0, 1, 3, 2, 4)  # [bs, C, H, W, D] -> [bs, C, W, H, D]

    scores = {'pred_semantic_1_1': out}

    return scores

  def weights_initializer(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight)
      nn.init.zeros_(m.bias)

  def weights_init(self):
    self.apply(self.weights_initializer)

  def get_parameters(self):
    return self.parameters()

  def compute_loss(self, scores, data):
    '''
    :param: prediction: the predicted tensor, must be [BS, C, W, H, D]
    '''

    target = data['3D_LABEL']['1_1']   # [bs, C, W, H, D]
    device, dtype = target.device, target.dtype
    class_weights = torch.ones(self.nbr_classes).to(device=device, dtype=dtype)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255, reduction='none').to(device=device)

    # Reduction is none to be able to apply the 2N data balancing after. The mean will be calculated then...
    loss_1_1 = criterion(scores['pred_semantic_1_1'], data['3D_LABEL']['1_1'].long())
    # F.cross_entropy(prediction, target.long(), weight=class_weights, ignore_index=255, reduction='none')

    # For SSCNet all classes have same weight and their weight is done by their 2N Data Balancing
    weight_db = self.get_data_balance_2N(data)
    # Calculate loss weighted by 2N data balancing
    # Remember target == 255 is ignored for the loss, this has to be considered for the mean..!
    # Also we are considering loss on only 2N free/occluded voxels, which is given by weight_db mask.
    # We do not consider 2N in occluded voxels only since is Lidar data, all scene needs to be completed.
    # Including outside FoV
    loss_1_1 = torch.sum(loss_1_1*weight_db) / torch.sum((weight_db != 1) & (target != 255))

    loss = {'total': loss_1_1, 'semantic_1_1': loss_1_1}

    return loss

  def get_data_balance_2N(self, data):
    '''
    Get a weight tensor for the loss computing. The weight tensor will ignore unknown voxels on target tensor
    (label==255). A random under sampling on free voxels with a relation 2:1 between free:occupied is obtained.
    The subsampling is done by considering only free occluded voxels. Explanation in SSCNet article
    (https://arxiv.org/abs/1611.08974)

    There is a discrepancy between data balancing explained on article and data balancing implemented on code
    https://github.com/shurans/sscnet/issues/33

    The subsampling will be done in all free voxels.. Not occluded only.. As Martin Gabarde did on TS3D.. There is
    a problem on what is explained for data balancing on SSCNet
    '''

    batch_target = data['3D_LABEL']['1_1']
    weight = torch.zeros_like(batch_target)
    for i, target in enumerate(batch_target):
      nbr_occupied = torch.sum((target > 0) & (target < 255))
      nbr_free = torch.sum(target == 0)
      free_indices = torch.where(target == 0)  # Indices of free voxels on target
      subsampling = torch.randint(nbr_free, (2 * nbr_occupied,))  # Random subsampling 2*nbr_occupied in range nbr_free
      mask = (free_indices[0][subsampling], free_indices[1][subsampling], free_indices[2][subsampling])  # New mask
      weight[i][mask] = 1  # Subsampled free voxels to be considered (2N)
      weight[i][(target > 0) & (target < 255)] = 1  # Occupied voxels

    # Returning weight that has N occupied voxels and 2N free voxels...
    return weight

  def get_target(self, data):
    '''
    Return the target to use for evaluation of the model
    '''
    return {'1_1': data['3D_LABEL']['1_1']}

  def get_scales(self):
    '''
    Return scales needed to train the model
    '''
    scales = ['1_1']
    return scales

  def get_validation_loss_keys(self):
    return ['total', 'semantic_1_1']

  def get_train_loss_keys(self):
    return ['total', 'semantic_1_1']