import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor


class ContractingBlock(nn.Module, ):
    def __init__(self, in_channels, out_channels, method="mp"):
        super(ContractingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if method == "mp":
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        elif method == "sc":
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        else:
            self.downsample = None
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.downsample:
            skip = x  # store the output for the skip connection
            x = self.downsample(x)
            return x, skip
        
        return x

class ExpandingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, method='tr'):
        super(ExpandingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        if method == "tr":
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        elif method == "up":
            self.upsample =  nn.Upsample(scale_factor=2, mode='nearest')
        else:
            raise ValueError("invalid input for expand_method")

        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, skip):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        
        x = self.upsample(x)
        
        # concatenate the skip connection
        x = torch.cat((x, skip), dim=1)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UNet(pl.LightningModule):
    def __init__(
            self,
            loss_fn: str="ce",
            contract_method: str="mp",
            expand_method: str="tr",
            in_channels: int=3,
            out_channels: int=3):
        """
        parameters
        ----------

        loss_fn: str
            valid values are
            - `ce` for cross entropy  
            - `dice` for dice loss

        contraction_method: str
            valid values are
            - `mp` for max pooling with 2*2 kernels
            - `sc` for strided convolution

        expand_method: str
            valid values are 
            - `up` - upsample convolution
            - `tr` - transpose convolution

        """
        super(UNet, self).__init__()
        
        self.loss_fn = loss_fn
        self.contract1 = ContractingBlock(in_channels, 64, contract_method)
        self.contract2 = ContractingBlock(64, 128, contract_method)
        self.contract3 = ContractingBlock(128, 256, contract_method)
        self.contract4 = ContractingBlock(256, 512, "")
        
        self.expand1 = ExpandingBlock(512, 256, expand_method)
        self.expand2 = ExpandingBlock(256, 128, expand_method)
        self.expand3 = ExpandingBlock(128, 64, expand_method)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        x, skip1 = self.contract1(x) # 32 -> 16
        x, skip2 = self.contract2(x) # 16  -> 8
        x, skip3 = self.contract3(x) # 8 -> 4
        x = self.contract4(x) # 4 -> 2
        
        # Expanding path
        x = self.expand1(x, skip3) # 8
        x = self.expand2(x, skip2)
        x = self.expand3(x, skip1)
        x = self.final_conv(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1E-4)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        masks_pred = self(images)
        if self.loss_fn == "ce":
            loss_fn_ = nn.CrossEntropyLoss()
            loss = loss_fn_(
                masks_pred, targets.squeeze(dim=1).long())
        elif self.loss_fn == "dice":
            pred_ohe = F.one_hot(
                masks_pred.argmax(dim=1),
                num_classes=masks_pred.shape[1])\
                .permute(0, 3, 1, 2).float()
            pred_ohe.requires_grad = True
            target_ohe = F.one_hot(
                targets.squeeze(dim=1),
                num_classes=masks_pred.shape[1])\
                .permute(0, 3, 1, 2).long()
            loss = self.dice_coeff(
                pred_ohe,
                target_ohe.long())
            # loss.backward(retain_graph=True)
        else:
            raise ValueError("invalid loss_fn values")
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def dice_loss(pred, target):
        smooth = 1e-5
        
        # flatten predictions and targets
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return 1 - dice

    @staticmethod
    def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        input = input.flatten(0, 1)
        target = target.flatten(0, 1)
        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

