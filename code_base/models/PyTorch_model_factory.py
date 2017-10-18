from code_base.models.PyTorch_fcn import FeatureResNet, SegResNet, iou
import torch
from torchvision import models
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from code_base.tools.logger import Logger
from datetime import datetime

from matplotlib import pyplot as plt
import os
import sys
plt.switch_backend('agg')  # Allow plotting when running remotely


# Build the model
class Model_Factory():
    def __init__(self, cf):
        # If we load from a pretrained model

        pretrained_net = FeatureResNet()
        pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
        self.net = SegResNet(cf.num_classes, pretrained_net).cuda()
        # Set the loss criterion
        # TODO: set the weight of the loss
        # TODO: confirm loss
        #self.crit = nn.BCELoss().cuda()
        #self.crit = nn.CrossEntropyLoss().cuda()
        self.crit = nn.NLLLoss2d().cuda()
        self.num_classes = cf.num_classes
        self.exp_dir = cf.savepath + '___' + datetime.now().strftime('%a, %d %b %Y-%m-%d %H:%M:%S')
        os.mkdir(self.exp_dir)
        # Enable log file
        self.log_file = os.path.join(self.exp_dir, "logfile.log")
        sys.stdout = Logger(self.log_file)

        # we print the configuration file here so that the configuration is traceable
        self.cf = cf
        print(help(cf))

        # Construct optimiser
        if cf.load_trained_model:
            print("Load from pretrained_model weight: "+cf.train_model_path)
            self.net.load_state_dict(torch.load(cf.train_model_path))
        params_dict = dict(self.net.named_parameters())
        params = []
        for key, value in params_dict.items():
            if 'bn' in key:
                # No weight decay on batch norm
                params += [{'params': [value], 'weight_decay': 0}]
            elif '.bias' in key:
                # No weight decay plus double learning rate on biases
                params += [{'params': [value], 'lr': 2 * cf.learning_rate, 'weight_decay': 0}]
            else:
                params += [{'params': [value]}]
        if cf.optimizer == 'rmsprop':
            self.optimiser = optim.RMSprop(params, lr=cf.learning_rate, momentum=cf.momentum, weight_decay=cf.weight_decay)
        elif cf.optimizer == 'sgd':
            self.optimiser = optim.SGD(params, lr=cf.learning_rate, momentum=cf.momentum, weight_decay=cf.weight_decay)
        self.scores, self.mean_scores = [], []

    def train(self, train_loader, epoch):
        self.net.train()
        for i, (input, target_one_hot, target) in enumerate(train_loader):
            self.optimiser.zero_grad()
            input, target = Variable(input.cuda(async=True)), Variable(target.cuda(async=True))
            #output = F.sigmoid(self.net(input))
            # TODO: check softmax loss
            #output = F.softmax(self.net(input))
            output = F.log_softmax(self.net(input))
            self.loss = self.crit(output, target)
            print(epoch, i, self.loss.data[0])
            self.loss.backward()
            self.optimiser.step()

    def test(self, val_loader, epoch):
        self.net.eval()
        total_ious = []
        for i, (input, _, target) in enumerate(val_loader):
            input, target = Variable(input.cuda(async=True), volatile=True), Variable(target.cuda(async=True), volatile=True)
            output = F.log_softmax(self.net(input))
            b, _, h, w = output.size()
            pred = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes).max(1)[1].view(b, h, w)
            total_ious.append(iou(pred, target, self.num_classes))

            # Save images
            if i % 100 == 0:
                pred = pred.data.cpu()
                pred_remapped = pred.clone()
                # Convert to full labels
                for k, v in self.cf.train_to_full.items():
                    pred_remapped[pred == k] = v
                # Convert to colour image
                pred = pred_remapped
                pred_colour = torch.zeros(b, 3, h, w)
                for k, v in self.cf.full_to_colour.items():
                    pred_r = torch.zeros(b, 1, h, w)
                    pred_r[(pred == k)] = v[0]
                    pred_g = torch.zeros(b, 1, h, w)
                    pred_g[(pred == k)] = v[1]
                    pred_b = torch.zeros(b, 1, h, w)
                    pred_b[(pred == k)] = v[2]
                    pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))
                save_image(pred_colour[0].float().div(255), os.path.join(self.exp_dir, str(epoch) + '_' + str(i) + '.png'))

        # Calculate average IoU
        total_ious = torch.Tensor(total_ious).transpose(0, 1)
        ious = torch.Tensor(self.num_classes - 1)
        for i, class_iou in enumerate(total_ious):
            ious[i] = class_iou[class_iou == class_iou].mean()  # Calculate mean, ignoring NaNs
        print(ious, ious.mean())
        self.scores.append(ious)

        # Save weights and scores
        torch.save(self.net.state_dict(), os.path.join(self.exp_dir, str(epoch) + '_net.pth'))
        torch.save(self.scores, os.path.join(self.exp_dir, 'scores.pth'))

        # Plot scores
        self.mean_scores.append(ious.mean())
        es = list(range(len(self.mean_scores)))
        plt.plot(es, self.mean_scores, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.savefig(os.path.join(self.exp_dir, 'ious.png'))
        plt.close()
