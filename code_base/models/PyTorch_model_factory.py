from code_base.models.PyTorch_fcn import FeatureResNet, SegResNet, iou
from code_base.models.PyTorch_drn import drn_c_26, drn_d_22, DRNSeg, DRNSegF
from code_base.models.PyTorch_PredictModels import LSTM_ManyToMany, LSTM_To_FC
from code_base.tools.PyTorch_model_training import calc_seq_err_robust
import torch
from torchvision import models
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from code_base.tools.logger import Logger
from datetime import datetime
import numpy as np

from matplotlib import pyplot as plt
import os
import sys
plt.switch_backend('agg')  # Allow plotting when running remotely

def adjust_learning_rate(lr, optimizer, epoch, decrease_epoch=50):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // decrease_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    root = '/home/public/CITYSCAPE'
    from PIL import Image
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        name = filenames[ind].replace(root, output_dir)
        # fn = os.path.join(output_dir, filenames[ind])
        out_dir = os.path.split(name)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        im.save(name)

def load_net_synthia(state_dict, net):
    for k, v in net.state_dict().items():
        pre_param = state_dict[k]
        if pre_param.size() == v.size():
            # param = torch.from_numpy(pre_param)
            v.copy_(pre_param)

        else:
            print (k)

# Build the model
class Model_Factory():
    def __init__(self, cf):
        # If we load from a pretrained model
        self.model_name = cf.model_name
        self.num_classes = cf.num_classes
        if cf.model_name == 'segnet_basic':
            pretrained_net = FeatureResNet()
            pretrained_net.load_state_dict(models.resnet34(pretrained=True).state_dict())
            self.net = SegResNet(cf.num_classes, pretrained_net).cuda()
        elif cf.model_name == 'drn_c_26':

            self.net = DRNSeg('drn_c_26', cf.num_classes, pretrained=True, linear_up=True)

            # self.net = drn_c_26(num_classes=cf.num_classes, pretrained=cf.pretrained_drn_c_26)

        elif cf.model_name == 'drn_d_22':
            self.net = DRNSeg('drn_d_22', cf.num_classes, pretrained=True, linear_up=False)
        elif cf.model_name == 'drn_d_38':
            self.net = DRNSeg('drn_d_38', cf.num_classes, pretrained=True, linear_up=False)
        # Set the loss criterion
        self.crit = nn.NLLLoss2d(ignore_index=19).cuda()

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
            load_net_synthia(torch.load(cf.train_model_path), self.net)
            # self.net.load_state_dict(torch.load(cf.train_model_path))

        # self.net = DRNSegF(self.net, 20)
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

        if torch.cuda.is_available():
            self.net = self.net.cuda()

    def train(self, train_loader, epoch):
        lr = adjust_learning_rate(self.cf.learning_rate, self.optimiser, epoch)
        print('learning rate:', lr)
        self.net.train()
        for i, (input, target_one_hot, target, _) in enumerate(train_loader):
            self.optimiser.zero_grad()
            input, target, target_one_hot = Variable(input.cuda(async=True)), Variable(target.cuda(async=True)), Variable(target_one_hot.cuda(async=True))
            output = F.log_softmax(self.net(input))
            self.loss = self.crit(output, target)
            print(epoch, i, self.loss.data[0])
            self.loss.backward()
            self.optimiser.step()

    def train_synthia(self, train_loader, epoch):
        lr = adjust_learning_rate(self.cf.learning_rate, self.optimiser, epoch)
        print('learning rate:', lr)
        self.net.train()
        for i, (input, target) in enumerate(train_loader):
            self.optimiser.zero_grad()
            input, target = Variable(input.cuda(async=True)), Variable(target.cuda(async=True))
            output = F.log_softmax(self.net(input))
            self.loss = self.crit(output, target)
            print(epoch, i, self.loss.data[0])
            self.loss.backward()
            self.optimiser.step()

    def test(self, val_loader, epoch):
        self.net.eval()
        total_ious = []
        for i, (input, _, target, _) in enumerate(val_loader):
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

    def test_and_save(self, val_loader):
        self.net.eval()
        for i, (input, _, target, filename) in enumerate(val_loader):
            input, target = Variable(input.cuda(async=True), volatile=True), Variable(target.cuda(async=True), volatile=True)
            output = F.log_softmax(self.net(input))
            b, _, h, w = output.size()
            pred = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes).max(1)[1].view(b, h, w)
            pred = pred.cpu().data.numpy()
            save_output_images(pred, filename, '/home/ty/code/autonomous_driving/Experiments/CityScape_semantic_segmentation')

    def test_synthia(self, epoch):
        torch.save(self.net.state_dict(), os.path.join(self.exp_dir, str(epoch) + '_net.pth'))

    def test_synthia_json(self, json_path):
        # torch.save(self.net.state_dict(), os.path.join(self.exp_dir, str(epoch) + '_net.pth'))
        self.net.eval()
        from skimage import io
        import json
        # img_name = os.path.join('/home/stevenwudi/PycharmProjects/autonomous_driving/Datasets/SYNTHIA-SEQS-01-DAWN/RGB/Stereo_Left/Omni_F/000001.png')
        # print ('-------')
        # print (img_name)
        save_root = '/home/public/synthia/synthia_segmentation'
        with open(json_path, 'r') as fp:
            predict_dict = json.load(fp)

        keys = predict_dict.keys()
        keys = sorted(keys)
        predict_list = []
        for i, key in enumerate(keys):
            image = io.imread(key)
            print (i, '----', key)
            folder = key.split('/')[-5]
            image_name = os.path.basename(key)
            image = image.transpose((2, 0, 1))
            image = image[np.newaxis, ...]
            image_tensor = torch.from_numpy(image)
            image_tensor = image_tensor.float().div(255)
            for t, m, s in zip(image_tensor, self.cf.rbg_mean, self.cf.rbg_std):
                t.sub_(m).div_(s)

            input = Variable(image_tensor.cuda(async=True))
            output = F.log_softmax(self.net(input))
            b, _, h, w = output.size()
            pred = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes).max(1)[1].view(b, h, w)
            pred = pred.data.cpu()

            pred_colour = torch.zeros(b, 3, h, w)
            for k, v in self.cf.full_to_colour.items():
                pred_r = torch.zeros(b, 1, h, w)
                pred_r[(pred == k)] = v[0]
                pred_g = torch.zeros(b, 1, h, w)
                pred_g[(pred == k)] = v[1]
                pred_b = torch.zeros(b, 1, h, w)
                pred_b[(pred == k)] = v[2]
                pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))

            save_path = os.path.join(save_root, folder)
            if not os.path.exists(save_path):
                os.makedirs(os.path.join(save_root, folder))
            save_image(pred_colour[0].float().div(255), os.path.join(save_path, image_name))
            car_dict = {'image_path': key, 'boundingbox': predict_dict[key], 'segment_path': os.path.join(save_path, image_name), }
            predict_list.append(car_dict)

        json_bbox_seg_path = os.path.join('/home/public/synthia', 'car_test_bbox_seg-shuffle.json')
        with open(json_bbox_seg_path, 'w') as fp:
            json.dump(predict_list, fp, indent=4)

    def test_synthia_json2(self, json_path):
        # torch.save(self.net.state_dict(), os.path.join(self.exp_dir, str(epoch) + '_net.pth'))
        self.net.eval()
        from skimage import io

        img_name = os.path.join('/home/stevenwudi/PycharmProjects/autonomous_driving/Datasets/segmentation/SYNTHIA_RAND_CVPR16/RGB/ap_000_02-11-2015_18-02-19_000157_0_Rand_10.png')
        image = io.imread(img_name)

        image = image.transpose((2, 0, 1))
        # image = image[np.newaxis, ...]
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.float().div(255)
        # for t, m, s in zip(image_tensor, self.cf.mean, self.cf.std):
        #     t.sub_(m).div_(s)
        image_tensor[0].sub_(self.cf.rgb_mean[0]).div_(self.cf.rgb_std[0])
        image_tensor[1].sub_(self.cf.rgb_mean[1]).div_(self.cf.rgb_std[1])
        image_tensor[2].sub_(self.cf.rgb_mean[2]).div_(self.cf.rgb_std[2])
        image_tensor.unsqueeze_(0)
        input = Variable(image_tensor.cuda(async=True))
        output = F.log_softmax(self.net(input))
        b, _, h, w = output.size()
        pred = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes).max(1)[1].view(b, h, w)
        pred = pred.data.cpu()

        pred_colour = torch.zeros(b, 3, h, w)
        for k, v in self.cf.full_to_colour.items():
            pred_r = torch.zeros(b, 1, h, w)
            pred_r[(pred == k)] = v[0]
            pred_g = torch.zeros(b, 1, h, w)
            pred_g[(pred == k)] = v[1]
            pred_b = torch.zeros(b, 1, h, w)
            pred_b[(pred == k)] = v[2]
            pred_colour.add_(torch.cat((pred_r, pred_g, pred_b), 1))


        save_image(pred_colour[0].float().div(255), os.path.join(self.exp_dir, 'a.png'))

# Build the model
class Model_Factory_LSTM():
    def __init__(self, cf):
        # If we load from a pretrained model
        self.model_name = cf.model_name   #['LSTM_ManyToMany', 'LSTM_To_FC']
        if cf.model_name == 'LSTM_ManyToMany':
            self.net = LSTM_ManyToMany(input_dim=cf.lstm_inputsize,
                                       hidden_size=cf.lstm_hiddensize,
                                       num_layers=cf.lstm_numlayers,
                                       output_size=cf.lstm_outputsize,
                                       cuda=cf.cuda)
        elif cf.model_name == 'LSTM_To_FC':
            self.net = LSTM_To_FC(future=cf.LSTM_To_FC,
                                  input_dim=cf.lstm_inputsize,
                                  hidden_size=cf.lstm_hiddensize,
                                  num_layers=cf.lstm_numlayers,
                                  output_dim=cf.lstm_output_dim,
                                  cuda=cf.cuda)
        # Set the loss criterion
        if cf.loss == 'MSE':
            self.crit = nn.MSELoss()
        self.net.float()
        if cf.cuda and torch.cuda.is_available():
            print('Using cuda')
            self.net = self.net.cuda()
            self.crit = self.crit.cuda()

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

        # use LBFGS as optimizer since we can load the whole data to train
        if cf.optimizer == 'LBFGS':
            self.optimiser = optim.LBFGS(self.net.parameters(), lr=cf.learning_rate)
        elif cf.optimizer == 'adam':
            self.optimiser = optim.Adam(self.net.parameters(), lr=cf.learning_rate)
        elif cf.optimizer == 'rmsprop':
            self.optimiser = optim.RMSprop(self.net.parameters(), lr=cf.learning_rate, momentum=cf.momentum, weight_decay=cf.weight_decay)
        elif cf.optimizer == 'sgd':
            self.optimiser = optim.SGD(self.net.parameters(), lr=cf.learning_rate, momentum=cf.momentum, weight_decay=cf.weight_decay)

        self.aveErrCenter, self.aveErrCoverage = [], []

    def train(self, train_input, train_target, cf):
        #print('learning rate:', lr)
        # begin to train
        def closure():
            self.optimiser.zero_grad()
            out = self.net(train_input)[0]
            loss = self.crit(out, train_target)
            if cf.cuda:
                print('loss: ', loss.data.cpu().numpy()[0])
            else:
                print('loss: ', loss.data.numpy()[0])
            loss.backward()
            return loss

        self.optimiser.step(closure)

    def test(self, valid_input, valid_target, data_std, data_mean, cf, epoch=None):
        pred = self.net(valid_input, future=cf.lstm_predict_frame)
        loss = self.crit(pred[1], valid_target)
        if cf.cuda:
            results = pred[1].data.cpu().numpy() * data_std + data_mean
            rect_anno = valid_target.data.cpu().numpy() * data_std + data_mean
        else:
            results = pred[1].data.numpy() * data_std + data_mean
            rect_anno = valid_target.data.numpy() * data_std + data_mean

        aveErrCoverage, aveErrCenter, errCoverage, errCenter = calc_seq_err_robust(results, rect_anno)
        print('aveErrCoverage: %.4f, aveErrCenter: %.2f' % (aveErrCoverage, aveErrCenter))
        if cf.cuda:
            print('valid loss:', loss.data.cpu().numpy()[0])
        else:
            print('valid loss:', loss.data.numpy()[0])

        # TODO: 3D evaluation
        # TODO: network saving and evaluation

        # Save weights and scores
        if epoch:
            model_checkpoint = 'Epoch:%2d_net_aveErrCoverage:%.4f_aveErrCenter:%.2f___.pth' % (epoch, aveErrCoverage, aveErrCenter)
        else:
            model_checkpoint = 'Final_test:_aveErrCoverage:%.4f_aveErrCenter:%.2f.pth' % (aveErrCoverage, aveErrCenter)
        torch.save(self.net.state_dict(), os.path.join(self.exp_dir, model_checkpoint))

        # Plot scores
        # self.aveErrCoverage.append(aveErrCoverage.mean())
        # es = list(range(len(self.aveErrCoverage)))
        # plt.plot(es, self.aveErrCoverage, 'b-')
        # plt.xlabel('aveErrCoverage')
        # plt.ylabel('Mean IoU')
        # plt.savefig(os.path.join(self.exp_dir, 'ious.png'))
        # plt.close()
