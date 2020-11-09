import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import torch
from pooling import GeM, MeanMaxPooling
from activation import Swish

class ImageModel(nn.Module):
    def __init__(self, cfg):
        super(ImageModel, self).__init__()
        in_channel = 3
        self.arch = timm.create_model(cfg.model.name, pretrained=True, in_chans=in_channel)
        out_channel = self.arch.num_features
        self.pool = MeanMaxPooling()
        if cfg.model.pool == 'MeanMax':
            self.pool = MeanMaxPooling()
            out_channel = out_channel * 2
        elif cfg.model.pool == 'GeM':
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.l0 = nn.Linear(out_channel, 256)
        self.bn0 = nn.BatchNorm1d(256)
        self.act = Swish()
        if cfg.model.slice:
            self.l1 = nn.Linear(256, 30)
        else:
            self.l1 = nn.Linear(256, 10)

    def forward(self, batch):
        image = batch['image']
        out = self.arch.forward_features(image)
        out = self.pool(out)
        out = out.view(out.shape[0], -1)
        out = self.act(self.bn0(self.l0(out)))
        out = self.l1(out)
        return out

    def get_feature(self, batch):
        with torch.no_grad():
            image = batch['image']
            out = self.arch.forward_features(image)
            out = self.pool(out)
            out = out.view(out.shape[0], -1)
            return out


class SeqModel(nn.Module):
    def __init__(self, cfg):
        super(SeqModel, self).__init__()
        arch = timm.create_model(cfg.model.name, pretrained=False)
        out_channel = arch.num_features
        self.cfg = cfg
        self.model = HopModelStem(cfg=cfg, in_features=out_channel)

    def forward(self, batch):
        x = batch['feature']
        x = x.permute(1, 0, 2)
        x = self.seq_model(x)[0]
        x = x.permute(1, 0, 2)

        mask = batch['mask']
        per_exam_x = torch.stack([x[idx][mask[idx]].mean(dim=0) for idx in range(x.size(0))])
        per_image_x = self.l2_per_image(x)
        per_exam_x = self.l2_per_exam(per_exam_x)
        return per_exam_x, per_image_x

    def predict(self, batch):
        with torch.no_grad():
            return self.forward(batch)


class ResNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout=0.0, spatial_dropout=False):
        super(ResNet, self).__init__()
        assert kernel_size % 2 == 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout) if spatial_dropout else nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout) if spatial_dropout else nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


class HopModelStem(nn.Module):
    def __init__(self, in_features, cfg):
        super(HopModelStem, self).__init__()
        self.cfg = cfg
        if cfg.second.backbone in ["cnn", "cnn_rnn"]:
            self.resnet1 = ResNet(
                in_features, 512, kernel_size=3, dropout=cfg.second.dropout_rate,
                spatial_dropout=cfg.second.spatial_dropout
            )
            self.deconv1 = nn.ConvTranspose1d(
                512, 512, kernel_size=3, stride=2, padding=1
            )
            self.resnet2 = ResNet(
                512, 256, kernel_size=5, dropout=cfg.second.dropout_rate,
                spatial_dropout=cfg.second.spatial_dropout
            )
            self.deconv2 = nn.ConvTranspose1d(
                256, 256, kernel_size=3, stride=2, padding=1
            )
            self.resnet3 = ResNet(
                256, 128, kernel_size=7, dropout=cfg.second.dropout_rate,
                spatial_dropout=cfg.second.spatial_dropout
            )
            self.deconv3 = nn.ConvTranspose1d(
                128, 128, kernel_size=3, stride=2, padding=1
            )
            self.resnet4 = ResNet(
                128, 64, kernel_size=9, dropout=cfg.second.dropout_rate,
                spatial_dropout=cfg.second.spatial_dropout
            )
            self.deconv4 = nn.ConvTranspose1d(
                64, 64, kernel_size=3, stride=2, padding=1
            )
            self.resnet5 = ResNet(
                64, 32, kernel_size=11, dropout=cfg.second.dropout_rate,
                spatial_dropout=cfg.second.spatial_dropout
            )
            self.deconv5 = nn.ConvTranspose1d(
                32, 32, kernel_size=3, stride=2, padding=1
            )
            self.out_features = 512 + 256 + 128 + 64 + 32
        elif cfg.second.backbone == "rnn":
            self.rnn1 = nn.LSTM(
                in_features,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
            self.deconv1 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            self.rnn2 = nn.LSTM(
                1024,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
            self.deconv2 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            self.out_features = 1024
        else:
            raise NotImplementedError()

        if cfg.second.backbone == "cnn_rnn":
            self.rnn = nn.LSTM(
                self.out_features,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
            self.out_features = 1024

    def forward(self, batch):
        """
        Input:
            data (torch.Tensor): shape [bs, seq_len, n_feature]
        """
        feature = batch["feature"].float()
        meta_feature = batch["meta_feature"].float()
        feature = torch.cat([feature, meta_feature], dim=-1)
        if self.cfg.second.backbone in ["cnn", "cnn_rnn"]:
            feature = feature.permute(0, 2, 1)
            outs = []
            for i in range(1, 6):
                feature = getattr(self, f"resnet{i}")(feature)
                out = getattr(self, f"deconv{i}")(feature)
                outs.append(out)
            feature = torch.cat(outs, dim=1).permute(0, 2, 1)
        elif self.cfg.second.backbone == "rnn":
            outs = []
            feature, _ = self.rnn1(feature)
            out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(out)
            feature, _ = self.rnn2(feature)
            out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(out)
            feature = torch.cat(outs, dim=-1)
        if self.cfg.second.backbone == "cnn_rnn":
            feature, _ = self.rnn(feature)
        return feature


class DoubleHopModel(nn.Module):
    def __init__(self, cfg):
        super(DoubleHopModel, self).__init__()
        self.cfg = cfg
        self.model_512 = HopModelStem(cfg=cfg, in_features=2055)
        self.model_384 = HopModelStem(cfg=cfg, in_features=1543)
        in_features = self.model_512.out_features + self.model_384.out_features
        out_feature = 512
        self.rnn = nn.LSTM(in_features,
                           out_feature,
                           num_layers=4,
                           batch_first=True,
                           bidirectional=True, dropout=0.2)
        self.exam_classfifier = nn.Linear(out_feature * 2, cfg.second.num_classes)
        self.image_classfifier = nn.Linear(out_feature * 2, 1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        return pool_out

    def forward(self, batch):
        feature_512 = self.model_512({'feature': batch['feature_512'],
                                      'meta_feature': batch['meta_feature_512']})

        feature_384 = self.model_384({'feature': batch['feature_384'],
                                      'meta_feature': batch['meta_feature_384']})

        feature = torch.cat([feature_512, feature_384], dim=-1)
        feature, _ = self.rnn(feature)
        pool_out = self.pooling(feature, batch["seq_len"])
        exam_out = self.exam_classfifier(pool_out)
        image_out = self.image_classfifier(feature)
        return exam_out, image_out

    def predict(self, batch):
        with torch.no_grad():
            return self.forward(batch)


class DeconvFeatureModel(nn.Module):
    def __init__(self, cfg):
        super(DeconvFeatureModel, self).__init__()
        self.cfg = cfg
        in_features = 2055
        self.model = HopModelStem(cfg=cfg, in_features=in_features)
        out_features = self.model.out_features
        self.exam_classfifier = nn.Linear(out_features, cfg.second.num_classes)
        self.image_classfifier = nn.Linear(out_features, 1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        return pool_out

    def forward(self, batch):
        feature = batch['feature_512']
        meta_feature = batch['meta_feature_512']
        feature = self.model({'feature': feature, 'meta_feature': meta_feature})
        pool_out = self.pooling(feature, batch["seq_len"])
        exam_out = self.exam_classfifier(pool_out)
        image_out = self.image_classfifier(feature)
        return exam_out, image_out

    def predict(self, batch):
        with torch.no_grad():
            return self.forward(batch)


class DoubleDeconvFeatureModel(nn.Module):
    def __init__(self, cfg):
        super(DoubleDeconvFeatureModel, self).__init__()
        self.cfg = cfg
        in_features = 2055 + 1543
        self.model = HopModelStem(cfg=cfg, in_features=in_features)
        out_features = self.model.out_features
        self.exam_classfifier = nn.Linear(out_features, cfg.num_classes)
        self.image_classfifier = nn.Linear(out_features, 1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        return pool_out

    def forward(self, batch):
        """
        Input:
            data (torch.Tensor): shape [bs, seq_len, n_feature]
        """
        feature_512 = batch["feature_512"].float()
        meta_feature_512 = batch["meta_feature_512"].float()
        feature_384 = batch["feature_384"].float()
        meta_feature_384 = batch["meta_feature_384"].float()

        feature = torch.cat([feature_512, feature_384], dim=-1)
        meta_feature = torch.cat([meta_feature_512, meta_feature_384], dim=-1)
        feature = self.model({'feature': feature, 'meta_feature': meta_feature})
        pool_out = self.pooling(feature, batch["seq_len"])
        exam_out = self.exam_classfifier(pool_out)
        image_out = self.image_classfifier(feature)
        return exam_out, image_out

    def predict(self, batch):
        with torch.no_grad():
            return self.forward(batch)
