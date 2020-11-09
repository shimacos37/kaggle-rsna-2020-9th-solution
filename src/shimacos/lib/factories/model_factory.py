import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from efficientnet_pytorch import EfficientNet
from .module import GeM


class ResNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout=0.0):
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
            nn.Dropout(dropout),
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
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


class CNN2D(nn.Module):
    def __init__(self, model_config):
        super(CNN2D, self).__init__()
        self.model_config = model_config
        self.cnn = timm.create_model(
            model_config.backbone,
            pretrained=True,
            num_classes=model_config.num_classes,
            in_chans=model_config.in_channels,
        )

    def forward_features(self, imgs):
        feature = self.cnn.forward_features(imgs)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = feature.view(feature.size(0), -1)
        return feature

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, seq_len, h, w, c]
        """
        if len(batch["data"].shape) == 5:
            imgs = batch["data"].permute(0, 1, 4, 2, 3)
            shape = imgs.shape
            imgs = imgs.view(shape[0] * shape[1], shape[2], shape[3], shape[4])
        else:
            imgs = batch["data"].permute(0, 3, 1, 2)
        self.feature = self.forward_features(imgs)
        out = self.cnn.get_classifier()(self.feature)
        if len(batch["data"].shape) == 5:
            out = out.view(shape[0], shape[1], -1)
        return out

    def get_feature(self):
        return self.feature


class CNN2DPooling(nn.Module):
    def __init__(self, model_config):
        super(CNN2DPooling, self).__init__()
        self.model_config = model_config
        if not "efficientnet" in self.model_config.backbone:
            self.cnn = timm.create_model(
                model_config.backbone,
                pretrained=True,
                num_classes=1,
                in_chans=3,
            )
            self.in_features = self.cnn.get_classifier().in_features
        else:
            self.cnn = EfficientNet.from_pretrained(model_config.backbone, advprop=True)
            conv0 = self.cnn._conv_stem
            self.cnn._conv_stem = nn.Conv2d(
                in_channels=3,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias,
            )
            self.in_features = self.cnn._fc.in_features
            # self.cnn.last_linear = nn.Linear(self.in_features, 1)

        self.resnet1 = ResNet(
            self.in_features, 512, kernel_size=3, dropout=model_config.dropout_rate
        )
        self.resnet2 = ResNet(
            512, 256, kernel_size=5, dropout=model_config.dropout_rate
        )
        self.resnet3 = ResNet(
            256, 128, kernel_size=7, dropout=model_config.dropout_rate
        )
        self.resnet4 = ResNet(128, 64, kernel_size=9, dropout=model_config.dropout_rate)
        self.resnet5 = ResNet(64, 32, kernel_size=11, dropout=model_config.dropout_rate)
        out_features = 512 + 256 + 128 + 64 + 32

        self.image_classifier = nn.Linear(out_features, 1)
        self.exam_classifier = nn.Linear(out_features, 9)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        # pool_out = feature.mean(1)
        return pool_out

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, seq_len, h, w, c]
        """
        imgs = batch["data"]
        shape = imgs.shape
        imgs = imgs.view(shape[0] * shape[1], shape[2], shape[3], shape[4])
        if not "efficientnet" in self.model_config.backbone:
            feature = self.cnn.forward_features(imgs)
        else:
            feature = self.cnn.extract_features(imgs)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = feature.view(feature.size(0), -1)
        feature = feature.view(shape[0], shape[1], -1)
        feature = feature.permute(0, 2, 1)
        outs = []
        for i in range(1, 6):
            feature = getattr(self, f"resnet{i}")(feature)
            outs.append(feature)
        self.feature = torch.cat(outs, dim=1).permute(0, 2, 1)

        pool_out = self.pooling(self.feature, batch["seq_len"])
        exam_out = self.exam_classifier(pool_out)
        img_out = self.image_classifier(self.feature)

        # if not "efficientnet" in self.model_config.backbone:
        #     img_out = self.cnn.get_classifier()(self.feature)
        # else:
        #     img_out = self.cnn.last_linear(self.feature)
        return img_out, exam_out

    def get_feature(self):
        return self.feature


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=201):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class DeconvFeatureModel(nn.Module):
    def __init__(self, model_config):
        super(DeconvFeatureModel, self).__init__()
        self.model_config = model_config
        if model_config.backbone in ["cnn", "cnn_rnn"]:
            self.resnet1 = ResNet(
                model_config.num_feature,
                512,
                kernel_size=3,
                dropout=model_config.dropout_rate,
            )
            self.deconv1 = nn.ConvTranspose1d(
                512, 512, kernel_size=3, stride=2, padding=1
            )
            self.resnet2 = ResNet(
                512, 256, kernel_size=5, dropout=model_config.dropout_rate
            )
            self.deconv2 = nn.ConvTranspose1d(
                256, 256, kernel_size=3, stride=2, padding=1
            )
            self.resnet3 = ResNet(
                256, 128, kernel_size=7, dropout=model_config.dropout_rate
            )
            self.deconv3 = nn.ConvTranspose1d(
                128, 128, kernel_size=3, stride=2, padding=1
            )
            self.resnet4 = ResNet(
                128, 64, kernel_size=9, dropout=model_config.dropout_rate
            )
            self.deconv4 = nn.ConvTranspose1d(
                64, 64, kernel_size=3, stride=2, padding=1
            )
            self.resnet5 = ResNet(
                64, 32, kernel_size=11, dropout=model_config.dropout_rate
            )
            self.deconv5 = nn.ConvTranspose1d(
                32, 32, kernel_size=3, stride=2, padding=1
            )
            out_features = 512 + 256 + 128 + 64 + 32
        elif model_config.backbone == "lstm":
            self.rnn1 = nn.LSTM(
                model_config.num_feature,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
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
            )
            self.deconv2 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            out_features = 1024
        elif model_config.backbone == "gru":
            self.rnn1 = nn.GRU(
                model_config.num_feature,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.deconv1 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            self.rnn2 = nn.GRU(
                1024,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            self.deconv2 = nn.ConvTranspose1d(
                1024, 512, kernel_size=3, stride=2, padding=1
            )
            out_features = 1024
        elif model_config.backbone in ["transformer", "transformer_rnn"]:
            self.linear = nn.Linear(model_config.num_feature, 2048)
            self.scale = math.sqrt(2048)
            self.pe = PositionalEncoding(2048, model_config.dropout_rate)
            encoder_layer1 = nn.TransformerEncoderLayer(
                2048,
                nhead=8,
                dim_feedforward=1024,
                dropout=model_config.dropout_rate,
                activation="gelu",
            )
            self.transformer1 = nn.TransformerEncoder(encoder_layer1, 1)
            self.deconv1 = nn.ConvTranspose1d(
                2048, 1024, kernel_size=3, stride=2, padding=1
            )
            encoder_layer2 = nn.TransformerEncoderLayer(
                1024,
                nhead=8,
                dim_feedforward=1024,
                dropout=model_config.dropout_rate,
                activation="gelu",
            )
            self.transformer2 = nn.TransformerEncoder(encoder_layer2, 1)
            self.deconv2 = nn.ConvTranspose1d(
                2048, 1024, kernel_size=3, stride=2, padding=1
            )
            out_features = 2048
        else:
            raise NotImplementedError()
        if model_config.backbone in ["cnn_rnn", "transformer_rnn"]:
            self.rnn = nn.LSTM(
                out_features,
                512,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            out_features = 1024
        self.exam_classfifier = nn.Linear(out_features, model_config.num_classes)
        self.image_classfifier = nn.Linear(out_features, 1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        return pool_out

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, seq_len, n_feature]
        """
        feature = batch["feature"].float()
        meta_feature = batch["meta_feature"].float()
        feature = torch.cat([feature, meta_feature], dim=-1)
        if self.model_config.backbone in ["cnn", "cnn_rnn"]:
            feature = feature.permute(0, 2, 1)
            outs = []
            for i in range(1, 6):
                feature = getattr(self, f"resnet{i}")(feature)
                out = getattr(self, f"deconv{i}")(feature)
                outs.append(out)
            feature = torch.cat(outs, dim=1).permute(0, 2, 1)
        elif self.model_config.backbone in ["lstm", "gru"]:
            outs = []
            feature, _ = self.rnn1(feature)
            out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(out)
            feature, _ = self.rnn2(feature)
            out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
            outs.append(out)
            feature = torch.cat(outs, dim=-1)
        elif self.model_config.backbone in ["transformer", "transformer_rnn"]:
            feature = torch.relu(self.linear(feature))
            feature = self.pe(feature.permute(1, 0, 2))
            outs = []
            feature = self.transformer1(feature)
            out = self.deconv1(feature.permute(1, 2, 0)).permute(2, 0, 1)
            outs.append(out)
            feature = self.transformer1(feature)
            out = self.deconv2(feature.permute(1, 2, 0)).permute(2, 0, 1)
            outs.append(out)
            feature = torch.cat(outs, dim=-1).permute(1, 0, 2)
        else:
            pass
        if self.model_config.backbone in ["cnn_rnn", "transformer_rnn"]:
            feature, _ = self.rnn(feature)
        # feature, _ = self.rnn(feature)
        pool_out = self.pooling(feature, batch["seq_len"])
        exam_out = self.exam_classfifier(pool_out)
        image_out = self.image_classfifier(feature)
        return image_out, exam_out

    def get_feature(self):
        return self.feature


class StackingModel(nn.Module):
    def __init__(self, model_config):
        super(StackingModel, self).__init__()
        self.model_config = model_config
        in_features = 120
        if model_config.backbone in ["cnn", "cnn_rnn"]:
            self.resnet1 = ResNet(
                in_features, 256, kernel_size=3, dropout=model_config.dropout_rate
            )
            self.resnet2 = ResNet(
                256, 128, kernel_size=3, dropout=model_config.dropout_rate
            )
            self.resnet3 = ResNet(
                128, 64, kernel_size=3, dropout=model_config.dropout_rate
            )
            self.resnet4 = ResNet(
                64, 32, kernel_size=3, dropout=model_config.dropout_rate
            )
            self.resnet5 = ResNet(
                32, 16, kernel_size=3, dropout=model_config.dropout_rate
            )
            out_features = 256 + 128 + 64 + 32 + 16
        elif model_config.backbone == "gru":
            self.rnn = nn.GRU(
                in_features,
                in_features,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            out_features = in_features * 2
        elif model_config.backbone == "lstm":
            self.rnn = nn.GRU(
                in_features,
                in_features,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            out_features = in_features * 2
        if self.model_config.backbone in ["cnn_rnn", "transformer_rnn"]:
            self.rnn = nn.GRU(
                out_features,
                out_features,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )
            out_features = out_features * 2
        else:
            pass
        self.exam_classfifier = nn.Linear(out_features, model_config.num_classes)
        self.image_classfifier = nn.Linear(out_features, 1)

    def pooling(self, feature, seq_lens):
        pool_out = torch.stack(
            [feature[idx, :seq_len].mean(0) for idx, seq_len in enumerate(seq_lens)]
        )
        # pool_out = feature.mean(1)
        return pool_out

    def forward(self, **batch):
        """
        Input:
            data (torch.Tensor): shape [bs, seq_len, n_feature]
        """
        feature = batch["feature"].float()
        if self.model_config.backbone in ["cnn", "cnn_rnn"]:
            feature = feature.permute(0, 2, 1)
            outs = []
            for i in range(1, 6):
                feature = getattr(self, f"resnet{i}")(feature)
                outs.append(feature)
            feature = torch.cat(outs, dim=1).permute(0, 2, 1)
        elif self.model_config.backbone in ["gru", "lstm"]:
            feature, _ = self.rnn(feature)
        else:
            pass
        if self.model_config.backbone in ["cnn_rnn", "transformer_rnn"]:
            feature, _ = self.rnn(feature)
        pool_out = self.pooling(feature, batch["seq_len"])
        exam_out = self.exam_classfifier(pool_out)
        image_out = self.image_classfifier(feature)
        return image_out, exam_out

    def get_feature(self):
        return self.feature


def get_2dcnn(model_config):
    model = CNN2D(model_config)
    return model


def get_pooling_2dcnn(model_config):
    model = CNN2DPooling(model_config)
    return model


def get_deconv_feature_model(model_config):
    model = DeconvFeatureModel(model_config)
    return model


def get_stacking_model(model_config):
    model = StackingModel(model_config)
    return model


def get_model(model_config):
    print("model name:", model_config.model_name)
    print("backbone name:", model_config.backbone)
    f = globals().get("get_" + model_config.model_name)
    return f(model_config)
