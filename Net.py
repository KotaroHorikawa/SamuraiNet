import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment, RandomErasing # RandAugmentを使用
import os
import argparse
import random
# from tqdm import tqdm # -> tqdm.notebookに変更
from tqdm.notebook import tqdm # Colabでの進捗バー表示用
# from Net import CNN # -> CNNクラスを直接定義
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        # ブロック内の畳み込みストライドは常に1
        stride = 1

        # GELUアクティベーションはforward内でF.geluを使って適用

        # ブロック内の最初の畳み込み層
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # バッチ正規化：学習を安定化

        # ブロック内の2番目の畳み込み層
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # バッチ正規化

        # チャネル数変更時の次元合わせのためのショートカット接続
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # 1x1畳み込みでチャネル数を変更
                nn.BatchNorm2d(out_channels)  # バッチ正規化
            )

    def forward(self, x):
        # 入力を保存（ショートカット用）
        identity = x

        # メインパス - GELUを使用
        out = F.gelu(self.bn1(self.conv1(x)))  # 畳み込み→バッチ正規化→GELU活性化
        out = self.bn2(self.conv2(out))  # 畳み込み→バッチ正規化（加算後に活性化）

        # ショートカットを適用
        identity = self.shortcut(identity)

        # メインパスにショートカットを足す（残差接続）
        out += identity
        out = F.gelu(out)  # 加算後にGELU活性化を適用

        return out

# --- 残差ブロックとGELUを使用した改良型CNN ---
class ResNetCNN(nn.Module):  # クラスの名前を分かりやすく変更
    def __init__(self, n_class=100):
        super(ResNetCNN, self).__init__()

        # GELUアクティベーションはforward内でF.geluを使って適用

        # 入力ステージ: 初期畳み込み層（オリジナルのCNNの最初のレイヤーと同様）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # RGB 3チャネル→64チャネル
        self.bn1 = nn.BatchNorm2d(64)  # バッチ正規化
        # アクティベーション（GELU）はforward内で適用

        # ステージ1: 入力64チャネル → 出力128チャネル、解像度は1/2に
        self.block11 = BasicBlock(64, 128)  # チャネル数変更あり、ショートカット必要
        self.block12 = BasicBlock(128, 128)  # チャネル数変更なし、通常のショートカット
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 解像度を1/2に縮小
        self.dropout1 = nn.Dropout(0.25)  # 25%の確率で特徴をドロップアウト（過学習防止）

        # ステージ2: 入力128チャネル → 出力256チャネル、解像度は1/2に
        self.block21 = BasicBlock(128, 256)  # チャネル数変更あり、ショートカット必要
        self.block22 = BasicBlock(256, 256)  # チャネル数変更なし
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 解像度を1/2に縮小
        self.dropout2 = nn.Dropout(0.25)  # 25%の確率でドロップアウト

        # ステージ3: 入力256チャネル → 出力512チャネル、解像度は1/2に
        self.block31 = BasicBlock(256, 512)  # チャネル数変更あり、ショートカット必要
        self.block32 = BasicBlock(512, 512)  # チャネル数変更なし
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 解像度を1/2に縮小
        self.dropout3 = nn.Dropout(0.25)  # 25%の確率でドロップアウト

        # ステージ4: 入力512 → 出力1024、解像度同じ（プーリングなし）
        # 注：このブロックはコメントアウトされており、使用されていない
        # self.block4 = BasicBlock(512, 1024)  # チャネル数変更あり、ショートカット必要
        # ステージ4後のプーリングはオリジナルアーキテクチャになし
        self.dropout4 = nn.Dropout(0.3)  # 30%の確率でドロップアウト

        # グローバル平均プーリング：特徴マップを1x1に集約
        self.gap = nn.AdaptiveAvgPool2d(1)  # どんなサイズの入力も1x1に適応的に変換

        # 最終ドロップアウトと分類層（全結合層）
        self.dropout5 = nn.Dropout(0.5)  # 50%の確率でドロップアウト（分類前に強めの正則化）
        self.fc = nn.Linear(512, n_class)  # 512次元の特徴ベクトルをn_class次元に変換

        # 重みの初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """ネットワークの重みパラメータを適切に初期化するメソッド"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 畳み込み層はKaimingの初期化（ReLU/GELUに適した初期化方法）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 'relu'はGELUにも適用可能
                if m.bias is not None:  # バイアスがある場合（BNを使用する場合はFalseのはず）
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # バッチ正規化層は標準的な初期化（重み=1、バイアス=0）
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全結合層は通常の分布で初期化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """順伝播計算（入力から出力を計算）"""
        # 入力ステージ - GELUを使用
        h = F.gelu(self.bn1(self.conv1(x)))  # 畳み込み→バッチ正規化→GELU活性化

        # ステージ1
        h = self.block11(h)  # 残差ブロック1-1
        h = self.block12(h)  # 残差ブロック1-2
        h = self.pool1(h)    # プーリングで解像度を半分に
        h = self.dropout1(h)

        # ステージ2
        h = self.block21(h)  # 残差ブロック2-1
        h = self.block22(h)  # 残差ブロック2-2
        h = self.pool2(h)    # プーリングで解像度を半分に
        h = self.dropout2(h)

        # ステージ3
        h = self.block31(h)  # 残差ブロック3-1
        h = self.block32(h)  # 残差ブロック3-2
        h = self.pool3(h)    # プーリングで解像度を半分に
        h = self.dropout3(h)

        # ステージ4 - 現在はコメントアウト（使用しない）
        # h = self.block4(h)
        # プーリングなし
        # h = self.dropout4(h)

        # グローバル平均プーリングとフラット化
        h = self.gap(h)                 # 空間的な広がりを1x1に集約
        h = h.view(h.size(0), -1)       # バッチサイズ×特徴次元のテンソルに変形

        # 最終ドロップアウトと分類
        h = self.dropout5(h)  # ドロップアウトで過学習を防止
        y = self.fc(h)        # 全結合層で最終的なクラス予測を生成

        return y
