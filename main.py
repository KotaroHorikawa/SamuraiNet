#coding: utf-8
##### ライブラリ読み込み #####
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from torchvision.transforms import RandomErasing  # RandomErasingをインポート
import os
import argparse
import random
from tqdm import tqdm
from Net import ResNetCNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
############################

# スクリプトの場所を取得（結果保存先の基準ディレクトリとして使用）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

############## dataloader関数 ##############
def dataload():
    # データの前処理と拡張（学習データ用）
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),     # ランダムクロップ：画像をランダムに切り取り
        transforms.RandomHorizontalFlip(),         # ランダム水平反転：左右反転をランダムに適用
        RandAugment(num_ops=3, magnitude=14),     # RandAugment：ランダムな画像拡張を適用（3種類の操作、強度14）
        transforms.ToTensor(),                     # テンソル変換：画像をPyTorchテンソルに変換
        transforms.RandomErasing(p=args.random_erasing_prob),  # ランダム消去：画像の一部をランダムに消去
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化：平均0.5、標準偏差0.5に正規化
    ])
    
    # テストデータ用の前処理（拡張なし、正規化のみ）
    test_transforms = transforms.Compose([
        transforms.ToTensor(),                     # テンソル変換
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
    ])

    # CIFAR100データセットのロード
    data_dir = '../data'  # データディレクトリ
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=False, transform=train_transforms)  # 学習データセット
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=False, transform=test_transforms)   # テストデータセット

    # DataLoaderの作成：ミニバッチの生成と並列処理の設定
    num_gpus = torch.cuda.device_count()  # 利用可能なGPU数を取得
    num_workers = 4 * num_gpus if num_gpus > 0 else 4  # GPUの数に応じてワーカー数をスケーリング
    
    # 学習用DataLoader：シャッフルあり、バッチサイズに合わないデータは捨てる
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batchsize, 
        shuffle=True,  # データをシャッフル
        drop_last=True,  # バッチサイズに満たない最後のバッチは捨てる
        num_workers=num_workers,  # 並列処理のワーカー数
        pin_memory=True  # GPUへの転送を高速化
    )
    
    # テスト用DataLoader：シャッフルなし
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batchsize, 
        shuffle=False,  # テストデータはシャッフルしない
        drop_last=True, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader


############## train関数 ##############
def train(epoch):
    # モデルを学習モードに設定
    model.train()

    # エポックごとの統計値を初期化
    sum_loss = 0  # 累積損失値
    correct = 0   # 正解数
    total = 0     # サンプル総数

    # ミニバッチごとの学習ループ
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, leave=False)):
        # データをGPUに転送
        inputs = inputs.cuda()   # 入力データをGPUに
        targets = targets.cuda() # 教師ラベルをGPUに

        # 教師ラベルをlong型に変換（CrossEntropyLoss要件）
        targets = targets.long()

        # 順伝播：モデルに入力を渡して出力を得る
        output = model(inputs)
        
        # 損失計算：モデル出力と教師ラベルから損失を計算
        loss = criterion(output, targets)

        # 勾配の初期化：前回のミニバッチの勾配情報をクリア
        optimizer.zero_grad()

        # 誤差逆伝播：損失から各パラメータの勾配を計算
        loss.backward()

        # パラメータ更新：計算した勾配に基づいてモデルのパラメータを更新
        optimizer.step()

        # 累積損失値の更新
        sum_loss += loss.item()

        # 精度の計算
        # 出力をソフトマックス関数で確率分布に変換
        output = F.softmax(output, dim=1)
        # 最大確率のクラスを予測クラスとして取得
        _, predicted = output.max(1)
        # サンプル総数を更新
        total += targets.size(0)
        # 正解数を更新（予測と教師ラベルが一致した数）
        correct += predicted.eq(targets).sum().item()

    # エポックの平均損失と精度を返す
    return sum_loss/(batch_idx+1), float(correct)/float(total)


############## test関数 ##############
def test(epoch):
    # モデルを評価モードに設定（ドロップアウト等の確率的挙動を無効化）
    model.eval()

    # 統計値の初期化
    sum_loss = 0  # 累積損失値
    correct = 0   # 正解数
    total = 0     # サンプル総数

    # パラメータの勾配計算を無効化（評価時は学習しないため）
    with torch.no_grad():
        # ミニバッチごとのループ
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, leave=False)):
            # データをGPUに転送
            inputs = inputs.cuda()
            targets = targets.cuda()

            # 教師ラベルをlong型に変換
            targets = targets.long()

            # 順伝播：モデルに入力を渡して出力を得る
            output = model(inputs)

            # 損失計算
            loss = criterion(output, targets)

            # 累積損失値の更新
            sum_loss += loss.item()

            # 精度の計算
            # 出力をソフトマックス関数で確率分布に変換
            output = F.softmax(output, dim=1)
            # 最大確率のクラスを予測クラスとして取得
            _, predicted = output.max(1)
            # サンプル総数を更新
            total += targets.size(0)
            # 正解数を更新
            correct += predicted.eq(targets).sum().item()

    # エポックの平均損失と精度を返す
    return sum_loss/(batch_idx+1), float(correct)/float(total)



############## main ##############
if __name__ == '__main__':
    ##### コマンドライン引数の設定 #####
    parser = argparse.ArgumentParser(description='CIFAR100 Training with DataParallel')
    # ミニバッチサイズの指定
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='各GPUで処理する画像の合計ミニバッチサイズ')
    # 学習エポック数の指定
    parser.add_argument('--num_epochs', '-e', type=int, default=100,
                        help='学習エポック数')
    # 結果出力ディレクトリの指定
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='結果を出力するディレクトリ')
    # クラス数の指定（CIFAR100なので通常は100）
    parser.add_argument('--classes', '-c', type=int, default=100,
                        help='クラス数')
    # 乱数シードの指定（再現性確保用）
    parser.add_argument('--seed', '-s', type=int, default=0,
                        help='乱数シード値')
    # 学習率の指定
    parser.add_argument('--lr', '-l', type=float, default=5e-2,
                        help='学習率')
    # 重み減衰率の指定（過学習防止用）
    parser.add_argument('--weight_decay', '-w', type=float, default=5e-4,
                        help='重み減衰率（L2正則化係数）')
    # ラベルスムージング係数の指定
    parser.add_argument('--label_smoothing', '-ls', type=float, default=0.1,
                        help='ラベルスムージング係数（0-1の範囲）')
    # モメンタムの指定
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='SGDのモメンタム係数')
    # Nesterovモメンタムの使用有無
    parser.add_argument('--nesterov', '-n', action='store_true',
                        help='Nesterovモメンタムを使用する場合に指定')
    # RandomErasing確率の指定
    parser.add_argument('--random_erasing_prob', '-rep', type=float, default=0.5,
                        help='RandomErasingを適用する確率')
    args = parser.parse_args()

    ##### 初期設定の表示 #####
    print("[実験条件]")
    print(" 学習エポック数   : {}".format(args.num_epochs))
    print(" ミニバッチサイズ : {}".format(args.batchsize))
    print(" クラス数         : {}".format(args.classes))
    print(" 学習率           : {}".format(args.lr))
    print(" 重み減衰率       : {}".format(args.weight_decay))
    print(" ラベルスムージング: {}".format(args.label_smoothing))
    print(" モメンタム       : {}".format(args.momentum))
    print(" Nesterovモメンタム: {}".format(args.nesterov))
    print(f" RandomErasing確率 : {args.random_erasing_prob}")
    

    ##### GPU設定 #####
    # GPUの利用可否をチェック
    if not torch.cuda.is_available():
        print("CUDAが利用できません。CPUで実行します。")
        device = torch.device('cpu')
        num_gpus = 0
    else:
        num_gpus = torch.cuda.device_count()
        print(f"CUDAが利用可能です。{num_gpus}個のGPUを使用します。")
        device = torch.device('cuda')  # デフォルトデバイスはcuda:0

    print("")


    ##### 保存ディレクトリ・ファイルの設定 #####
    # スクリプトと同じディレクトリに結果保存用ディレクトリを作成
    results_dir = os.path.join(SCRIPT_DIR, args.out)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 各種ログファイルのパス設定
    PATH_1 = os.path.join(results_dir, "trainloss.txt")      # 学習損失ログ
    PATH_2 = os.path.join(results_dir, "testloss.txt")       # テスト損失ログ
    PATH_3 = os.path.join(results_dir, "trainaccuracy.txt")  # 学習精度ログ
    PATH_4 = os.path.join(results_dir, "testaccuracy.txt")   # テスト精度ログ
    PATH_5 = os.path.join(results_dir, "best_model.pth")     # 最良モデル保存先

    # ログファイルの初期化
    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass
    with open(PATH_4, mode = 'w') as f:
        pass

    # 乱数シードの設定（再現性確保のため）
    random.seed(args.seed)         # Pythonの乱数
    np.random.seed(args.seed)      # NumPyの乱数
    torch.manual_seed(args.seed)   # PyTorchの乱数
    if num_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)    # 複数GPU用の乱数設定
        torch.backends.cudnn.deterministic = True # 完全な再現性のため
        torch.backends.cudnn.benchmark = False    # 再現性のためベンチマーク無効化


    # モデルの設定
    # ベースモデルのインスタンス化
    model = ResNetCNN(n_class=args.classes)

    # 複数GPUがある場合はDataParallelでラップ
    if num_gpus > 1:
        print(f"{num_gpus}個のGPUでnn.DataParallelを使用します。")
        model = nn.DataParallel(model)

    # モデルを指定デバイス（GPU/CPU）に転送
    model.to(device)

    # 損失関数の設定（ラベルスムージング付きクロスエントロピー）
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # オプティマイザの設定 - SGD（確率的勾配降下法）
    # Nesterovモメンタムはモメンタムの変種で、より良い収束性能を持つことがある
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,                      # 学習率
        momentum=args.momentum,          # モメンタム係数
        weight_decay=args.weight_decay,  # 重み減衰率（L2正則化）
        nesterov=args.nesterov           # Nesterovモメンタムの使用有無
    )
    
    # 学習率スケジューラの設定（コサインアニーリング）
    # エポックが進むにつれて学習率を徐々に小さくする
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # データローダーの取得
    train_loader, test_loader = dataload()

    # ベストモデル保存用の変数
    best_accuracy = 0.0

    ##### 学習と評価のループ #####
    # 学習履歴の記録用配列
    train_losses = []     # 学習損失履歴
    test_losses = []      # テスト損失履歴
    train_accuracies = [] # 学習精度履歴
    test_accuracies = []  # テスト精度履歴
    lr_history = []       # 学習率履歴

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # 現在の学習率を記録
        lr_history.append(optimizer.param_groups[0]['lr'])

        # 学習
        train_loss, train_accuracy = train(epoch)
        # 評価
        test_loss, test_accuracy = test(epoch)

        # 履歴の記録
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy * 100)  # パーセンテージに変換
        test_accuracies.append(test_accuracy * 100)    # パーセンテージに変換

        # 学習率の更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ベストモデルの保存（テスト精度が過去最高の場合）
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # DataParallelラッパーではなく、元のモデルの状態を保存
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(model_state_dict, PATH_5)
            print(f"Best model saved with accuracy: {best_accuracy*100:.2f}%")

        ##### 結果表示 #####
        # オプティマイザから現在の学習率を取得
        current_lr = optimizer.param_groups[0]['lr']
        print("Epoch{:3d}/{:3d}  TrainLoss={:.4f}  TestAccuracy={:.2f}%  LR={:.6f}".format(
            epoch+1, args.num_epochs, train_loss, test_accuracy*100, current_lr))


        ##### ログファイルへの出力 #####
        with open(PATH_1, mode = 'a') as f:
            f.write("{}\t{:.4f}\n".format(epoch+1, train_loss))
        with open(PATH_2, mode = 'a') as f:
            f.write("{}\t{:.4f}\n".format(epoch+1, test_loss))
        with open(PATH_3, mode = 'a') as f:
            f.write("{}\t{:.2f}\n".format(epoch+1, (train_accuracy*100)))
        with open(PATH_4, mode = 'a') as f:
            f.write("{}\t{:.2f}\n".format(epoch+1, (test_accuracy)*100))


    # 最終的な結果を表示
    print("Training completed!")
    print(f"Best test accuracy: {best_accuracy*100:.2f}%")

    # 学習曲線のグラフ作成と保存
    epochs = list(range(1, args.num_epochs + 1))
    
    # 損失のグラフ
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "loss_graph.png"))
    plt.close()
    
    # 精度のグラフ
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "accuracy_graph.png"))
    plt.close()
    
    # 学習率のグラフ
    if args.num_epochs > 1:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, lr_history, 'g-')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule (Cosine Annealing)')
        plt.grid(True)
        plt.ylim(bottom=0)  # 学習率のグラフが0から始まるようにする
        plt.savefig(os.path.join(results_dir, "lr_schedule.png"))
        plt.close()

    print(f"Graphs saved to {results_dir}/")



