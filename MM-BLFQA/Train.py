import sys
import time
import argparse
import random
from tqdm import tqdm
from Val import val
from Utils import *
from models import Network
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=70, help='number of epochs to train')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='Seed for initializing training')
    parser.add_argument('--trainset_dir', type=str, default='../Datasets/HUST_LFI_5x32x64/')
    parser.add_argument('--log_dir', type=str, default='./log', help='root directory for logs')

    return parser.parse_args()


def train(train_loader, cfg, test_scene_id):
    # ===== build log path =====
    log_dir = os.path.join(cfg.log_dir, f"{test_scene_id[0]}_{test_scene_id[-1]}")
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = open(os.path.join(log_dir, "log.txt"), "a")

    print(cfg)
    print(test_scene_id)

    net = Network().to(cfg.device)
    optimizer = torch.optim.SGD([paras for paras in net.parameters() if paras.requires_grad == True],
                                lr=cfg.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # ===== loss functions =====
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss()

    loss_list = []
    best_SRCC = 0
    for idx_epoch in range(0, cfg.n_epochs):

        net.train()
        loss_epoch = []

        start_time = time.time()
        for idx_iter, (data, score_label, cls_label, level_label) in enumerate(tqdm(train_loader, leave=False)):
            data = data.to(cfg.device)
            score_label = score_label.to(cfg.device)
            cls_label = cls_label.to(cfg.device)
            level_label = level_label.to(cfg.device)

            cls_label = cls_label.view(cls_label.size()[0])
            level_label = level_label.view(level_label.size()[0])
            score_label = score_label.view(score_label.size()[0], -1)

            score_out, cls_out, level_out = net(data)

            loss_main = criterion_mse(score_out, score_label)
            loss_aux1 = criterion_ce(cls_out, cls_label)
            loss_aux2 = criterion_ce(level_out, level_label)
            loss = loss_main + 0.1 * loss_aux1 + 0.1 * loss_aux2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())

        loss_list.append(float(np.array(loss_epoch).mean()))
        end_time = time.time()
        print('Test Epoch----%5d,'
              ' loss---%f,'
              ' Time---%f s'
              ' lr---%7f s'
              % (idx_epoch + 1,
                 float(np.array(loss_epoch).mean()),
                 end_time - start_time,
                 scheduler.get_last_lr()[0]))

        save_ckpt(
            {
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,
            },
            log_dir,
            f"MMBLFQA_epoch{idx_epoch + 1}.pth.tar"
        )

        # validation
        start_time_val = time.time()
        net.eval()
        val_SRCC = val(
            net,
            valset_dir=cfg.trainset_dir,
            test_scene_id=test_scene_id,
            device=cfg.device
        )
        net.train()
        end_time_val = time.time()
        print('Val_Time----    %f s'
              % (end_time_val - start_time_val)
              )

        if val_SRCC > best_SRCC:
            best_SRCC = val_SRCC
            save_ckpt({'state_dict': net.state_dict()}, log_dir, "best_model.pth.tar")

        # Linux
        if idx_epoch + 1 != cfg.n_epochs:
            os.remove(os.path.join(log_dir, f"MMBLFQA_epoch{idx_epoch + 1}.pth.tar"))

        scheduler.step()


def save_ckpt(state, save_path='./log', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))


def main(cfg):
    scene_num = 20  # for HUST-LFI dataset
    full_dataset_dir = cfg.trainset_dir
    for i in range(0, scene_num, 4):
        test_scene_id = list(range(i, i + 4))
        train_set = MyTrainSetLoader_Kfold(dataset_dir=full_dataset_dir, test_scene_id=test_scene_id)
        train_loader = DataLoader(dataset=train_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
                                  shuffle=True, pin_memory=True)
        train(train_loader, cfg, test_scene_id)


if __name__ == '__main__':
    cfg = parse_args()

    # set random seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(cfg)
