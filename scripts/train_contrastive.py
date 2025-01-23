# bash로 실행시 권한 부여: chmod +x train.sh

import argparse
import warnings
import sys
import os


def get_parser():
    parser = argparse.ArgumentParser(description="AIGCDetection @cby Training")
    parser.add_argument("--model_name", default='efficientnet-b0', help="Setting the model name", type=str)
    parser.add_argument("--embedding_size", default=1024, help="Setting the embedding_size", type=int)
    parser.add_argument("--pre_layer", default=None, help="Setting the pre_layer: srm or dct", type=str)
    parser.add_argument("--num_classes", default=2, help="Setting the num classes", type=int)
    parser.add_argument('--freeze_extractor', action='store_true', help='Whether to freeze extractor?')
    parser.add_argument("--model_path", default=None, help="Setting the model path", type=str)
    parser.add_argument('--no_strict', action='store_true', help='Whether to load model without strict?')
    parser.add_argument("--root_path", default='/disk4/chenby/dataset/MSCOCO',
                        help="Setting the root path for dataset loader", type=str)
    parser.add_argument("--fake_root_path", default='/disk4/chenby/dataset/AIGC_MSCOCO',
                        help="Setting the fake root path for dataset loader", type=str)
    parser.add_argument('--is_dire', action='store_true', help='Whether to using DIRE?')
    parser.add_argument('--post_aug_mode', default=None, help='Stetting the post aug mode during test phase.')
    parser.add_argument('--save_txt', default=None, help='Stetting the save_txt path.')
    parser.add_argument("--fake_indexes", default='1',
                        help="Setting the fake indexes, multi class using '1,2,3,...' ", type=str)
    parser.add_argument("--dataset_name", default='MSCOCO', help="Setting the dataset name", type=str)
    parser.add_argument("--device_id", default='0',
                        help="Setting the GPU id, multi gpu split by ',', such as '0,1,2,3'", type=str)
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    parser.add_argument('--is_crop', action='store_true', help='Whether to crop image?')
    parser.add_argument('--is_spectrum', action='store_true', help='Whether to using spectrum?')
    parser.add_argument("--batch_size", default=64, help="Setting the batch size", type=int)
    parser.add_argument("--epoch_start", default=0, help="Setting the epoch start", type=int)
    parser.add_argument("--num_epochs", default=50, help="Setting the num epochs", type=int)
    parser.add_argument("--num_workers", default=4, help="Setting the num workers", type=int)
    parser.add_argument('--is_warmup', action='store_true', help='Whether to using lr warmup')
    parser.add_argument("--lr", default=1e-3, help="Setting the learning rate", type=float)
    parser.add_argument("--save_flag", default='', help="Setting the save flag", type=str)
    parser.add_argument("--sampler_mode", default='', help="Setting the sampler mode", type=str)
    parser.add_argument('--is_test', action='store_true', help='Whether to predict the test set?')
    parser.add_argument('--is_amp', action='store_true', help='Whether to using amp autocast(使用混合精度加速)?')
    parser.add_argument("--inpainting_dir", default='full_inpainting', help="rec_image dir", type=str)
    parser.add_argument("--threshold", default=0.5, help="Setting the valid or testing threshold.", type=float)
    # contrastive learning params
    parser.add_argument('--alpha', default=0.3, type=float, help="Setting the alpha for contrastive learning")
    parser.add_argument('--pos_margin', default=0.0, type=float)
    parser.add_argument('--neg_margin', default=1.0, type=float)
    parser.add_argument('--tau', default=0.5, type=float)
    parser.add_argument('--loss_name', default='ContrastiveLoss', type=str)
    parser.add_argument('--use_miner', action='store_true', help='Whether to using miner')
    parser.add_argument('--memory_size', default=None, type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--loss_mode', default='drct', type=str, help='Setting the loss mode: drct or mine')
    parser.add_argument('--recon_weight', default=0.3, type=float, help='Setting the recon weight')
    parser.add_argument('--real_weight', default=0.3, type=float, help='Setting the real weight')
    parser.add_argument('--delta', default=1.0, type=float, help='Setting the delta')
    parser.add_argument('--data_size', default=0, type=int, help='Data size')
    parser.add_argument('--astep', default=1, type=int, help='Setting the accumulation steps')
    args = parser.parse_args()

    return args


warnings.filterwarnings("ignore")
sys.path.append('..')
args = get_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

import torch
import torch.nn as nn
import torch.optim as optim
from catalyst.data import BalanceClassSampler
import time
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score, f1_score
import pytorch_warmup as warmup

from utils.utils import Logger, AverageMeter, calculate_fnr
from network.models import get_models
from data.dataset import AIGCDetectionDataset, CLASS2LABEL_MAPPING, GenImage_LIST
from utils.losses import LabelSmoothing, CombinedLoss, FocalLoss
from data.transform import create_train_transforms, create_val_transforms



def merge_tensor(img, label, is_train=True):
    def shuffle_tensor(img, label):
        indices = torch.randperm(img.size(0))
        return img[indices], label[indices]
    if isinstance(img, list) and isinstance(label, list):
        img, label = torch.cat(img, dim=0), torch.cat(label, dim=0)
        if is_train:
            img, label = shuffle_tensor(img, label)
    return img, label


def eval_model(model, epoch, eval_loader, is_save=True, threshold=0.5, alpha=0.5,
               save_txt=None):
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    eval_process = tqdm(eval_loader)
    labels = []
    outputs = []

    with torch.no_grad():
        for i, (img, label) in enumerate(eval_process):
            optimizer.zero_grad()
            current_lr = optimizer.param_groups[0]['lr']

            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

            if args.loss_mode == 'mine':
                # img shape: [batch_size, 4, 3, 224, 224]
                batch_size, num_views, channels, height, width = img.shape

                real_images = img[:, 0, :, :, :].cuda()  # Real images
                real_recons = img[:, 1, :, :, :].cuda()  # Real reconstructions
                fake_images = img[:, 2, :, :, :].cuda()  # Fake images
                fake_recons = img[:, 3, :, :, :].cuda()  # Fake reconstructions

                # Generate embeddings for each view and reshape to [batch_size, 4, embedding_size]
                real_embeddings = model(real_images, return_feature=True)[1].unsqueeze(1)  # [batch_size, 1, embedding_size]
                real_recon_embeddings = model(real_recons, return_feature=True)[1].unsqueeze(1)  # [batch_size, 1, embedding_size]
                fake_embeddings = model(fake_images, return_feature=True)[1].unsqueeze(1)  # [batch_size, 1, embedding_size]
                fake_recon_embeddings = model(fake_recons, return_feature=True)[1].unsqueeze(1)  # [batch_size, 1, embedding_size]

                combined_embeddings = torch.cat([
                    real_embeddings,
                    real_recon_embeddings,
                    fake_embeddings,
                    fake_recon_embeddings
                ], dim=1)  # [batch_size, 4, embedding_size]

                label = label.cuda()

                y_pred, _ = model(img.view(-1, channels, height, width).cuda(), return_feature=True)
                y_pred = nn.Softmax(dim=1)(y_pred)

                # Pass combined embeddings and labels to contrastive_loss
                loss = (1 - alpha) * criterion(y_pred, label.view(-1)) + alpha * contrastive_loss(
                    combined_embeddings, label
                )

                outputs.append(1 - y_pred[:, 0])
                labels.append(label)
                acc = (torch.max(y_pred.detach(), 1)[1] == label.view(-1)).sum().item() / label.view(-1).size(0)
                losses.update(loss.item(), label.view(-1).size(0))
                accuracies.update(acc, label.view(-1).size(0))
            
            else:  # args.loss_mode == 'drct'
                img, label = merge_tensor(img, label, is_train=False)

                img, label = img.cuda(), label.cuda()

                y_pred, embeddings = model(img, return_feature=True)
                y_pred = nn.Softmax(dim=1)(y_pred)

                loss = (1 - alpha) * criterion(y_pred, label) + alpha * contrastive_loss(embeddings, label)

                outputs.append(1 - y_pred[:, 0])
                labels.append(label)
                acc = (torch.max(y_pred.detach(), 1)[1] == label).sum().item() / img.size(0)
                losses.update(loss.item(), img.size(0))
                accuracies.update(acc, img.size(0))

            if i > 0 and i % 1 == 0:
                eval_process.set_description("Epoch: %d, Loss: %.4f, Acc: %.4f" %
                                             (epoch, losses.avg, accuracies.avg))

    outputs = torch.cat(outputs, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    labels[labels > 0] = 1
    if args.loss_mode == 'mine':
        labels_flatten = labels.flatten()
    else:
        labels_flatten = labels
    auc = roc_auc_score(labels_flatten, outputs)
    recall = recall_score(labels_flatten, outputs > threshold)
    precision = precision_score(labels_flatten, outputs > threshold)
    binary_acc = accuracy_score(labels_flatten, outputs > threshold)
    f1 = f1_score(labels_flatten, outputs > threshold)
    fnr = calculate_fnr(labels_flatten, outputs > threshold)
    print(f'AUC:{auc}-Recall:{recall}-Precision:{precision}-BinaryAccuracy:{binary_acc}, f1: {f1}, fnr:{fnr}')

    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg, '.4f'),
            'acc': format(accuracies.avg, '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })

    print("Val:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))

    if save_txt is not None:
        return binary_acc, auc, recall, precision, f1, fnr

    return accuracies.avg





def train_model(model, criterion, optimizer, epoch, scaler=None, alpha=0.5, accumulation_steps=1):
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    training_process = tqdm(train_loader)

    optimizer.zero_grad()  # 최적화기 초기화

    for i, (XI, label) in enumerate(training_process):
        current_lr = optimizer.param_groups[0]['lr']
        if i > 0 and i % 1 == 0:
            training_process.set_description(
                "Epoch: %d, LR: %.8f, Loss: %.4f, Acc: %.4f" % (
                    epoch, current_lr, losses.avg, accuracies.avg))

        if args.loss_mode == 'mine':
            # Unpack data for 'mine' mode
            real, real_recon, fake, fake_recon = XI[:, 0], XI[:, 1], XI[:, 2], XI[:, 3]
            real, real_recon = real.cuda(), real_recon.cuda()
            fake, fake_recon = fake.cuda(), fake_recon.cuda()
            label = torch.stack([torch.tensor(l, dtype=torch.long) for l in label]).transpose(0, 1).cuda()

            # Concatenate all embeddings for the forward pass
            embeddings = torch.cat([real, real_recon, fake, fake_recon], dim=0)

        elif args.loss_mode == 'drct':
            # Process data normally for 'drct' mode
            XI, label = merge_tensor(XI, label, is_train=True)
            embeddings = XI.cuda()
            label = label.cuda()

        # Forward pass with autocasting
        if scaler is None:
            y_pred, embeddings = model(embeddings, return_feature=True)
        else:
            with autocast():
                y_pred, embeddings = model(embeddings, return_feature=True)

        # Loss computation
        if args.loss_mode == 'mine':
            y_pred = y_pred.view(args.batch_size, 4, -1)
            y_pred = y_pred.reshape(-1, y_pred.size(-1)) 
            new_label = label.reshape(-1)
            embeddings = embeddings.view(args.batch_size, 4, -1)
            total_num = 4 * args.batch_size
        else:
            new_label = label
            total_num = embeddings.size(0)
   

        loss = (1 - alpha) * criterion(y_pred, new_label) + alpha * contrastive_loss(embeddings, label)

        # Backward pass with gradient accumulation
        loss = loss / accumulation_steps  # 축적 단계를 고려한 손실 스케일링

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        # Gradient accumulation step
        if (i + 1) % accumulation_steps == 0:
            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad()  # 기울기 초기화

        # Compute metrics
        acc = (torch.max(y_pred.detach(), 1)[1] == new_label).sum().item() / total_num
        losses.update(loss.item() * accumulation_steps, total_num)  # 원래 크기 복원
        accuracies.update(acc, total_num)

        # Learning rate scheduling during warmup
        if args.is_warmup:
            with warmup_scheduler.dampening():
                scheduler.step()

    if not args.is_warmup:
        scheduler.step()

    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg, '.4f'),
        'acc': format(accuracies.avg, '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t Acc:{1:.4f}".format(losses.avg, accuracies.avg))

    # Clear memory
    torch.cuda.empty_cache()
    del losses, accuracies
    gc.collect()




# python train.py --device_id=0 --model_name=efficientnet-b0 --input_size=224 --batch_size=48 --fake_indexes=1 --is_amp --save_flag=
if __name__ == '__main__':
    torch.cuda.empty_cache()
    batch_size = args.batch_size * torch.cuda.device_count()
    writeFile = f"../output/{args.dataset_name}/{args.fake_indexes.replace(',', '_')}/" \
                f"{args.model_name.split('/')[-1]}_{args.input_size}{args.save_flag}/logs"
    store_name = writeFile.replace('/logs', '/weights')
    print(
        f'Using gpus:{args.device_id},batch size:{batch_size},gpu_count:{torch.cuda.device_count()},num_classes:{args.num_classes}')
    # Load model
    model = get_models(model_name=args.model_name, num_classes=args.num_classes,
                       embedding_size=args.embedding_size, freeze_extractor=args.freeze_extractor)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=not args.no_strict)
        print('Model found in {}'.format(args.model_path))
    else:
        print('No model found, initializing random model.')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothing(smoothing=0.05).cuda(device_id)

    contrastive_loss = CombinedLoss(loss_name=args.loss_name, embedding_size=args.embedding_size,
                                    pos_margin=args.pos_margin, neg_margin=args.neg_margin, tau=args.tau,
                                    memory_size=args.memory_size, use_miner=args.use_miner, num_classes=args.num_classes,
                                    recon_weight=args.recon_weight, real_weight=args.real_weight, delta=args.delta)
    is_train = not args.is_test
    print(f"EXPERIMENT: {args.loss_mode})")
    if is_train:
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'acc', 'lr'])
        # setting data loader
        xdl = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='train',
                                   num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                   transform=create_train_transforms(size=args.input_size, is_crop=args.is_crop), 
                                   mode=args.loss_mode, data_size=args.data_size
                                   )
        sampler = BalanceClassSampler(labels=xdl.get_labels(), mode=args.sampler_mode) if args.sampler_mode != '' else None  # "upsampling"
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=sampler is None, num_workers=args.num_workers, sampler=sampler, drop_last=True)   
        train_dataset_len = len(xdl)

        xdl_eval = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.fake_indexes, phase='val',
                                        num_classes=args.num_classes, inpainting_dir=args.inpainting_dir, is_dire=args.is_dire,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop), mode=args.loss_mode, data_size=args.data_size
                                        )
        
        eval_loader = DataLoader(xdl_eval, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
        if not args.is_warmup:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5)
        else:
            num_steps = train_dataset_len * args.num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

        best_acc = 0.5 if args.epoch_start == 0 else eval_model(model, args.epoch_start - 1, eval_loader, is_save=False)
        for epoch in range(args.epoch_start, args.num_epochs):
            if args.loss_mode == 'mine':
                train_model(model, criterion, optimizer, epoch, scaler=GradScaler() if args.is_amp else None, alpha=args.alpha, accumulation_steps=args.astep)
            else:
                train_model(model, criterion, optimizer, epoch, scaler=GradScaler() if args.is_amp else None, alpha=args.alpha, accumulation_steps=1)
            if epoch % 1 == 0 or epoch == args.num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader, alpha=args.alpha)
                if best_acc < acc:
                    best_acc = acc
                    save_path = '{}/{}_acc{:.4f}.pth'.format(store_name, epoch, acc)
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
            print(f'Current best acc:{best_acc}')
        last_save_path = '{}/last_acc{:.4f}.pth'.format(store_name, acc)
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_save_path)
        else:
            torch.save(model.state_dict(), last_save_path)
    else:
        start = time.time()
        epoch_start = 1
        num_epochs = 1
        xdl_test = AIGCDetectionDataset(args.root_path, fake_root_path=args.fake_root_path, fake_indexes=args.ake_indexes,
                                        phase='test', num_classes=args.num_classes, is_dire=args.is_dire,
                                        post_aug_mode=args.post_aug_mode, inpainting_dir=args.inpainting_dir,
                                        transform=create_val_transforms(size=args.input_size, is_crop=args.is_crop),
                                        mode=args.loss_mode
                                        )
        test_loader = DataLoader(xdl_test, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataset_len = len(xdl_test)
        print('test_dataset_len:', test_dataset_len)
        out_metrics = eval_model(model, epoch_start, test_loader, is_save=False, threshold=args.threshold, save_txt=args.save_txt)
        print('Total time:', time.time() - start)
        # 保存测试结果
        if args.save_txt is not None:
            os.makedirs(os.path.dirname(args.save_txt), exist_ok=True)
            acc, auc, recall, precision, f1, fnr = out_metrics
            with open(args.save_txt, 'a') as file:
                if args.dataset_name == 'GenImage':
                    class_name = GenImage_LIST[int(args.fake_indexes) - 1]
                else:
                    class_name = list(CLASS2LABEL_MAPPING.keys())[int(args.fake_indexes)]
                result_str = f'model_path:{args.model_path}, post_aug_mode:{args.post_aug_mode}, class_name:{class_name}\n' \
                             f'acc:{acc:.4f}, auc:{auc:.4f}, recall:{recall:.4f}, precision:{precision:.4f}, ' \
                             f'f1:{f1:.4f}, fnr: {fnr}\n'
                file.write(result_str)
            print(f'The result was saved in {args.save_txt}')
