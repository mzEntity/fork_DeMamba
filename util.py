import models
from evaluation_metrics import detection_metrics
import time
import torch
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, average_precision_score, roc_auc_score

# Old checkpoints have identical tensor shapes but incompatible temporal semantics.
PIPELINE_VERSION = 'temporal_blocks_v2'

def checkpoint_accuracy(cfg, video_ids, predictions, labels):
    sources = cfg.get('aigvdbench_checkpoint_sources')
    if cfg['dataset'] != 'AIGVDBench' or sources is None:
        return accuracy_score(labels, predictions)
    if not isinstance(sources, list) or not sources:
        raise ValueError('aigvdbench_checkpoint_sources must be a non-empty list')
    groups = ['/'.join(str(path).split('/')[1:-1]) for path in video_ids]
    missing = set(sources) - set(groups)
    if missing:
        raise ValueError(f'Checkpoint selection sources absent from validation: {sorted(missing)}')
    mask = np.asarray([group in sources for group in groups])
    return accuracy_score(np.asarray(labels)[mask], np.asarray(predictions)[mask])

def build_model(cfg):
    model_name = cfg['model']
    if model_name == 'F3Net':
        model = models.Det_F3_Net()
    if model_name == 'NPR':
        model = models.resnet50_npr()
    if model_name == 'STIL':
        model = models.Det_STIL()
    if model_name == 'XCLIP_DeMamba':
        model = models.XCLIP_DeMamba(cfg['pretrained_model_path'])
    if model_name == 'CLIP_DeMamba':
        model = models.CLIP_DeMamba()
    if model_name == 'XCLIP':
        model = models.XCLIP(cfg['pretrained_model_path'])
    return model

def eval_model(cfg, model, val_loader, loss_ce, val_batch_size):
    model.eval()
    outpred_list = []
    gt_label_list = []
    video_list = []
    valLoss = 0
    lossTrainNorm = 0
    print("******** Start Testing. ********")

    with torch.no_grad():  # No need to track gradients during validation
        for i, (_, input, target, binary_label, video_id) in enumerate(tqdm(val_loader, desc="Validation", total=len(val_loader))):
            if i == 0:
                ss_time = time.time()
            
            input = input[:,0]
            varInput = torch.autograd.Variable(input.float().cuda())
            varTarget = torch.autograd.Variable(target.contiguous().cuda())
            var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())

            logit = model(varInput)
            lossvalue = loss_ce(logit, var_Binary_Target)

            valLoss += lossvalue.item()
            lossTrainNorm += 1
            outpred_list.append(logit[:,0].sigmoid().cpu().detach().numpy())
            gt_label_list.append(varTarget.cpu().detach().numpy())
            video_list.append(video_id)
    
    valLoss = valLoss / lossTrainNorm

    outpred = np.concatenate(outpred_list, 0)
    gt_label = np.concatenate(gt_label_list, 0)
    video_list = np.concatenate(video_list, 0)
    pred_labels = [1 if item >= 0.5 else 0 for item in outpred]
    true_labels = np.argmax(gt_label, axis=1)

    pred_accuracy = accuracy_score(true_labels, pred_labels)

    return pred_accuracy, video_list, pred_labels, true_labels, outpred

def train_one_epoch(cfg, model, loss_ce, scheduler, optimizer, epochID, max_epoch, max_acc, train_loader, val_loader, snapshot_path):
    model.train()
    trainLoss = 0
    lossTrainNorm = 0

    ss_time = time.time()
    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Training epoch {epochID + 1}")
    for batchID, (index, input, target, binary_label) in enumerate(pbar):
        if batchID == 0:
            ss_time = time.time()
        input = input[:,0].float()
        varInput = torch.autograd.Variable(input).cuda()
        varTarget = torch.autograd.Variable(target.contiguous().cuda())
        var_Binary_Target = torch.autograd.Variable(binary_label.contiguous().cuda())
        optimizer.zero_grad()

        logit = model(varInput)
        lossvalue = loss_ce(logit, var_Binary_Target)
        
        lossvalue.backward()
        optimizer.step()

        trainLoss += lossvalue.item()
        lossTrainNorm += 1
        pbar.set_postfix(loss=trainLoss / lossTrainNorm)
        del lossvalue

    trainLoss = trainLoss / lossTrainNorm
    scheduler.step()
    
    if (epochID+1) % 1 == 0:
        pred_accuracy, video_id, pred_labels, true_labels, outpred = eval_model(cfg, model, val_loader, loss_ce, cfg['val_batch_size'])    
        selection_accuracy = checkpoint_accuracy(cfg, video_id, pred_labels, true_labels)

        torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict(), "pipeline_version": PIPELINE_VERSION, "config": cfg},
            snapshot_path + "/last"+ ".pth",
            )
        torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict(), "pipeline_version": PIPELINE_VERSION, "config": cfg},
            snapshot_path + "/Epoch_" + str(epochID + 1) + ".pth",
        )

        if cfg.get("save_best_checkpoint", True) and pred_accuracy > max_acc:
            max_epoch, max_acc = epochID, pred_accuracy
            torch.save(
            {"epoch": epochID + 1, "model_state_dict": model.state_dict(), "pipeline_version": PIPELINE_VERSION, "config": cfg},
            snapshot_path + "/best_acc"+ ".pth",
            )

        df_result = pd.DataFrame({
            'data_path': video_id,
            'predicted_label': pred_labels,
            'actual_label': true_labels,
            'predicted_prob':outpred
        })

        metrics = detection_metrics(true_labels, outpred)
        ap = (average_precision_score(true_labels, outpred)
              if np.any(np.asarray(true_labels) == 1) else float('nan'))
        summary = (
            f"ACC@0.5: {pred_accuracy:.4%} | AUC: {metrics['auc']:.4%} | "
            f"ACC@5%FPR: {metrics['acc_at_fpr']:.4%} | "
            f"FPR: {metrics['fpr']:.4%} | TPR: {metrics['tpr']:.4%} | "
            f"threshold: {metrics['threshold']:.9g}"
        )
        print(f"Validation epoch {epochID + 1}: {summary}")
        temp_result_txt = snapshot_path+'/Epoch_'+str(epochID + 1)+'_accuracy.txt'
        with open(temp_result_txt, 'w') as file:
            file.write(summary + "\n")
            file.write(f"AP: {ap:.4%}\n")
            file.write("ACC@5%FPR uses score >= threshold; real=0, fake=1; "
                       "threshold is measured on this evaluation split.\n")

        if cfg['dataset'] == 'GenVideo':
            prefixes = ["fake/ModelScope", "fake/Morph", "fake/MoonValley",
                        "fake/HotShot", "fake/EvalCrafter_T2V_Dataset/show_1",
                        "fake/Sora", "fake/Wild", "fake/VideoCrafter",
                        "fake/Lavies", "fake/Gen2"]
            video_nums = [700, 700, 626, 700, 700, 56, 926, 1400, 1400, 1380]

            condition = df_result['data_path'].apply(lambda x: x.startswith("real"))
            temp_df_val = df_result[condition].copy()
            temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
            accuracy = temp_df_val['correct'].mean()
            FP = int((1 - accuracy) * 10000)

            for index, temp_prefixes in enumerate(prefixes):
                condition = df_result['data_path'].apply(lambda x: x.startswith(temp_prefixes))
                temp_df_val = df_result[condition].copy()
                temp_df_val['correct'] = temp_df_val['predicted_label'] == temp_df_val['actual_label']
                accuracy = temp_df_val['correct'].mean()

                TP = int(accuracy * video_nums[index])
                FN = int((1 - accuracy) * video_nums[index])
                P, R = TP / (TP + FP), TP / (TP + FN)
                F1 = 2 * P * R / (P + R)

                condition |= df_result['data_path'].str.startswith('real')
                temp_df_val = df_result[condition]
                true_labels = temp_df_val['actual_label']
                pred_probs = temp_df_val['predicted_prob']
                ap = average_precision_score(true_labels, pred_probs)
                with open(temp_result_txt, 'a') as file:
                    name = temp_prefixes.split('/')[-1]
                    file.write(f"文件名: {name}, Recall是: {accuracy}\n")
                    file.write(f"文件名: {name}, F1是: {F1}\n")
                    file.write(f"文件名: {name}, AP是: {ap}\n")
        else:
            # For other datasets, report accuracy by the source portion of video_id.
            source = df_result['data_path'].apply(
                lambda path: '/'.join(str(path).split('/')[:3])
                if len(str(path).split('/')) >= 3 else str(path)
            )
            with open(temp_result_txt, 'a') as file:
                for source_name, source_df in df_result.groupby(source):
                    accuracy = (source_df['predicted_label'] == source_df['actual_label']).mean()
                    line = f"来源: {source_name}, N: {len(source_df)}, Accuracy/Recall: {accuracy:.4f}"
                    # A single fake source contains only positives; evaluate its
                    # ranking against the common real validation set.
                    if (source_df['actual_label'] == 1).all():
                        paired = pd.concat([source_df, df_result[df_result['actual_label'] == 0]])
                        if paired['actual_label'].nunique() == 2:
                            source_auc = roc_auc_score(paired['actual_label'], paired['predicted_prob'])
                            source_ap = average_precision_score(paired['actual_label'], paired['predicted_prob'])
                            line += f", AUC(vs Real): {source_auc:.6f}, AP(vs Real): {source_ap:.6f}"
                    file.write(line + '\n')
                    print(line)

        print("*****Average Training loss",str(trainLoss),"*****\n")
        if cfg.get("save_best_checkpoint", True):
            print(f"Epoch {epochID + 1}: ACC={pred_accuracy:.4%}; "
                  f"best epoch={max_epoch + 1}, ACC={max_acc:.4%}")
    end_time = time.time()

    return max_epoch, max_acc, end_time - ss_time
