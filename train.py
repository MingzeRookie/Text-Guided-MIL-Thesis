import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import wandb 
try:
    from src.datasets.wsi_multimodal_spatial_dataset import WsiMultimodalSpatialDataset
    from src.models.multimodal_text_guided_mil import MultimodalTextGuidedMIL
    from src.utils.data_utils import multimodal_spatial_collate_fn
    print("成功从 src.datasets, src.models, src.utils 导入自定义模块。")
except ImportError as e:
    print(f"导入自定义模块时出错: {e}")
    print("请确保 train.py 位于 BMW/ 目录下，并且 BMW/src/ 目录及其子模块存在且包含 __init__.py 文件。")
    WsiMultimodalSpatialDataset = None
    MultimodalTextGuidedMIL = None
    multimodal_spatial_collate_fn = None

logger = logging.getLogger(__name__)

def get_metrics(y_true, y_pred_probs, y_pred_labels, num_classes):
    """计算并返回分类指标"""
    metrics = {}
    y_true_np = np.array(y_true)
    y_pred_probs_np = np.array(y_pred_probs)
    y_pred_labels_np = np.array(y_pred_labels)

    if not y_true_np.size or not y_pred_probs_np.size or not y_pred_labels_np.size:
        logger.warning("计算指标时发现空数组，将返回默认指标。")
        return {'auc': 0.5, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}


    if num_classes == 2:
        try:
            probs_for_auc = y_pred_probs_np if y_pred_probs_np.ndim == 1 else y_pred_probs_np[:, 1]
            if len(np.unique(y_true_np)) < 2: # 检查标签是否只有一个类别
                 metrics['auc'] = 0.5 # 或者 np.nan，取决于您希望如何处理
                 logger.warning("计算二分类AUC时只有一个类别存在于y_true中，AUC设为0.5。")
            else:
                metrics['auc'] = roc_auc_score(y_true_np, probs_for_auc)
        except ValueError as e:
            logger.warning(f"计算二分类AUC时出错: {e}. y_true unique: {np.unique(y_true_np)}. AUC设为0.5。")
            metrics['auc'] = 0.5 
        avg_method = 'binary'
    else: 
        try:
            if len(np.unique(y_true_np)) < 2:
                metrics['auc'] = 0.5
                logger.warning("计算多分类AUC时只有一个类别存在于y_true中，AUC设为0.5。")
            else:
                metrics['auc'] = roc_auc_score(y_true_np, y_pred_probs_np, multi_class='ovr', average='weighted')
        except ValueError as e:
            logger.warning(f"计算多分类AUC时出错: {e}. y_true unique: {np.unique(y_true_np)}, y_pred_probs shape: {y_pred_probs_np.shape}. AUC设为0.5。")
            metrics['auc'] = 0.5 
        avg_method = 'weighted'
        
    metrics['f1'] = f1_score(y_true_np, y_pred_labels_np, average=avg_method, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_true_np, y_pred_labels_np)
    metrics['precision'] = precision_score(y_true_np, y_pred_labels_np, average=avg_method, zero_division=0)
    metrics['recall'] = recall_score(y_true_np, y_pred_labels_np, average=avg_method, zero_division=0)
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num, num_epochs, num_classes, grad_clip_norm=None, log_to_wandb=False): # log_to_wandb 默认设为 False
    model.train()
    total_loss = 0.0
    batch_losses = []
    all_labels_list = []
    all_preds_probs_for_auc_list = [] 
    all_preds_labels_list = [] 

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{num_epochs} [训练中]", leave=False)
    for batch_idx, batch_data in enumerate(progress_bar):
        (image_patch_features_b, patch_grid_indices_b, 
         text_feat_b, labels_b, grid_shapes_b,
         original_patch_coordinates_b, patch_mask_b) = batch_data

        image_patch_features_b = image_patch_features_b.to(device)
        patch_grid_indices_b = patch_grid_indices_b.to(device)
        text_feat_b = text_feat_b.to(device)
        labels_b = labels_b.to(device)
        grid_shapes_b = grid_shapes_b.to(device)
        patch_mask_b = patch_mask_b.to(device)

        optimizer.zero_grad()
        # logits = model(
        #     image_patch_features_batch=image_patch_features_b,
        #     patch_grid_indices_batch=patch_grid_indices_b,
        #     text_feat_batch=text_feat_b, # 移除或设为None
        #     grid_shapes_batch=grid_shapes_b,
        #     patch_mask_batch=patch_mask_b
        # )
        # 修改为:
        logits = model(
            image_patch_features_batch=image_patch_features_b,
            patch_grid_indices_batch=patch_grid_indices_b,
            grid_shapes_batch=grid_shapes_b,
            # text_feat_batch=text_feat_b, # 如果模型forward签名已修改为可选，则可省略
            patch_mask_batch=patch_mask_b
        )

        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"Epoch {epoch_num+1}, Batch {batch_idx}: Logits 包含 NaN 或 Inf！跳过此批次。Logits: {logits}")
            continue

        loss = criterion(logits, labels_b)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Epoch {epoch_num+1}, Batch {batch_idx}: 损失为 NaN 或 Inf！Logits: {logits}, Labels: {labels_b}")
            continue 

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        
        batch_losses.append(loss.item()) # 记录每个有效batch的loss
        total_loss += loss.item() # 累加有效loss

        probs_all_classes = F.softmax(logits, dim=1)
        if num_classes == 2:
            preds_probs_for_auc_batch = probs_all_classes[:, 1].detach().cpu().numpy()
        else:
            preds_probs_for_auc_batch = probs_all_classes.detach().cpu().numpy()
        preds_labels_batch = torch.argmax(logits, dim=1).detach().cpu().numpy()
        
        all_labels_list.extend(labels_b.cpu().numpy().tolist())
        if preds_probs_for_auc_batch.ndim == 1: # 二分类，每个样本一个概率值
            all_preds_probs_for_auc_list.extend(preds_probs_for_auc_batch.tolist())
        else: # 多分类，每个样本是概率分布列表
            all_preds_probs_for_auc_list.extend(preds_probs_for_auc_batch.tolist()) 
            
        all_preds_labels_list.extend(preds_labels_batch.tolist())

        progress_bar.set_postfix(loss=loss.item())
        # if log_to_wandb and batch_idx > 0 and batch_idx % 20 == 0: 
        #      wandb.log({"train/batch_loss": loss.item(), "epoch_float": epoch_num + (batch_idx/len(loader))})


    avg_loss = np.mean(batch_losses) if batch_losses else float('nan') # 计算有效batch的平均loss
    
    if not all_labels_list:
        logger.warning(f"训练 Epoch {epoch_num+1}: 没有有效的标签或预测来计算指标。")
        epoch_metrics = {'auc': 0.5, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    else:
        epoch_metrics = get_metrics(
            np.array(all_labels_list), 
            np.array(all_preds_probs_for_auc_list), 
            np.array(all_preds_labels_list), 
            num_classes
        )
    
    log_data_train = {"train/epoch_loss": avg_loss, "epoch": epoch_num + 1}
    for key, value in epoch_metrics.items():
        log_data_train[f"train/{key}"] = value
    
    # if log_to_wandb:
    #     wandb.log(log_data_train)

    logger.info(f"训练 Epoch {epoch_num+1} 平均损失: {avg_loss:.4f}, AUC: {epoch_metrics['auc']:.4f}, F1 (weighted): {epoch_metrics['f1']:.4f}")
    return avg_loss, epoch_metrics

def evaluate(model, loader, criterion, device, epoch_num, num_classes, log_to_wandb=False): # log_to_wandb 默认设为 False
    model.eval()
    total_loss = 0.0
    batch_losses = []
    all_labels_list = []
    all_preds_probs_for_auc_list = []
    all_preds_labels_list = []

    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1} [评估中]", leave=False)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            (image_patch_features_b, patch_grid_indices_b, 
             text_feat_b, labels_b, grid_shapes_b,
             original_patch_coordinates_b, patch_mask_b) = batch_data

            image_patch_features_b = image_patch_features_b.to(device)
            patch_grid_indices_b = patch_grid_indices_b.to(device)
            text_feat_b = text_feat_b.to(device)
            labels_b = labels_b.to(device)
            grid_shapes_b = grid_shapes_b.to(device)
            patch_mask_b = patch_mask_b.to(device)

            logits = model(
                image_patch_features_batch=image_patch_features_b,
                patch_grid_indices_batch=patch_grid_indices_b,
                text_feat_batch=text_feat_b,
                grid_shapes_batch=grid_shapes_b,
                patch_mask_batch=patch_mask_b
            )
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error(f"评估 Epoch {epoch_num+1}, Batch {batch_idx}: Logits 包含 NaN 或 Inf！")
            
            loss = criterion(logits, labels_b)
            if not (torch.isnan(loss) or torch.isinf(loss)):
                batch_losses.append(loss.item())
                total_loss += loss.item()
            else:
                logger.warning(f"评估 Epoch {epoch_num+1}, Batch {batch_idx}: 损失为 NaN 或 Inf！")

            probs_all_classes = F.softmax(logits, dim=1)
            if num_classes == 2:
                preds_probs_for_auc_batch = probs_all_classes[:, 1].cpu().numpy()
            else:
                preds_probs_for_auc_batch = probs_all_classes.cpu().numpy()
            preds_labels_batch = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_labels_list.extend(labels_b.cpu().numpy().tolist())
            if preds_probs_for_auc_batch.ndim == 1:
                 all_preds_probs_for_auc_list.extend(preds_probs_for_auc_batch.tolist())
            else:
                 all_preds_probs_for_auc_list.extend(preds_probs_for_auc_batch.tolist())
            all_preds_labels_list.extend(preds_labels_batch.tolist())
            
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = np.mean(batch_losses) if batch_losses else float('nan')
    
    if not all_labels_list:
        logger.warning(f"评估 Epoch {epoch_num+1}: 没有有效的标签或预测来计算指标。")
        epoch_metrics = {'auc': 0.5, 'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
    else:
        epoch_metrics = get_metrics(
            np.array(all_labels_list), 
            np.array(all_preds_probs_for_auc_list), 
            np.array(all_preds_labels_list), 
            num_classes
        )
    
    log_data_val = {"val/epoch_loss": avg_loss, "epoch": epoch_num + 1}
    for key, value in epoch_metrics.items():
        log_data_val[f"val/{key}"] = value

    # if log_to_wandb:
    #     wandb.log(log_data_val)

    logger.info(f"评估 Epoch {epoch_num+1} 平均损失: {avg_loss:.4f}, AUC: {epoch_metrics['auc']:.4f}, F1 (weighted): {epoch_metrics['f1']:.4f}")
    return avg_loss, epoch_metrics

@hydra.main(config_path="configs", config_name="config_bmw_multimodal_spatial_baseline", version_base=None)
def main(cfg: DictConfig):
    # --- 2. 初始化 wandb (暂时注释掉) ---
    # run_name = cfg.experiment_params.get("run_name", None) 
    # if run_name is None or run_name == "null":
    #     run_name = f"{cfg.model_params.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    #
    # wandb.init(
    #     project=cfg.experiment_params.get("wandb_project", "BMW-Multimodal-MIL"), 
    #     entity=cfg.experiment_params.get("wandb_entity", None), 
    #     name=run_name,
    #     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 
    # )
    
    # try: # try...finally for wandb.finish()
    if WsiMultimodalSpatialDataset is None or MultimodalTextGuidedMIL is None or multimodal_spatial_collate_fn is None:
        logger.error("自定义模块未能加载，请检查导入路径和环境。程序将退出。")
        # wandb.finish(exit_code=1) # 暂时注释掉
        return

    torch.manual_seed(cfg.experiment_params.seed)
    np.random.seed(cfg.experiment_params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.experiment_params.seed)

    device = torch.device(f"cuda:{cfg.experiment_params.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info("加载的配置:\n%s", OmegaConf.to_yaml(cfg))
    
    num_classes_from_config = cfg.model_params.num_classes

    logger.info("正在加载训练数据集...")
    train_dataset = WsiMultimodalSpatialDataset(
        data_manifest_csv_path=cfg.data.train_manifest_csv_path,
        image_feature_dir=cfg.data.image_feature_dir,
        text_feature_path=cfg.data.text_feature_path,
        patch_size_pixels=list(cfg.data.patch_size_pixels),
        label_column_name=cfg.data.label_column_name,
        wsi_id_column_name=cfg.data.wsi_id_column_name
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_params.batch_size,
        shuffle=True,
        num_workers=cfg.experiment_params.num_workers,
        collate_fn=multimodal_spatial_collate_fn,
        pin_memory=True
    )
    logger.info(f"训练数据集加载完成，样本数: {len(train_dataset)}")

    logger.info("正在加载验证数据集...")
    val_dataset = WsiMultimodalSpatialDataset(
        data_manifest_csv_path=cfg.data.val_manifest_csv_path,
        image_feature_dir=cfg.data.image_feature_dir,
        text_feature_path=cfg.data.text_feature_path,
        patch_size_pixels=list(cfg.data.patch_size_pixels),
        label_column_name=cfg.data.label_column_name,
        wsi_id_column_name=cfg.data.wsi_id_column_name
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train_params.batch_size, 
        shuffle=False,
        num_workers=cfg.experiment_params.num_workers,
        collate_fn=multimodal_spatial_collate_fn,
        pin_memory=True
    )
    logger.info(f"验证数据集加载完成，样本数: {len(val_dataset)}")

    logger.info(f"正在实例化模型: {cfg.model_params.name}")
    model_config_for_init = OmegaConf.create({
        "patch_feature_dim": cfg.model_params.patch_feature_dim,
        "text_feature_dim": cfg.model_params.text_feature_dim,
        "num_classes": num_classes_from_config,
        "model_params": cfg.model_params 
    })
    model = MultimodalTextGuidedMIL(config=model_config_for_init).to(device)
    logger.info("模型实例化完成。")
    
    # if cfg.experiment_params.get("wandb_watch_model", False): # 暂时禁用 watch
    #     wandb.watch(model, log=cfg.experiment_params.get("wandb_watch_log_level", "all"), 
    #                 log_freq=cfg.experiment_params.get("wandb_watch_log_freq", 100)) 

    criterion = nn.CrossEntropyLoss()
    if cfg.train_params.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=cfg.train_params.learning_rate, weight_decay=cfg.train_params.weight_decay)
    elif cfg.train_params.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.train_params.learning_rate, weight_decay=cfg.train_params.weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {cfg.train_params.optimizer}")

    scheduler = None
    if cfg.train_params.scheduler and cfg.train_params.scheduler.lower() != 'none':
        if cfg.train_params.scheduler.lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train_params.num_epochs - cfg.train_params.get('warmup_epochs', 0), eta_min=cfg.train_params.get('eta_min', 1e-6))
        elif cfg.train_params.scheduler.lower() == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)
    
    best_val_auc = 0.0
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"实验输出将保存在: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    history_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_auc', 'train_f1', 
                                       'val_loss', 'val_auc', 'val_f1'])
    
    grad_clip_norm_val = cfg.train_params.get("grad_clip_norm", None)

    for epoch in range(cfg.train_params.num_epochs):
        start_time = time.time()
        
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg.train_params.num_epochs, num_classes_from_config, grad_clip_norm=grad_clip_norm_val, log_to_wandb=False) # log_to_wandb=False
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, epoch, num_classes_from_config, log_to_wandb=False) # log_to_wandb=False

        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            # if cfg.experiment_params.get("use_wandb", False): # 暂时注释掉
            #     wandb.log({"train/learning_rate": current_lr, "epoch": epoch + 1})
            logger.info(f"Epoch {epoch+1}: 当前学习率: {current_lr}") # 添加本地日志记录学习率
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])
            elif epoch >= cfg.train_params.get('warmup_epochs', 0) : 
                scheduler.step()
        
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_auc': train_metrics['auc'], 'train_f1': train_metrics['f1'],
            'val_loss': val_loss, 'val_auc': val_metrics['auc'], 'val_f1': val_metrics['f1']
        }
        history_df = pd.concat([history_df, pd.DataFrame([epoch_data])], ignore_index=True)
        history_df.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            checkpoint_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'config': OmegaConf.to_container(cfg) 
            }, checkpoint_path)
            logger.info(f"Epoch {epoch+1}: 新的最佳模型已保存到 {checkpoint_path} (验证 AUC: {best_val_auc:.4f})")
            # if cfg.experiment_params.get("wandb_save_best_model", False) and cfg.experiment_params.get("use_wandb", False): # 暂时注释掉
            #     wandb.save(checkpoint_path, base_path=output_dir, policy="live") 
        
        latest_checkpoint_path = os.path.join(output_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_auc': val_metrics['auc'],
            'config': OmegaConf.to_container(cfg)
        }, latest_checkpoint_path)

        end_time = time.time()
        logger.info(f"Epoch {epoch+1} 完成，耗时: {end_time - start_time:.2f} 秒")

    logger.info("训练完成！")
    logger.info(f"最佳验证 AUC: {best_val_auc:.4f}")
    # if cfg.experiment_params.get("use_wandb", False): # 暂时注释掉
    #     wandb.summary["best_val_auc"] = best_val_auc 

    # finally: # try...finally for wandb.finish() (暂时注释掉)
        # if cfg.experiment_params.get("use_wandb", False):
        #     wandb.finish()


if __name__ == "__main__":
    main()
