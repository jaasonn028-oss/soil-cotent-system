import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 创建work文件夹
work_dir = 'work'
os.makedirs(work_dir, exist_ok=True)

class MultiModalSoilDataset(Dataset):
    """多模态土壤数据集类"""
    
    def __init__(self, sample_ids, organic_matter, pseudo_image_dir, real_image_dir, transform=None, mode='train'):
        self.sample_ids = sample_ids
        self.organic_matter = organic_matter
        self.pseudo_image_dir = pseudo_image_dir
        self.real_image_dir = real_image_dir
        self.transform = transform
        self.mode = mode
        
        # 检查哪些样本有真实图像
        self.has_real_image = []
        for sample_id in sample_ids:
            real_image_path = os.path.join(real_image_dir, f"{int(sample_id)}.jpg")
            self.has_real_image.append(os.path.exists(real_image_path))
        
        print(f"总样本数: {len(sample_ids)}")
        print(f"有真实图像的样本数: {sum(self.has_real_image)}")
        print(f"缺失真实图像的样本数: {len(sample_ids) - sum(self.has_real_image)}")
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        organic_matter_value = self.organic_matter[idx]
        
        # 加载伪图像（从work文件夹）
        pseudo_image_path = os.path.join(self.pseudo_image_dir, f"样本_{int(sample_id)}.png")
        try:
            pseudo_image = Image.open(pseudo_image_path)
            if pseudo_image.mode != 'RGB':
                pseudo_image = pseudo_image.convert('RGB')
        except:
            pseudo_image = Image.new('RGB', (64, 64), color=(128, 128, 128))
        
        # 加载真实图像
        real_image_path = os.path.join(self.real_image_dir, f"{int(sample_id)}.jpg")
        if os.path.exists(real_image_path):
            try:
                real_image = Image.open(real_image_path)
                if real_image.mode != 'RGB':
                    real_image = real_image.convert('RGB')
            except:
                real_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        else:
            real_image = Image.new('RGB', (224, 224), color=(200, 200, 200))
        
        # 应用图像变换
        if self.transform:
            pseudo_image = self.transform['pseudo'](pseudo_image)
            real_image = self.transform['real'](real_image)
        
        has_real_image = torch.tensor([1.0 if self.has_real_image[idx] else 0.0])
        target = torch.FloatTensor([organic_matter_value])
        
        return pseudo_image, real_image, has_real_image, target, sample_id

class MultiModalSoilModel(nn.Module):
    """多模态土壤有机质预测模型"""
    
    def __init__(self, pseudo_dropout=0.3, real_dropout=0.5):
        super(MultiModalSoilModel, self).__init__()
        
        # 伪图像分支
        self.pseudo_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(pseudo_dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 真实图像分支
        self.real_branch = models.resnet18(pretrained=True)
        self.real_branch.fc = nn.Sequential(
            nn.Linear(self.real_branch.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(real_dropout),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(real_dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 64 + 1, 32),  # 64(伪图像) + 64(真实图像) + 1(掩码)
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
    def forward(self, pseudo_x, real_x, mask):
        # 伪图像特征提取
        pseudo_features = self.pseudo_branch(pseudo_x)
        
        # 真实图像特征提取
        real_features = self.real_branch(real_x)
        
        # 根据图像可用性调整真实图像特征的权重
        mask_expanded = mask.expand(-1, real_features.size(1))
        real_features = real_features * mask_expanded
        
        # 特征融合
        combined = torch.cat([pseudo_features, real_features, mask], dim=1)
        output = self.fusion_net(combined)
        
        return output

def create_data_loaders(sample_ids, organic_matter, batch_size=8, fold_idx=None, train_indices=None, val_indices=None, use_all_data=False):
    """创建数据加载器"""
    
    # 图像变换
    train_transforms = {
        'pseudo': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        'real': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    val_transforms = {
        'pseudo': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        'real': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # 路径设置 - 伪图像从work文件夹读取
    pseudo_image_dir = os.path.join('work', '降维结果', '伪图像')
    real_image_dir = "图像数据"
    
    if fold_idx is not None:
        print(f"\n=== 第 {fold_idx + 1} 折交叉验证 ===")
    
    if use_all_data:
        # 使用全部数据
        train_idx = np.arange(len(sample_ids))
        val_idx = np.array([])  # 空验证集
        print("使用全部数据进行训练")
    elif train_indices is not None and val_indices is not None:
        train_idx = train_indices
        val_idx = val_indices
    else:
        # 如果没有传入索引，使用默认分割
        all_indices = np.arange(len(sample_ids))
        train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {len(train_idx)}")
    if len(val_idx) > 0:
        print(f"验证集大小: {len(val_idx)}")
    
    # 创建数据集
    train_dataset = MultiModalSoilDataset(
        sample_ids[train_idx], organic_matter[train_idx],
        pseudo_image_dir, real_image_dir,
        transform=train_transforms, mode='train'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_idx)), shuffle=True)
    
    if len(val_idx) > 0:
        val_dataset = MultiModalSoilDataset(
            sample_ids[val_idx], organic_matter[val_idx],
            pseudo_image_dir, real_image_dir,
            transform=val_transforms, mode='val'
        )
        val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_idx)), shuffle=False)
    else:
        val_loader = None
    
    return train_loader, val_loader, train_idx, val_idx

def train_model(model, train_loader, val_loader=None, epochs=100, patience=15, fold_idx=None):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 只有在有验证集时才使用学习率调度和早停
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    else:
        scheduler = None
    
    train_losses = []
    val_losses = [] if val_loader is not None else []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 创建模型结果保存目录
    model_output_dir = os.path.join(work_dir, '模型结果')
    if fold_idx is not None:
        model_output_dir = os.path.join(model_output_dir, f'fold_{fold_idx + 1}')
    else:
        model_output_dir = os.path.join(model_output_dir, 'final_model_all_data')
    os.makedirs(model_output_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for pseudo_imgs, real_imgs, masks, targets, _ in train_loader:
            pseudo_imgs = pseudo_imgs.to(device)
            real_imgs = real_imgs.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(pseudo_imgs, real_imgs, masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_train_loss)
        
        # 验证（只有在有验证集时执行）
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for pseudo_imgs, real_imgs, masks, targets, _ in val_loader:
                    pseudo_imgs = pseudo_imgs.to(device)
                    real_imgs = real_imgs.to(device)
                    masks = masks.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(pseudo_imgs, real_imgs, masks)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batch_count += 1
            
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            val_losses.append(avg_val_loss)
            
            if scheduler is not None:
                scheduler.step(avg_val_loss)
            
            # 早停检查（只有在有验证集时执行）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(model_output_dir, 'best_model.pth'))
            else:
                patience_counter += 1
        else:
            # 如果没有验证集，只保存训练损失最低的模型
            if avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                torch.save(model.state_dict(), os.path.join(model_output_dir, 'best_model.pth'))
        
        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if val_loader is not None:
                print(f'Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, '
                      f'验证损失: {avg_val_loss:.4f}, 学习率: {current_lr:.6f}, '
                      f'早停计数: {patience_counter}/{patience}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 学习率: {current_lr:.6f}')
        
        # 早停检查（只有在有验证集时执行）
        if val_loader is not None and patience_counter >= patience:
            print(f"早停于第 {epoch+1} 个epoch")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(model_output_dir, 'best_model.pth')))
    
    # 绘制损失曲线并保存
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_loader is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    if fold_idx is not None:
        plt.title(f'Training and Validation Loss - Fold {fold_idx + 1}')
    else:
        plt.title('Training Loss (All Data)')
        
    plt.grid(True, alpha=0.3)
    
    if fold_idx is not None:
        plt.savefig(os.path.join(model_output_dir, f'training_loss_fold_{fold_idx + 1}.png'), dpi=150, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(model_output_dir, 'training_loss_all_data.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return model, best_val_loss, model_output_dir

def evaluate_model(model, val_loader, sample_ids, organic_matter, val_idx, fold_idx=None):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    actuals = []
    sample_ids_list = []
    has_real_image_list = []
    
    with torch.no_grad():
        for pseudo_imgs, real_imgs, masks, targets, batch_sample_ids in val_loader:
            pseudo_imgs = pseudo_imgs.to(device)
            real_imgs = real_imgs.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            outputs = model(pseudo_imgs, real_imgs, masks)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
            sample_ids_list.extend(batch_sample_ids.cpu().numpy())
            has_real_image_list.extend(masks.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    sample_ids_list = np.array(sample_ids_list).flatten()
    has_real_image_list = np.array(has_real_image_list).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    print(f"第 {fold_idx + 1} 折评估结果:" if fold_idx is not None else "评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 按是否有真实图像分组评估
    if len(has_real_image_list) > 0:
        real_mask = has_real_image_list.astype(bool)
        
        if np.any(real_mask):
            mse_real = mean_squared_error(actuals[real_mask], predictions[real_mask])
            rmse_real = np.sqrt(mse_real)
            r2_real = r2_score(actuals[real_mask], predictions[real_mask])
            
            print(f"有真实图像的样本评估:")
            print(f"MSE: {mse_real:.4f}, RMSE: {rmse_real:.4f}, R²: {r2_real:.4f}")
        
        if np.any(~real_mask):
            mse_pseudo = mean_squared_error(actuals[~real_mask], predictions[~real_mask])
            rmse_pseudo = np.sqrt(mse_pseudo)
            r2_pseudo = r2_score(actuals[~real_mask], predictions[~real_mask])
            
            print(f"仅有伪图像的样本评估:")
            print(f"MSE: {mse_pseudo:.4f}, RMSE: {rmse_pseudo:.4f}, R²: {r2_pseudo:.4f}")
    
    # 绘制预测 vs 实际值
    plt.figure(figsize=(10, 8))
    
    colors = ['red' if has_real else 'blue' for has_real in has_real_image_list]
    
    plt.scatter(actuals, predictions, c=colors, alpha=0.7, s=60)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=2)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='With Real Image'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Only Pseudo Image')
    ]
    plt.legend(handles=legend_elements)
    
    plt.xlabel('Actual Organic Matter')
    plt.ylabel('Predicted Organic Matter')
    title = f'Predicted vs Actual - Fold {fold_idx + 1}\nR² = {r2:.4f}, RMSE = {rmse:.4f}' if fold_idx is not None else f'Predicted vs Actual\nR² = {r2:.4f}, RMSE = {rmse:.4f}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'Sample_ID': sample_ids_list,
        'Actual_Organic_Matter': actuals,
        'Predicted_Organic_Matter': predictions,
        'Has_Real_Image': has_real_image_list.astype(int)
    })
    results_df['Absolute_Error'] = np.abs(results_df['Actual_Organic_Matter'] - results_df['Predicted_Organic_Matter'])
    
    # 确保输出目录存在
    outcome_dir = os.path.join(work_dir, '模型结果')
    os.makedirs(outcome_dir, exist_ok=True)
    
    if fold_idx is not None:
        plt.savefig(os.path.join(outcome_dir, f'prediction_results_fold_{fold_idx + 1}.png'), dpi=150, bbox_inches='tight')
        results_df.to_excel(os.path.join(outcome_dir, f'prediction_results_fold_{fold_idx + 1}.xlsx'), index=False)
    else:
        plt.savefig(os.path.join(outcome_dir, 'prediction_results.png'), dpi=150, bbox_inches='tight')
        results_df.to_excel(os.path.join(outcome_dir, 'prediction_results.xlsx'), index=False)
    
    plt.close()
    
    return mse, rmse, r2, results_df

def cross_validation(sample_ids, organic_matter, n_splits=5, epochs=100, batch_size=8):
    """执行五折交叉验证"""
    print("开始五折交叉验证...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    all_sample_ids = []
    all_has_real_image = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(sample_ids)):
        print(f"\n=== 第 {fold_idx + 1} 折交叉验证 ===")
        print(f"训练样本数: {len(train_idx)}")
        print(f"验证样本数: {len(val_idx)}")
        
        # 创建数据加载器
        train_loader, val_loader, _, _ = create_data_loaders(
            sample_ids, organic_matter, batch_size=batch_size, fold_idx=fold_idx, 
            train_indices=train_idx, val_indices=val_idx
        )
        
        # 创建模型
        model = MultiModalSoilModel()
        
        # 训练模型
        model, best_val_loss, model_dir = train_model(
            model, train_loader, val_loader, epochs=epochs, patience=15, fold_idx=fold_idx
        )
        
        # 评估模型
        mse, rmse, r2, results_df = evaluate_model(
            model, val_loader, sample_ids, organic_matter, val_idx, fold_idx=fold_idx
        )
        
        # 保存结果
        fold_results.append({
            'fold': fold_idx + 1,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'best_val_loss': best_val_loss
        })
        
        # 收集所有预测结果
        all_predictions.extend(results_df['Predicted_Organic_Matter'].values)
        all_actuals.extend(results_df['Actual_Organic_Matter'].values)
        all_sample_ids.extend(results_df['Sample_ID'].values)
        all_has_real_image.extend(results_df['Has_Real_Image'].values)
        
        print(f"第 {fold_idx + 1} 折完成")
    
    # 计算交叉验证总体结果
    overall_mse = mean_squared_error(all_actuals, all_predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_r2 = r2_score(all_actuals, all_predictions)
    
    print(f"\n=== 五折交叉验证总体结果 ===")
    print(f"总体 MSE: {overall_mse:.4f}")
    print(f"总体 RMSE: {overall_rmse:.4f}")
    print(f"总体 R²: {overall_r2:.4f}")
    
    # 绘制总体预测结果
    plt.figure(figsize=(10, 8))
    
    colors = ['red' if has_real else 'blue' for has_real in all_has_real_image]
    
    plt.scatter(all_actuals, all_predictions, c=colors, alpha=0.7, s=60)
    plt.plot([min(all_actuals), max(all_actuals)], [min(all_actuals), max(all_actuals)], 'k--', lw=2)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='With Real Image'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Only Pseudo Image')
    ]
    plt.legend(handles=legend_elements)
    
    plt.xlabel('Actual Organic Matter')
    plt.ylabel('Predicted Organic Matter')
    plt.title(f'5-Fold Cross Validation Overall Results\nR² = {overall_r2:.4f}, RMSE = {overall_rmse:.4f}')
    plt.grid(True, alpha=0.3)
    
    # 确保输出目录存在
    outcome_dir = os.path.join(work_dir, '模型结果')
    os.makedirs(outcome_dir, exist_ok=True)
    plt.savefig(os.path.join(outcome_dir, 'cv_overall_results.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存交叉验证结果
    cv_results_df = pd.DataFrame(fold_results)
    cv_results_df.loc[len(cv_results_df)] = ['Overall', overall_mse, overall_rmse, overall_r2, np.nan]
    cv_results_df.to_excel(os.path.join(outcome_dir, 'cross_validation_results.xlsx'), index=False)
    
    # 保存总体预测结果
    overall_results_df = pd.DataFrame({
        'Sample_ID': all_sample_ids,
        'Actual_Organic_Matter': all_actuals,
        'Predicted_Organic_Matter': all_predictions,
        'Has_Real_Image': all_has_real_image
    })
    overall_results_df['Absolute_Error'] = np.abs(overall_results_df['Actual_Organic_Matter'] - overall_results_df['Predicted_Organic_Matter'])
    overall_results_df.to_excel(os.path.join(outcome_dir, 'cv_overall_predictions.xlsx'), index=False)
    
    return cv_results_df, overall_results_df

def main():
    """主函数"""
    print("开始多模态土壤有机质含量预测（五折交叉验证）...")
    
    # 1. 加载数据
    print("步骤1: 加载数据...")
    
    # 读取降维后的数据
    pca_data_path = os.path.join('work', '降维结果', '降维后的光谱数据.xlsx')
    if not os.path.exists(pca_data_path):
        print("错误: 未找到降维后的数据文件，请先运行生成伪图像的代码！")
        return
    
    pca_df = pd.read_excel(pca_data_path)
    sample_ids = pca_df['样本编号'].values
    organic_matter = pca_df['有机质含量'].values
    
    print(f"样本数量: {len(sample_ids)}")
    print(f"有机质含量范围: {organic_matter.min():.2f} - {organic_matter.max():.2f}")
    
    # 2. 执行五折交叉验证
    print("步骤2: 执行五折交叉验证...")
    cv_results, overall_results = cross_validation(
        sample_ids, organic_matter, n_splits=5, epochs=100, batch_size=8
    )
    
    print("五折交叉验证完成！")
    print(f"总体 R²: {cv_results.iloc[-1]['r2']:.4f}")
    print(f"总体 RMSE: {cv_results.iloc[-1]['rmse']:.4f}")
    
    # 3. 训练最终模型（使用全部数据）
    print("步骤3: 训练最终模型（使用全部数据）...")
    
    # 创建最终训练的数据加载器（使用全部数据）
    train_loader, val_loader, train_idx, val_idx = create_data_loaders(
        sample_ids, organic_matter, batch_size=8, use_all_data=True
    )
    
    # 创建最终模型
    final_model = MultiModalSoilModel()
    
    total_params = sum(p.numel() for p in final_model.parameters())
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 训练最终模型（使用全部数据，无验证集）
    # 使用交叉验证中观察到的平均收敛轮数
    optimal_epochs = 50  # 根据交叉验证结果调整
    final_model, best_loss, model_dir = train_model(
        final_model, train_loader, val_loader=None, epochs=optimal_epochs
    )
    
    # 保存最终模型
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'optimal_epochs': optimal_epochs,
        'final_loss': best_loss
    }, os.path.join(model_dir, 'final_model_all_data.pth'))
    
    print(f"最终模型已保存到: {os.path.join(model_dir, 'final_model_all_data.pth')}")
    print("训练完成!")

if __name__ == "__main__":
    main()