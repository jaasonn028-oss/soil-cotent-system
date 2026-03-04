import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # 无GUI模式
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# 确保所有计算在CPU上进行
device = torch.device("cpu")

# 自定义激活函数模块
class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

# SE注意力模块（用于带SE的倒置残差块）
class SqueezeExcite1D(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)

# 倒置残差块（1D版本）
class InvertedResidual1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, use_se=False):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_res = stride == 1 and in_ch == out_ch
        self.use_se = use_se
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                h_swish()
            ])
        
        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            h_swish(),
            
            SqueezeExcite1D(hidden_dim) if use_se else nn.Identity(),
            
            nn.Conv1d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

# 数据预处理模块（保持不变）
def load_spectral_data(file_path):
    """加载光谱数据并进行预处理"""
    data = pd.read_excel(file_path, header=None)
    spectrum = data.iloc[0, 1:].values.astype(float)
    reflectivity = data.iloc[1:, 1:].values.astype(float)
    target = data.iloc[1:, 0].values.astype(float)
    
    # 数据标准化（按样本标准化）
    reflectivity = (reflectivity - reflectivity.mean(axis=1, keepdims=True)) / (
        reflectivity.std(axis=1, keepdims=True) + 1e-6)
    return torch.FloatTensor(reflectivity).to(device), torch.FloatTensor(target).to(device)

# 优化后的数据加载器（保持CPU设置）
class SpectralDataLoader:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        
    def create_loaders(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=test_size, random_state=42
        )
        
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train).to(device),
                torch.FloatTensor(y_train).to(device)
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        test_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_test).to(device),
                torch.FloatTensor(y_test).to(device)
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        return train_loader, test_loader

# 轻量化MobileNetV3（通道数压缩版）
class MobileNetV3_1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super().__init__()
        # 输入特征调整（减少输出通道）
        self.feature_adjust = nn.Sequential(
            nn.Conv1d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(8),
            h_swish()
        )
        
        # 精简的倒置残差块配置
        self.blocks = nn.Sequential(
            InvertedResidual1D(8, 12, stride=2, expand_ratio=3, use_se=False),
            InvertedResidual1D(12, 12, stride=1, expand_ratio=2, use_se=False),
            InvertedResidual1D(12, 16, stride=2, expand_ratio=3, use_se=True),
            InvertedResidual1D(16, 24, stride=2, expand_ratio=4, use_se=True)
        )
        
        # 输出层优化
        self.final_conv = nn.Sequential(
            nn.Conv1d(24, 48, 1, bias=False),
            nn.BatchNorm1d(48),
            h_swish()
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.Hardswish(),
            nn.Dropout(0.1),
            nn.Linear(24, num_classes)
        )
        self.train_losses = []    # 新增训练损失记录
        self.test_losses = []     # 新增测试损失记录

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        x = self.feature_adjust(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

# 训练流程优化（添加CPU设置）
class MobileNetTrainer:
    def __init__(self, model, lr=0.001, patience=1000):
        self.model = model.to(device)
        self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=5
        )
        self.best_loss = float('inf')
        self.patience = patience
        self.counter = 0
        self.train_losses = []
        self.test_losses = []    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = F.mse_loss(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                total_loss += F.mse_loss(pred.squeeze(), y).item()
        return total_loss / len(test_loader)
    
    def run(self, train_loader, test_loader, epochs=200):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            test_loss = self.validate(test_loader)
            self.scheduler.step(test_loss)
            self.train_losses.append(train_loss)#为保持loss
            self.test_losses.append(test_loss)    
            if test_loss < self.best_loss:#！！！如果不要早停从本行开始注释
                self.best_loss = test_loss
                self.counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")#早停机制
                    break
            #注释到本行结束
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        self._save_loss_plot()
        
        # 加载最佳模型进行最终评估
        self.model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
        final_r2 = self._save_final_evaluation(test_loader)
        return final_r2
    def _save_loss_plot(self):
        """保存训练损失曲线图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Training Process')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(os.path.join(folder_path, 'training_loss.png'))
        plt.close()
    
    def _save_final_evaluation(self, test_loader):
        """最终评估并保存结果"""
        # 收集预测结果
        all_preds, all_targets = [], []
        self.model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                all_preds.extend(pred.squeeze().cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        # 计算R²
        r2 = r2_score(all_targets, all_preds)
        
        # 保存预测结果
        results = pd.DataFrame({
            'True': all_targets,
            'Predicted': all_preds
        })
        results.to_csv(os.path.join(folder_path, 'predictions.csv'), index=False)
        
        # 保存R²散点图
        self._save_r2_plot(all_targets, all_preds, r2)
        
        # 保存评估报告
        with open(os.path.join(folder_path, 'report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Best Test Loss: {self.best_loss:.4f}\n")
            f.write(f"Final R² Score: {r2:.4f}\n")
        
        return r2

    def _save_r2_plot(self, y_true, y_pred, r2):
        """保存R²可视化图"""
        plt.figure(figsize=(8, 8))
        
        # 绘制散点
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # 绘制理想线
        max_val = max(max(y_true), max(y_pred))
        min_val = min(min(y_true), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # 添加标注
        plt.title(f'Prediction vs True Value (R² = {r2:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(alpha=0.3)
        
        plt.savefig(os.path.join(folder_path, 'r2_visualization.png'))
        plt.close()

# 主程序入口
if __name__ == "__main__":
    # 创建保存目录（保持原始设置）
    folder_path = 'saved_files'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 数据加载
    X, y = load_spectral_data('95-海洋光谱仪.xlsx')
    loader = SpectralDataLoader(batch_size=16)
    train_loader, test_loader = loader.create_loaders(X, y)
    
    # 模型初始化
    model = MobileNetV3_1D(input_channels=1, num_classes=1)
    
    # 训练执行
    trainer = MobileNetTrainer(model, lr=0.001, patience=1000)
    final_r2 =trainer.run(train_loader, test_loader, epochs=500)

    # 保存模型  
    
    # 打印最终结果
    print(f"\n{'='*30}")
    print(f"Training Completed!")
    print(f"Best Model Saved to: best_model.pth")
    print(f"Visualization Saved to: {folder_path}")
    print(f"Final R² Score: {final_r2:.4f}")
    print(f"{'='*30}\n")
