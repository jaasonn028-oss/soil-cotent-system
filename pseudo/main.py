# generate_pseudo_images.py
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import matplotlib.pyplot as plt

# 设置matplotlib字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 创建work文件夹
work_dir = 'work'
os.makedirs(work_dir, exist_ok=True)

def generate_pseudo_images():
    """生成伪图像的主函数"""
    print("开始降维和伪图像生成...")
    
    # 读取Excel文件
    file_path = '光谱数据.xlsx'
    data = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
    
    # 提取样本编号和有机质含量
    sample_ids = data.iloc[:, 0].values
    organic_matter = data.iloc[:, 1].values
    
    # 提取光谱数据（从第三列开始）
    spectral_data = data.iloc[:, 2:].values
    
    print(f"数据形状: {spectral_data.shape}")
    print(f"样本数量: {len(sample_ids)}")
    print(f"波段数量: {spectral_data.shape[1]}")
    
    # 标准化光谱数据
    scaler = StandardScaler()
    spectral_data_scaled = scaler.fit_transform(spectral_data)
    
    # 执行PCA降维
    n_components = min(50, spectral_data.shape[1], spectral_data.shape[0])
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(spectral_data_scaled)
    
    print(f"降维后保留的主成分数量: {pca_result.shape[1]}")
    print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 在work文件夹内创建保存结果的子文件夹
    output_dir = os.path.join(work_dir, '降维结果')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存降维后的数据到work文件夹
    pca_df = pd.DataFrame({
        '样本编号': sample_ids,
        '有机质含量': organic_matter
    })
    for i in range(pca_result.shape[1]):
        pca_df[f'主成分_{i+1}'] = pca_result[:, i]
    
    pca_df.to_excel(os.path.join(output_dir, '降维后的光谱数据.xlsx'), index=False)
    
    # 选择前三个主成分作为RGB通道
    optimal_components = [0, 1, 2]
    
    # 在work文件夹内创建伪图像保存目录
    images_dir = os.path.join(output_dir, '伪图像')
    os.makedirs(images_dir, exist_ok=True)
    
    # 将主成分值映射到RGB颜色的函数
    def map_pc_to_rgb(pc_values, component_indices):
        if len(component_indices) < 3:
            component_indices = list(range(min(3, len(pc_values))))
        
        pc1 = pc_values[component_indices[0]]
        pc2 = pc_values[component_indices[1]]
        pc3 = pc_values[component_indices[2]]
        
        global_min = np.min(pca_result[:, component_indices])
        global_max = np.max(pca_result[:, component_indices])
        
        if global_max > global_min:
            r = (pc1 - global_min) / (global_max - global_min)
            g = (pc2 - global_min) / (global_max - global_min)
            b = (pc3 - global_min) / (global_max - global_min)
        else:
            r = g = b = 0.5
        
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        
        return r, g, b
    
    # 创建带有噪声的伪图像函数
    def create_noisy_pseudo_image(pc_values, component_indices, image_size=(64, 64), noise_intensity=0.1):
        base_r, base_g, base_b = map_pc_to_rgb(pc_values, component_indices)
        
        width, height = image_size
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        image_array[:, :, 0] = base_r
        image_array[:, :, 1] = base_g
        image_array[:, :, 2] = base_b
        
        if noise_intensity > 0:
            noise = np.random.normal(0, noise_intensity * 64, (height, width, 3))
            noisy_image = image_array.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        else:
            noisy_image = image_array
        
        return Image.fromarray(noisy_image)
    
    # 为每个样本创建伪图像并保存到work文件夹
    image_paths = []
    channel_info = []
    
    print("开始为每个样本生成伪图像...")
    
    for i, (sample_id, pc_values) in enumerate(zip(sample_ids, pca_result)):
        try:
            image = create_noisy_pseudo_image(
                pc_values, optimal_components, 
                image_size=(64, 64), noise_intensity=0.1
            )
            
            image_path = os.path.join(images_dir, f'样本_{int(sample_id)}.png')
            image.save(image_path)
            image_paths.append(image_path)
            
            base_r, base_g, base_b = map_pc_to_rgb(pc_values, optimal_components)
            channel_info.append({
                '样本编号': int(sample_id),
                '有机质含量': organic_matter[i],
                '使用的主成分': f'PC{optimal_components[0]+1}, PC{optimal_components[1]+1}, PC{optimal_components[2]+1}',
                'RGB颜色_R': base_r,
                'RGB颜色_G': base_g,
                'RGB颜色_B': base_b,
                'PC1值': pc_values[optimal_components[0]],
                'PC2值': pc_values[optimal_components[1]],
                'PC3值': pc_values[optimal_components[2]]
            })
            
            print(f"已创建样本 {sample_id} 的伪图像")
            
        except Exception as e:
            print(f"创建样本 {sample_id} 的图像时出错: {e}")
    
    # 保存通道使用信息到work文件夹
    channel_df = pd.DataFrame(channel_info)
    channel_df.to_excel(os.path.join(output_dir, '图像生成信息.xlsx'), index=False)
    
    # 绘制主成分解释方差图
    plt.figure(figsize=(10, 6))
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
    plt.xlabel('主成分')
    plt.ylabel('解释方差比例')
    plt.title('各主成分解释方差')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95%方差线')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差')
    plt.title('累计解释方差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '主成分分析结果.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n降维和伪图像生成完成！")
    print(f"降维后的数据已保存到: {os.path.join(output_dir, '降维后的光谱数据.xlsx')}")
    print(f"伪图像已保存到: {images_dir}")
    print(f"图像数量: {len(sample_ids)}")
    print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")
    
    return sample_ids, organic_matter, images_dir

if __name__ == "__main__":
    generate_pseudo_images()