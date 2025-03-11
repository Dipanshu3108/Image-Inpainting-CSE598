import matplotlib.pyplot as plt
import numpy as np

def create_all_metrics_plot():
    shapes = ['Line', 'Triangle', 'Circle', 'Rectangle']
    metrics = ['SSIM', 'MSE', 'PSNR', 'FID', 'KID', 'LPIPS']
    
    # Data
    pde_data = {
        'SSIM': [0.988, 0.94, 0.950, 0.888],
        'MSE': [24.201, 193.603, 184.201, 511.232],
        'PSNR': [41.699, 30.329, 30.029, 24.103],
        'FID': [13.917, 29.962, 21.985, 49.792],
        'KID': [0.007, 0.012, 0.008, 0.019],
        'LPIPS': [0.546, 0.574, 0.563, 0.596]
    }
    
    lama_data = {
        'SSIM': [0.998, 0.962, 0.968, 0.917],
        'MSE': [1.615, 78.481, 80.259, 243.890],
        'PSNR': [47.912, 34.725, 34.898, 28.134],
        'FID': [1.49, 3.131, 2.988, 6.009],
        'KID': [0.001, 0.001, 0.001, 0.002],
        'LPIPS': [0.558, 0.559, 0.559, 0.566]
    }
    
    # Create subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PDE vs LaMa on different masking shapes ', fontsize=14, y=1.02)
    
    x = np.arange(len(shapes))
    width = 0.35
    
    # Flatten axs for easier iteration
    axs_flat = axs.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axs_flat[idx]
        
        rects1 = ax.bar(x - width/2, pde_data[metric], width, label='PDE')
        rects2 = ax.bar(x + width/2, lama_data[metric], width, label='LaMa')
        
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(shapes, rotation=45)
        ax.legend()
        
        if metric in ['MSE', 'FID', 'KID', 'LPIPS']:
            ax.set_ylabel('Lower is better')
        else:
            ax.set_ylabel('Higher is better')
            
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)
                
        autolabel(rects1)
        autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('pde_vs_LaMa_different_masking_shapes.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_all_metrics_plot()


import matplotlib.pyplot as plt
import numpy as np

def create_noisy_metrics_plot():
    shapes = ['Line', 'Triangle', 'Circle', 'Rectangle']
    metrics = ['SSIM', 'MSE', 'PSNR', 'FID', 'KID', 'LPIPS']
    
    pde_noisy = {
        'SSIM': [0.994, 0.932, 0.941, 0.872],
        'MSE': [8.116, 182.310, 188.590, 477.696],
        'PSNR': [39.676, 29.066, 29.043, 23.663],
        'FID': [39.67, 71.031, 59.232, 108.285],
        'KID': [0.012, 0.022, 0.018, 0.043],
        'LPIPS': [0.720, 0.757, 0.750, 0.787]
    }
    
    lama_noisy = {
        'SSIM': [0.991, 0.928, 0.938, 0.862],
        'MSE': [11.444, 149.311, 139.465, 374.033],
        'PSNR': [38.136, 29.389, 29.393, 24.719],
        'FID': [2.795, 6.414, 5.148, 12.726],
        'KID': [0.001, 0.003, 0.002, 0.007],
        'LPIPS': [0.747, 0.760, 0.751, 0.769]
    }
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PDE vs LaMa Performance Metrics (With Noise)', fontsize=14, y=1.02)
    
    x = np.arange(len(shapes))
    width = 0.35
    
    for idx, metric in enumerate(metrics):
        ax = axs.flatten()[idx]
        
        rects1 = ax.bar(x - width/2, pde_noisy[metric], width, label='PDE', color='#ff9999')
        rects2 = ax.bar(x + width/2, lama_noisy[metric], width, label='LaMa', color='#66b3ff')
        
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(shapes, rotation=45)
        ax.legend()
        
        if metric in ['MSE', 'FID', 'KID', 'LPIPS']:
            ax.set_ylabel('Lower is better')
        else:
            ax.set_ylabel('Higher is better')
            
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)
                
        autolabel(rects1)
        autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('pde_vs_lama_noisy_metrics_comparison.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_noisy_metrics_plot()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_bias_plots():
    shapes = ['Line', 'Triangle', 'Circle', 'Rectangle', 'Overall']
    metrics = ['SSIM', 'MSE', 'PSNR']
    attributes = ['Male/Female', 'Young/Old', 'PaleSkin/DarkSkin']
    
    # Organize data into arrays for heatmap
    pde_data = {
        'SSIM': [
            [-0.005, 0.004, -0.001],  # Line
            [-0.006, 0.004, -0.009],  # Triangle
            [-0.004, 0.007, 0.003],   # Circle
            [-0.004, 0.010, 0.011],   # Rectangle
            [-0.005, 0.006, 0.001]    # Overall
        ],
        'MSE': [
            [0.430, -2.151, -2.360],
            [0.101, -0.126, 0.202],
            [0.054, -0.263, -0.139],
            [0.082, -0.075, -0.126],
            [0.091, -0.161, -0.058]
        ],
        'PSNR': [
            [0.006, 0.006, 0.014],
            [-0.010, 0.007, -0.040],
            [0.015, 0.003, 0.025],
            [-0.019, 0.020, -0.028],
            [0.000, 0.008, -0.004]
        ]
    }
    
    lama_data = {
        'SSIM': [
            [0.000, 0.000, 0.000],
            [0.000, 0.003, 0.010],
            [0.004, -0.005, -0.002],
            [0.003, -0.003, 0.001],
            [0.002, -0.001, 0.002]
        ],
        'MSE': [
            [-0.068, 0.060, 0.107],
            [0.029, -0.070, -0.269],
            [-0.003, 0.149, 0.289],
            [0.029, 0.145, 0.191],
            [0.022, 0.105, 0.153]
        ],
        'PSNR': [
            [0.007, -0.005, -0.003],
            [0.001, 0.008, 0.038],
            [0.025, -0.036, -0.036],
            [-0.007, -0.008, 0.010],
            [0.007, -0.010, 0.002]
        ]
    }
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Bias Analysis: PDE vs LaMa', fontsize=14, y=1.02)
    
    for idx, metric in enumerate(metrics):
        # PDE plot
        ax = axs[0][idx]
        sns.heatmap(pde_data[metric], 
                   xticklabels=attributes,
                   yticklabels=shapes,
                   cmap='RdBu',
                   center=0,
                   annot=True,
                   fmt='.3f',
                   ax=ax)
        ax.set_title(f'PDE {metric}')
        
        # LaMa plot
        ax = axs[1][idx]
        sns.heatmap(lama_data[metric],
                   xticklabels=attributes,
                   yticklabels=shapes,
                   cmap='RdBu',
                   center=0,
                   annot=True,
                   fmt='.3f',
                   ax=ax)
        ax.set_title(f'LaMa {metric}')
        
    plt.tight_layout()
    plt.savefig('bias_analysis.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_bias_plots()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_noisy_bias_plots():
    shapes = ['Line', 'Triangle', 'Circle', 'Rectangle', 'Overall']
    attributes = ['Male/Female', 'Young/Old', 'PaleSkin/DarkSkin']
    
    pde_noisy = {
        'SSIM': [
            [0.000, 0.001, 0.002],
            [0.006, 0.016, 0.024],
            [-0.009, 0.005, 0.011],
            [-0.015, 0.011, -0.004],
            [-0.004, 0.008, 0.008]
        ],
        'MSE': [
            [0.069, -0.267, -0.411],
            [-0.036, -0.107, -1.063],
            [0.098, 0.188, 0.467],
            [0.187, -0.200, -0.037],
            [0.125, -0.009, 0.050]
        ],
        'PSNR': [
            [0.002, 0.012, 0.026],
            [0.029, 0.015, 0.043],
            [-0.026, -0.032, -0.100],
            [-0.027, 0.001, -0.070],
            [-0.004, 0.000, -0.015]
        ]
    }
    
    lama_noisy = {
        'SSIM': [
            [0.000, 0.000, 0.000],
            [-0.001, 0.005, 0.017],
            [0.004, -0.005, -0.002],
            [-0.003, -0.002, 0.008],
            [0.000, -0.001, 0.006]
        ],
        'MSE': [
            [-0.040, 0.013, -0.094],
            [-0.009, -0.057, -0.250],
            [-0.048, 0.151, 0.193],
            [0.015, 0.114, 0.172],
            [-0.004, 0.084, 0.105]
        ],
        'PSNR': [
            [0.006, -0.001, 0.011],
            [0.002, 0.007, 0.044],
            [0.016, -0.022, -0.016],
            [-0.006, -0.006, 0.005],
            [0.005, -0.005, 0.012]
        ]
    }
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Bias Analysis with Noisy Images: PDE vs LaMa', fontsize=14, y=1.02)
    
    metrics = ['SSIM', 'MSE', 'PSNR']
    for idx, metric in enumerate(metrics):
        # PDE plot
        sns.heatmap(pde_noisy[metric], 
                   xticklabels=attributes,
                   yticklabels=shapes,
                   cmap='RdBu',
                   center=0,
                   annot=True,
                   fmt='.3f',
                   ax=axs[0][idx])
        axs[0][idx].set_title(f'PDE {metric} (Noisy)')
        
        # LaMa plot
        sns.heatmap(lama_noisy[metric],
                   xticklabels=attributes,
                   yticklabels=shapes,
                   cmap='RdBu',
                   center=0,
                   annot=True,
                   fmt='.3f',
                   ax=axs[1][idx])
        axs[1][idx].set_title(f'LaMa {metric} (Noisy)')
    
    plt.tight_layout()
    plt.savefig('pde_vs_lama_noisy_bias_analysis.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    create_noisy_bias_plots()