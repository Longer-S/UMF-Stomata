# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import cv2
import numpy as np
from PIL import Image
from collections import OrderedDict
from net import AIFNet
from args_fusion import args
import time  
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")

def load_model(path, in_c, out_c):
    model = AIFNet(in_c, out_c)
    sd = torch.load(path, map_location=device)
    if list(sd.keys())[0].startswith('module.'):
        new_sd = OrderedDict([(k[7:], v) for k, v in sd.items()])
        model.load_state_dict(new_sd)
    else:
        model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

def draw_colorbar(height, n_stack):
    bar_width, text_area_width = 60, 80
    gradient = np.linspace(0, 255, height).astype(np.uint8).reshape(height, 1)
    gradient = np.repeat(gradient, bar_width, axis=1)
    gradient = np.flipud(gradient) 
    colorbar_img = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

    canvas = np.ones((height, bar_width + text_area_width, 3), dtype=np.uint8) * 255
    canvas[:, :bar_width, :] = colorbar_img

    for i in range(n_stack):
        y_pos = int((height - 1) - (i / (n_stack - 1 if n_stack > 1 else 1)) * (height - 1))
        cv2.line(canvas, (bar_width, y_pos), (bar_width + 10, y_pos), (0, 0, 0), 1)
        text = str(i)
        font_scale, thickness = 0.6, 1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_y = y_pos + text_size[1] // 2
        cv2.putText(canvas, text, (bar_width + 15, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    return canvas

def run_visualize_decision(model, focus_stack, output_root, folder_name, use_tiling=True, tile_overlap=32):
    """
    Args:
        use_tiling (bool): 是否切片
        tile_overlap (int): 切片之间的重叠像素数 (用于计算步长)
    """
    n, h, w, c = focus_stack.shape
    
    full_img = None 
    full_mask = None

    # ================= 模式 1: 切片推理 (Tiling) =================
    if use_tiling:
        ps = args.PATCH_SIZE
        
        # --- 动态计算步长 (不再写死 4) ---
        stride = ps - tile_overlap
        if stride <= 0:
            raise ValueError(f"重叠量 ({tile_overlap}) 必须小于切片大小 ({ps})，否则步长为负！")

        print(f"模式: 切片推理 | Patch: {ps} | Overlap: {tile_overlap} | Stride: {stride}")
        
        # 1. 准备 patches
        rgb_stack = np.transpose(focus_stack, (0, 3, 1, 2)) # (N, C, H, W)
        patches, coords = [], []
        
        # --- 使用动态 stride 生成坐标 ---
        # 核心区域
        for i in range(0, h - ps + 1, stride):
            for j in range(0, w - ps + 1, stride): 
                coords.append((i, j))
        
        # 补齐边缘 (右边和下边)
        for i in range(0, h - ps + 1, stride): coords.append((i, w - ps))
        for j in range(0, w - ps + 1, stride): coords.append((h - ps, j))
        coords.append((h - ps, w - ps)) # 右下角
        
        # 去重 (防止某些尺寸下边缘块被添加多次)
        coords = sorted(list(set(coords)))

        for (y, x) in coords:
            patches.append(rgb_stack[:, :, y:y+ps, x:x+ps])
        
        patches_tensor = torch.from_numpy(np.stack(patches)).float().to(device)
        
        # 2. 推理
        all_outs, all_masks = [], []
        batch_size = 32 
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size): 
                b = patches_tensor[i : i + batch_size]
                o, c,m = model(b)
                all_outs.append(o.cpu())
                all_masks.append(m.cpu())
                
        outs = torch.cat(all_outs).numpy()
        masks = torch.cat(all_masks).numpy()

        # 3. 重建大图
        full_img = np.zeros((3, h, w))
        full_mask = np.zeros((n, 1, h, w))
        cnt = np.zeros((h, w))
        
        for idx, (y, x) in enumerate(coords):
            full_img[:, y:y+ps, x:x+ps] += outs[idx]
            full_mask[:, :, y:y+ps, x:x+ps] += masks[idx]
            cnt[y:y+ps, x:x+ps] += 1
            
        full_img /= cnt
        full_mask /= cnt

    # ================= 模式 2: 全图推理 =================
    else:
        print("模式: 全图推理 (Full Image)")
        input_data = np.transpose(focus_stack, (0, 3, 1, 2)) 
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            o, m, _ = model(input_tensor)
            
        full_img = o.squeeze(0).cpu().numpy()
        full_mask = m.squeeze(0).cpu().numpy()

    # ================= 4. 可视化保存 =================
    decision_map = np.argmax(full_mask, axis=0).reshape(h, w)
    decision_vis = ((decision_map / (n - 1 if n > 1 else 1)) * 255).astype(np.uint8)
    decision_color = cv2.applyColorMap(decision_vis, cv2.COLORMAP_JET)
    colorbar = draw_colorbar(h, n)
    decision_with_bar = np.hstack([decision_color, colorbar])

    if not os.path.exists(output_root): os.makedirs(output_root)
    res_img_transposed = np.transpose(full_img, (1, 2, 0))
    res_img_uint8 = np.clip(res_img_transposed * 255, 0, 255).astype(np.uint8)
    res_bgr = cv2.cvtColor(res_img_uint8, cv2.COLOR_RGB2BGR)
    # === 修改点: 保存为 JPG，并设置质量为 100 (最高) ===
    save_path = os.path.join(output_root, f"{folder_name}.jpg")
    cv2.imwrite(save_path, res_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # cv2.imwrite(os.path.join(output_root, f"{folder_name}_fused.png"), res_bgr)
    
    # cv2.imwrite(os.path.join(output_root, f"{folder_name}_decision.png"), decision_with_bar)
    print(f"已完成: {folder_name}") 

def main():
    test_root = "data/"         
    output_root = './stacks_maize_output/' 
    model_path = args.model_path 
    n_stack = args.n_stack      
    
    # =============== 配置参数 ===============
    # 1. 是否切片 (False=全图, True=切片)
    USE_SLIDING_WINDOW = True 
    
    # 2. 切片重叠量 (单位: 像素)
    TILE_OVERLAP = 16
    # =======================================
    # 设置输入顺序模式 ===
    # 可选项: 'normal' (顺序), 'reverse' (倒序), 'random' (随机)
    ORDER_MODE = 'normal'
    model = load_model(model_path, 3, 1)
    print(f"当前测试模式: [ {ORDER_MODE} ]")
    if ORDER_MODE == 'random':
        print("注意: 随机模式下，每次运行结果可能不同 (除非固定种子)。")

    subfolders = [f for f in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, f))]
    subfolders.sort()
    
    print(f"检测到 {len(subfolders)} 个任务文件夹...")

    # === 修改点 2: 初始化统计变量 ===
    total_time = 0
    processed_count = 0

    for folder in subfolders:
        folder_path = os.path.join(test_root, folder)
        focus_stack = []
        
        found_all = True
        for j in range(n_stack):
            img_found = False
            for ext in ['.jpg', '.png', '.JPG', '.jpeg', '.tif']:
                img_name = f"{folder}_{j:02d}{ext}"
                img_p = os.path.join(folder_path, img_name)
                if os.path.exists(img_p):
                    img = Image.open(img_p).convert('RGB')
                    focus_stack.append(np.array(img, dtype=np.float32)/255.0)
                    img_found = True
                    break
            if not img_found:
                print(f"错误: 找不到 {folder}_{j:02d}")
                found_all = False
                break
        
        if found_all and len(focus_stack) == n_stack:

            # 根据模式调整 focus_stack 的顺序 ===
            if ORDER_MODE == 'reverse':
                focus_stack.reverse() # 倒序
            elif ORDER_MODE == 'random':
                random.shuffle(focus_stack) # 随机打乱
            # elif ORDER_MODE == 'normal': pass # 保持原样
            # 单次任务计时开始 ===
            start_time = time.time()
            
            run_visualize_decision(
                model, 
                np.stack(focus_stack), 
                output_root, 
                folder, 
                use_tiling=USE_SLIDING_WINDOW,
                tile_overlap=TILE_OVERLAP 
            )
            
            # 计时结束并统计 ===
            end_time = time.time()
            stack_time = end_time - start_time
            total_time += stack_time
            processed_count += 1
            
            print(f"已完成: {folder} | 耗时: {stack_time:.4f} 秒")
        else:
            print(f"跳过: {folder}")

    # 输出平均时间 ===
    if processed_count > 0:
        avg_time = total_time / processed_count
        print("-" * 30)
        print(f"所有任务处理完毕。")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"平均每组耗时: {avg_time:.4f} 秒")
    else:
        print("未处理任何图像。")

if __name__ == '__main__':
    main()