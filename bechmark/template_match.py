import time

import cv2
import torch
import fastcv
import numpy as np

TEMPLATE_SIZES=[10,20,30]

def benchmark_template_match(sizes=[1024, 2048, 4096], runs=50):
    results = []
    
    for i,size in enumerate(sizes):
        print(f"\n=== Benchmarking {size}x{size} image ===")
        
        img_np = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        img_np = img_np.astype(np.float32)
        img_torch = torch.from_numpy(img_np).cuda()
        img_torch = img_torch.permute(2,0,1)

        template_np = np.random.randint(0, 256, (TEMPLATE_SIZES[i], TEMPLATE_SIZES[i], 4), dtype=np.uint8)
        template_np = template_np.astype(np.float32)
        template_torch = torch.from_numpy(template_np).cuda()
        template_torch = template_torch.permute(2,0,1)
        template_cv = cv2.cvtColor(template_np, cv2.COLOR_BGRA2BGR)


        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.matchTemplate(img_np, template_cv, cv2.TM_SQDIFF)
        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.template_match(img_torch, template_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_time = (end - start) / runs * 1000  # ms per run

        results.append((TEMPLATE_SIZES[i], size, cv_time, fc_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUDA): {fc_time:.4f} ms")
    
    return results

 
if __name__ == "__main__":
    results = benchmark_template_match()
    print("\n=== Final Results ===")
    print("Template Size\tImage Size\t\tOpenCV (CPU)\t\tfastcv (CUDA)")
    for template_size, size, cv_time, fc_time in results:
        print(f"{template_size}x{template_size}\t\t{size}x{size}\t\t{cv_time:.4f} ms\t\t{fc_time:.4f} ms")
