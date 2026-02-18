import numpy as np

def run_manual_convolution():
    print("\n-------------------------------------------------")
    print(" MANUAL CNN LAYER IMPLEMENTATION (Step-by-Step)")
    print("-------------------------------------------------")

    # 1. INPUT: A 5x5 Matrix (Simulating a pixel grid)
    # 0 = Black, 1 = White/Grey
    input_image = np.array([
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0]
    ])

    # 2. KERNEL: A 3x3 Filter (Vertical Edge Detector)
    kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])

    # 3. Calculate Output Dimensions
    # Formula: (Input_H - Kernel_H) + 1
    input_h, input_w = input_image.shape
    kernel_h, kernel_w = kernel.shape
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    
    output_map = np.zeros((output_h, output_w))

    print(f"Input Shape: {input_image.shape}")
    print(f"Kernel Shape: {kernel.shape}")
    print(f"Calculated Output Shape: {output_map.shape}\n")

    # 4. THE CONVOLUTION LOOP
    for i in range(output_h):       # Slide down
        for j in range(output_w):   # Slide right
            # Extract the patch
            patch = input_image[i:i+kernel_h, j:j+kernel_w]
            
            # Element-wise multiplication sum (Dot Product)
            value = np.sum(patch * kernel)
            output_map[i, j] = value
            
            # VISUALIZATION for the very first step (Top-Left)
            if i == 0 and j == 0:
                print(f" Top-Left Calculation (Step 1):")
                print(f"   Patch:\n{patch}")
                print(f"   Kernel:\n{kernel}")
                print(f"   Calculation: Sum({patch.flatten()} * {kernel.flatten()})")
                print(f"   Result: {value}\n")

    print("-------------------------------------------------")
    print(" FINAL FEATURE MAP:")
    print(output_map)
    print("-------------------------------------------------")

if __name__ == "__main__":
    run_manual_convolution()