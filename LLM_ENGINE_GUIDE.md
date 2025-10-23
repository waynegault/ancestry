# LM Studio Engine Selection Guide

## Your Available Engines

### 1. **CUDA llama.cpp (Windows)** ✅ **RECOMMENDED**
- **Best for:** Your RTX 3050 Ti GPU
- **Performance:** Fastest (uses NVIDIA CUDA)
- **When to use:** Always (unless it doesn't work)
- **Pros:**
  - Maximum GPU acceleration
  - Fastest inference times
  - Best for 4GB VRAM models
- **Cons:**
  - Requires NVIDIA GPU (you have one ✅)

---

### 2. **CUDA 12 llama.cpp (Windows)**
- **Best for:** Newer CUDA 12.x features
- **Performance:** Similar to CUDA llama.cpp
- **When to use:** If regular CUDA has issues
- **Pros:**
  - Latest CUDA features
  - May be slightly faster on newer GPUs
- **Cons:**
  - Requires CUDA 12.x drivers
  - May not be compatible with older systems

---

### 3. **Vulkan llama.cpp (Windows)**
- **Best for:** Cross-platform GPU acceleration
- **Performance:** Good (but slower than CUDA)
- **When to use:** If CUDA doesn't work
- **Pros:**
  - Works on AMD, Intel, and NVIDIA GPUs
  - More compatible
- **Cons:**
  - Slower than CUDA on NVIDIA GPUs
  - Less optimized for your RTX 3050 Ti

---

### 4. **CPU llama.cpp (Windows)**
- **Best for:** Systems without GPU
- **Performance:** Slowest
- **When to use:** Only as last resort
- **Pros:**
  - Works on any system
  - No GPU required
- **Cons:**
  - Very slow (minutes per response)
  - Not recommended for your system

---

## Recommendation for Your Hardware

**Dell XPS 15 9520 (i9-12900HK, 64GB RAM, RTX 3050 Ti 4GB)**

### **Use: CUDA llama.cpp (Windows)** ✅

**Why:**
- Your RTX 3050 Ti has 4GB VRAM
- CUDA provides maximum GPU acceleration
- Should give 2-5 second responses for 4GB models
- Should give 5-10 second responses for 7-8GB models

### **GPU Offload Settings:**

| Model Size | GPU Offload | Expected Speed |
|------------|-------------|----------------|
| 2.5GB (Qwen3-4B) | Maximum (all layers) | 2-4 seconds |
| 4GB (Mistral-7B, DeepSeek-R1) | Maximum (all layers) | 3-6 seconds |
| 8GB (Qwen2.5-Coder-14B) | Partial (~50%) | 8-15 seconds |

**Note:** For 8GB models, you may need to offload only part of the model to GPU due to 4GB VRAM limit.

---

## Testing Procedure

### **Step 1: Set Engine**
1. In LM Studio, click the engine dropdown (top right)
2. Select **"CUDA llama.cpp (Windows)"**
3. Verify checkmark appears

### **Step 2: Test Each Model**
Run the comprehensive test script:
```bash
python test_all_llm_models.py
```

The script will:
1. Guide you through loading each model
2. Test speed and quality
3. Compare all models
4. Recommend the best one

### **Step 3: Compare Results**
After testing, you'll see:
- Average response times
- Quality scores
- Pass rates
- Overall recommendation

---

## Expected Results

### **Qwen3-4B (2.5GB)** - Smallest, Fastest
- **Speed:** 2-4 seconds ⚡
- **Quality:** Good
- **Best for:** Quick responses, low VRAM usage

### **Mistral-7B-v0.3 (4GB)** - Balanced
- **Speed:** 3-6 seconds ✅
- **Quality:** Very Good
- **Best for:** Balance of speed and quality

### **DeepSeek-R1-Distill-Qwen-7B (4GB)** - Reasoning
- **Speed:** 3-6 seconds ✅
- **Quality:** Excellent (reasoning-focused)
- **Best for:** Complex genealogical analysis

### **Qwen2.5-Coder-14B (8GB)** - Largest, Slowest
- **Speed:** 8-15 seconds ⚠️
- **Quality:** Excellent
- **Best for:** Maximum quality (if speed is acceptable)

---

## Troubleshooting

### **If CUDA doesn't work:**
1. Try **CUDA 12 llama.cpp**
2. If that fails, try **Vulkan llama.cpp**
3. Update NVIDIA drivers
4. Restart LM Studio

### **If responses are slow:**
1. Check GPU Offload slider (should be maximum)
2. Verify CUDA engine is selected
3. Close other GPU-intensive apps
4. Try a smaller model

### **If model won't load:**
1. Model may be too large for 4GB VRAM
2. Try reducing GPU Offload
3. Try a smaller model
4. Check available system RAM

---

## Quick Start

**Recommended Setup for Best Results:**

1. **Engine:** CUDA llama.cpp (Windows)
2. **Model:** Mistral-7B-v0.3 or DeepSeek-R1-Distill-Qwen-7B
3. **GPU Offload:** Maximum
4. **Expected Speed:** 3-6 seconds
5. **Quality:** Excellent for genealogical work

**Run this to test:**
```bash
python test_all_llm_models.py
```

The script will help you find the perfect model for your needs! 🚀

