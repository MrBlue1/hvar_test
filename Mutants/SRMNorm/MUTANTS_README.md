# RMSNorm 变异体清单

## 生成信息
- 源程序: M00.py (RMSNorm)
- 变异体数量: 110
- 生成时间: 2026-02-25 21:56:06

## 变异算子分类

### 1. 算术运算符变异 (AOR) - M01-M12
- M01-M03: 加法运算符变异 (+ → -, *, /)
- M04-M06: 乘法运算符变异 (* → +, -, /)
- M07-M09: weight乘法变异
- M10-M12: pow(2)变异

### 2. 关系运算符变异 (ROR) - M13-M20
- M13: keepdim参数变异
- M14-M15: mean维度变异
- M16: mean→sum
- M17-M18: rsqrt变异
- M19-M20: 初始化变异

### 3. 常量变异 (CR) - M21-M30
- M21-M25: epsilon值变异
- M26-M30: 维度大小变异

### 4. 语句删除 (SDL) - M31-M36
- M31: 删除keepdim
- M32: 删除epsilon
- M33: 删除weight乘法
- M34: 删除rsqrt
- M35: 删除pow(2)
- M36: 删除mean

### 5. 变量替换 (VVR) - M37-M40
- M37-M40: 变量名替换

### 6. 函数调用变异 (FCR) - M41-M50
- M41-M50: 函数替换

### 7. 交换变异 (SOR) - M61-M63
- M61-M63: 操作数交换

### 8. 一元运算符变异 (UOI) - M64-M67
- M64-M67: 添加/修改一元运算符

### 9. 数据类型变异 (DTR) - M68-M70
- M68-M70: 数据类型修改

### 10. 返回值变异 (RVR) - M73-M75
- M73-M75: return语句变异

### 11. 索引变异 (IOR) - M82-M84
- M82-M84: 维度索引变异

### 12. RMSNorm特定变异 (RMS) - M90-M99
- M90-M91: LayerNorm化变异
- M92-M94: rsqrt操作变异
- M95: 添加bias
- M96-M99: weight操作变异

### 13. 数值精度变异 (NPR) - M100-M102
- M100-M102: epsilon缩放和噪声

### 14. 内存访问变异 (MAR) - M103-M104
- M103-M104: 内存操作变异

### 15. 设备相关变异 (DVR) - M106-M107
- M106-M107: 设备指定变异

### 16. 广播变异 (BCR) - M108-M109
- M108-M109: 广播行为变异

### 17. 梯度变异 (GDR) - M110
- M110: no_grad装饰器

## 使用方法
```bash
# 运行原始程序
python M00.py

# 运行变异体
python M01.py
python M02.py
...
python M110.py
```

## 等价变异体提示
以下变异体可能是等价的（需验证）：
- M12 (pow(2) → **2): 功能等价
- M62 (交换律): 乘法交换律
- M63 (加法交换律): epsilon交换

## 参考文献
- Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. NeurIPS 2019.
- LLaMA官方实现: https://github.com/facebookresearch/llama
