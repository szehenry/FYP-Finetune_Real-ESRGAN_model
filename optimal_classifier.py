
def optimal_blur_detector(image):
    """基于数据分析的最优模糊检测器"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 提取关键特征
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    
    diff_h = np.abs(np.diff(gray, axis=1))
    diff_v = np.abs(np.diff(gray, axis=0))
    texture_score = np.mean(diff_h) + np.mean(diff_v)
    
    # 决策规则 (按重要性排序)
    score = 0
    
    # texture_score (权重: 2.55)
    if texture_score > 198.429:
        score += 2.55
    
    # laplacian_var (权重: 1.80)
    if laplacian_var > 866.382:
        score += 1.80
    
    # high_freq_ratio (权重: 1.03)
    if high_freq_ratio > 0.659:
        score += 1.03
    
    
    # 判断阈值 (基于权重总和)
    threshold_score = 3.23
    is_blurry = score >= threshold_score
    
    return is_blurry, laplacian_var
    