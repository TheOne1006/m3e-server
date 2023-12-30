from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
import numpy as np


# 插值法
def interpolate_vector(vector, target_length):
    original_indices = np.arange(len(vector))
    target_indices = np.linspace(0, len(vector)-1, target_length)
    f = interp1d(original_indices, vector, kind='linear')
    return f(target_indices)


# 多项式扩展
def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
        expanded_embedding = np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
    return expanded_embedding
