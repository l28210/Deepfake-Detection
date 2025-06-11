import numpy as np
# 两个分布的余弦相似度
def cos_similarity(distribution_a:list, distribution_b:list):
    a = np.array(distribution_a)
    b = np.array(distribution_b)
    return np.dot(a,b)/(np.linalg.norm(a)*(np.linalg.norm(b)))

# 每个clients的个数 转换 分布比例
def count2proportion(cli_count:list) -> list:
    total_count = sum(cli_count)
    return [x/total_count for x in cli_count]