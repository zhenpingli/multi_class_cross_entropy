from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer
from math import log

y_true = ['1', '4', '5'] # 样本的真实标签
y_pred = [[0.1, 0.6, 0.3, 0, 0, 0, 0, 0, 0, 0],
          [0, 0.3, 0.2, 0, 0.5, 0, 0, 0, 0, 0],
          [0.6, 0.3, 0, 0, 0, 0.1, 0, 0, 0, 0]
         ]               # 样本的预测概率
labels = ['0','1','2','3','4','5','6','7','8','9'] # 所有标签


# 利用sklearn中的log_loss()函数计算交叉熵
sk_log_loss = log_loss(y_true, y_pred, labels=labels)
print("Loss by sklearn is:%s." %sk_log_loss)

# 利用公式实现交叉熵
# 交叉熵的计算公式网址为：
# http://scikit-learn.org/stable/modules/model_evaluation.html#log-loss

# 对样本的真实标签进行标签二值化
lb = LabelBinarizer()
lb.fit(labels)
transformed_labels = lb.transform(y_true)
# print(transformed_labels)

N = len(y_true)  # 样本个数
K = len(labels)  # 标签个数

eps = 1e-15      # 预测概率的控制值
Loss = 0         # 损失值初始化

for i in range(N):
    for k in range(K):
        # 控制预测概率在[eps, 1-eps]内，避免求对数时出现问题
        if y_pred[i][k] < eps:
            y_pred[i][k] = eps
        if y_pred[i][k] > 1-eps:
            y_pred[i][k] = 1-eps
        # 多分类问题的交叉熵计算公式
        Loss -= transformed_labels[i][k]*log(y_pred[i][k])

Loss /= N
print("Loss by equation is:%s." % Loss)


##copy from
#https://segmentfault.com/a/1190000015787250