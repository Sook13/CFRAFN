import swanlab
import random
from swanlab.plugin.notification import WXWorkCallback
# 调试模式
swanlab.init(mode='disabled')

wxwork_callback = WXWorkCallback(
    webhook_url="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=15a5c31b-8d22-433c-9e11-f815793ac58b",
)

# 创建一个SwanLab项目
swanlab.init(
    # 设置项目名
    project="my-awesome-project",
    
    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    },

    callbacks=[wxwork_callback]
)

# 模拟一次训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # 记录训练指标
  swanlab.log({"acc": acc, "loss": loss})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()