from stable_baselines3.common.callbacks import BaseCallback

class PrintActionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # 从环境中获取动作
        action = self.locals['actions']  # 在stable-baselines3中，actions 存储在 self.locals 字典
        print(f"Action taken: {action}")
        return True  # 返回 True 继续训练
