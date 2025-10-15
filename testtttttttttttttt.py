# 保存为 run_client_demo.py，然后用
#   uv run python run_client_demo.py
# 运行即可（确保 serve_policy.py 已经在 8000 端口启动）。

import time
import numpy as np

from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.policies import aloha_policy  # 换其他环境就 Import 对应的 make_*_example

def describe_actions(actions: np.ndarray) -> str:
    """把动作 chunk 的形状、均值等打印出来，方便观察。"""
    return (
        f"shape={actions.shape}, "
        f"mean={actions.mean():.4f}, "
        f"min={actions.min():.4f}, "
        f"max={actions.max():.4f}"
    )

def main():
    client = WebsocketClientPolicy(host="127.0.0.1", port=8000)
    print("连接成功，开始推理循环…")

    for step in range(500):  # 连续请求 5 次，真实使用时改成 while True 或按需调用
        obs = aloha_policy.make_aloha_example()  # 用真实机器人观测替换这里
        response = client.infer(obs)             # {"actions": (H, D), "policy_timing": ..., ... }

        actions = response["actions"]
        print(f"[step {step}] 动作统计: {describe_actions(actions)}")
        print(f"          前 1 帧动作: {actions[0]}")
        print(f"          policy_timing (ms): {response['policy_timing']['infer_ms']:.2f}")


        time.sleep(0.2)

    print("推理循环结束。")

if __name__ == "__main__":
    main()
