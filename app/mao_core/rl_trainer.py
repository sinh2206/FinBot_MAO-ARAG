"""
RL Trainer: Tính toán reward dựa trên chất lượng và chi phí.
Lưu log để cải thiện planner.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Huấn luyện Planner thông qua reward = F1 - alpha * cost.
    Lưu log các episode.
    """

    def __init__(self, alpha: float = 0.01, log_dir: str = "storage/evaluation_logs"):
        """
        Args:
            alpha: Hệ số cân bằng giữa chất lượng và chi phí.
            log_dir: Thư mục lưu log.
        """
        self.alpha = alpha
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def compute_reward(self, f1_score: float, cost: float) -> float:
        """
        Tính reward = F1 - alpha * cost.

        Args:
            f1_score: Điểm F1 (0-1).
            cost: Chi phí USD.

        Returns:
            Reward (có thể âm).
        """
        return f1_score - self.alpha * cost

    def log_episode(self, query: str, plan: list, context: Dict[str, Any], 
                    f1_score: float, cost: float, reward: float):
        """
        Ghi lại một episode (một câu hỏi) vào file JSON.
        """
        episode = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "plan": plan,
            "context": {k: str(v)[:200] for k, v in context.items()},  # trim để tránh quá lớn
            "f1_score": f1_score,
            "cost": cost,
            "reward": reward,
            "alpha": self.alpha
        }
        # Lưu vào file theo ngày
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"rl_episodes_{date_str}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
        logger.info(f"Logged episode: reward={reward:.4f}, cost={cost:.6f}, f1={f1_score:.3f}")

    # Hàm này có thể được gọi sau mỗi câu trả lời để cập nhật planner
    async def update_planner(self, planner, episodes):
        """
        (Giả lập) Cập nhật planner dựa trên các episode.
        Trong thực tế, sẽ dùng RL thuật toán PPO để fine-tune prompt hoặc model.
        """
        # Ở đây chỉ là placeholder
        logger.info("Updating planner with RL (simulated)...")
        # Có thể thay đổi system prompt của planner dựa trên reward
        pass