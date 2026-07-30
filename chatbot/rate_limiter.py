"""
chatbot/rate_limiter.py — Rate limiter dùng chung cho mọi nơi gọi LLM.

Giữ tốc độ gọi API không vượt quá GEMINI_RATE_LIMIT_RPM request/phút.
Instance `gemini_rate_limiter` là module-level singleton — import cái này
ở mọi nơi gọi LLM thay vì tạo mới RateLimiter() riêng từng file.
"""

import time
import logging
import threading
from chatbot.config import GEMINI_RATE_LIMIT_RPM

logger = logging.getLogger(__name__)


class RateLimiter:
    """Giữ tốc độ gọi API không vượt quá `rpm` request/phút.
    Thread-safe, dùng chung 1 instance cho toàn bộ process.
    """

    def __init__(self, rpm: int = 12):
        # dư 3 so với hạn mức 15 RPM của Gemini free tier
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        """Block cho đến khi đủ khoảng cách tối thiểu giữa 2 lần gọi LLM."""
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                logger.debug("[rate_limiter] waiting %.3fs before next call", remaining)
                time.sleep(remaining)
            self._last_call = time.monotonic()


# ── Module-level singleton — import cái này ở mọi nơi gọi LLM ───────────────
gemini_rate_limiter = RateLimiter(rpm=GEMINI_RATE_LIMIT_RPM)
