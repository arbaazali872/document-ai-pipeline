import os
import time
import signal
from transformers import TrainerCallback

class TrainingTimer:
    """Auto-kill training after max hours"""
    def __init__(self, max_hours):
        self.max_hours = max_hours
        self.start_time = time.time()
    
    def check(self):
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours > self.max_hours:
            print(f"\n⏰ MAX TIME REACHED ({self.max_hours}h)")
            print("🛑 Auto-stopping to save costs...")
            os.kill(os.getpid(), signal.SIGTERM)
        return elapsed_hours

class CostMonitorCallback(TrainerCallback):
    """Monitor training time and cost"""
    def __init__(self, timer, cost_per_hour=0.34):
        self.timer = timer
        self.start_time = None
        self.cost_per_hour = cost_per_hour
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n💰 Cost monitoring started (${self.cost_per_hour}/hour)")
        print(f"⏰ Auto-kill timer: {self.timer.max_hours}h")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed = self.timer.check()
        
        if self.start_time and state.global_step % 50 == 0:
            cost = elapsed * self.cost_per_hour
            remaining = self.timer.max_hours - elapsed
            print(f"⏱️  Time: {elapsed:.2f}h | 💵 Cost: ${cost:.3f} | ⏰ Remaining: {remaining:.2f}h")