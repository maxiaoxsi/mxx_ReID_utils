import time
import datetime
import matplotlib.pyplot as plt
import re
import os

class Logger:
    def __init__(self, path_log) -> None:
        self._path_log = path_log

    def __call__(self, str_log):
        with open(self._path_log, 'a') as f:
            time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{time_str}] {str_log}\n")

    def warning(self, str_log):
        with open(self._path_log, 'a') as f:
            self(f"warning: {str_log}")

    def paint_loss(self):
        steps = []
        losses = []
        with open(self._path_log, 'r') as f:
            for line in f:
                step_match = re.search(r'step:(\d+)', line)
                loss_match = re.search(r'train_loss:([\d.]+)', line)
                step = int(step_match.group(1))
                loss = float(loss_match.group(1))
                if loss > 0.18 or loss < 0.04:
                    continue
                if step_match and loss_match:
                    steps.append(step)
                    losses.append(loss)
        loss = 0
        for i in range(10, len(steps) - 10):
            for j in (-10,  10):
                loss = loss + losses[i + j]
            loss = loss / 20
            losses[i] = loss
        for i in range(10):
            losses[i] = 0
        for i in range(len(steps) - 10, len(steps)):
            losses[i] = 0

        plt.figure(figsize=(30, 5))
        plt.plot(steps, losses, 'b-', label='Training Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss over Steps')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        dir_log = os.path.dirname(self._path_log)
        plt.savefig(os.path.join(dir_log, 'loss.png'), dpi=300, bbox_inches='tight')
        # 显示图表  
        plt.close()

