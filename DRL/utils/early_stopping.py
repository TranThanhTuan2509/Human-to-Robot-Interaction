from pytorch_lightning.callbacks import Callback

class EarlyStopping(Callback):
    def __init__(self, monitor='average reward', patience=1000, mode='max', min_delta=0.3, verbose=False):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_score = None

        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor_op = min if mode == 'min' else max

    def on_validation_end(self, episode, current):
        if current is None:
            if self.verbose:
                print(f"EarlyStopping: Metric `{self.monitor}` is not available.")
            return

        # Convert to float
        current = float(current)

        # Initialize best score
        if self.best_score is None:
            self.best_score = current
            if self.verbose:
                print(f"EarlyStopping: Initializing best score to {self.best_score:.4f}")
            return

        # Check improvement
        if (self.mode == 'min' and current < self.best_score - self.min_delta) or \
           (self.mode == 'max' and current > self.best_score + self.min_delta):
            #  Be able to improve
            self.best_score = current
            self.wait_count = 0
            if self.verbose:
                print(f"EarlyStopping: Improvement detected. New best {self.monitor} = {self.best_score:.4f}")
            return False
        else:
            self.wait_count += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Wait count = {self.wait_count}/{self.patience}")
            if self.wait_count >= self.patience:
                self.stopped_epoch = episode
                if self.verbose:
                    print(f"EarlyStopping: Stopping at epoch {self.stopped_epoch} due to no improvement in `{self.monitor}`.")
                return True
