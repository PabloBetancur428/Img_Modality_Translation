# benchmark_workers.py
import time
from config import settings
from dataset import get_dataloader

def benchmark(nw):
    # pass nw into get_dataloader
    loader = get_dataloader(
        settings.data_root + "/clean_baseline",
        settings.data_root + "/clean_follow_up",
        settings.data_root + "/patients_with_qsm.xlsx",
        num_workers=nw
    )
    # confirm it took effect
    print(f"→ loader.num_workers = {loader.num_workers}")

    # burn a few batches to fill the pipeline
    t0 = time.time()
    for i, _ in enumerate(loader):
        if i >= 50:
            break
    t1 = time.time()

    print(f"num_workers={nw:<2} → {(t1 - t0)/50:.3f}s per batch")

if __name__ == "__main__":
    for nw in [6, 12, 24, 32, 48]:
        benchmark(nw)
