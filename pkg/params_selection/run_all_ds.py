import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..experiments.paper_format import ds_rename

def run_job(model, data, device):
    cmd = ["python", "-m", f"pkg.params_selection.{model}", "-data", data]
    if device is not None:
        cmd.append("-device")
        cmd.append(device)
    return subprocess.run(cmd, capture_output=False, text=True)

if __name__=='__main__':
    datasets = list(ds_rename.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-device', type=str, required=False, default=None)
    parser.add_argument('-n_workers', type=int, required=False, default=1)
    args = parser.parse_args()
    model = args.model
    n_workers = args.n_workers
    device = args.device
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_job, model, data, device): data for data in datasets}
        for future in as_completed(futures):
            print('DONE:', futures[future])
