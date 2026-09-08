"""Normal initial endpoint for a frozen direct-control segment; no audit/proposals."""
import os
from common import EXP_ROOT, atomic_json, read_json
from verge_round_core import config, ROOT, VERSION
from verge_control_block import prepared_round, validate_prepared
from verge_round_runtime import rows, endpoint, VLLMServerPool, VLLMRolloutClient


def main():
    cfg = config()  # Independently validates the submitted block and source.
    path = ROOT / "round_frozen.json"
    if path.exists():
        validate_prepared(cfg, read_json(path))
        return
    port = 20000 + int(os.environ.get("SLURM_JOB_ID", "0")) % 25000
    with VLLMServerPool(gpus=(0,), base_port=port) as urls:
        client = VLLMRolloutClient(urls[0], 0)
        client.load_lora(VERSION+"_start", EXP_ROOT / cfg["solver_start"])
        initial = endpoint(client, ROOT / "initial", checkpoint=cfg["solver_start"],
            selection=rows(cfg["target_selection"])[:cfg["endpoint_instances"]],
            scope=rows(cfg["scope_dataset"])[:cfg["scope_instances"]])
        atomic_json(path, prepared_round(cfg, initial))
        client.use_base()


if __name__ == "__main__":
    main()
