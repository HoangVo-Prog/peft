import os
from datetime import datetime
from typing import Optional




def setup_wandb(
    task: str,
    model_name: str,
    project: Optional[str],
    entity: Optional[str],
    run_name: Optional[str],
    offline_fallback: bool = True,
) -> str:
    """Login and init W&B. Uses Kaggle Secrets if available or env WANDB_API_KEY.
    Returns a run_name string.
    """
    import wandb


    run = run_name or f"{task}-{model_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"


    key = os.environ.get("WANDB_API_KEY")
    if key is None:
        try:
            from kaggle_secrets import UserSecretsClient # type: ignore
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            key = None


    if key:
        wandb.login(key=key)
    elif offline_fallback:
        os.environ["WANDB_MODE"] = "offline"


    wandb.init(
        project=project,
        entity=entity,
        name=run,
    )
    return run