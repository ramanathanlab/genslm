import subprocess
from argparse import ArgumentParser
from pathlib import Path

import jinja2
from pydantic import BaseModel, validator


class HPCSettings(BaseModel):
    allocation: str
    queue: str
    time: str
    nodes: int
    job_name: str
    config: Path

    @validator("config")
    def config_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise FileNotFoundError(f"Config file does not exist: {v}")
        return v


def format_and_submit(template_name: str, settings: HPCSettings) -> None:
    """Add settings to a submit script, save to temp file, and submit to HPC scheduler"""

    env = jinja2.Environment(
        loader=jinja2.PackageLoader("gene_transformer.hpc"),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )

    try:
        template = env.get_template(template_name + ".j2")
    except jinja2.exceptions.TemplateNotFound:
        raise ValueError(f"template {template_name} not found.")

    submit_script = template.render(settings.dict())

    sbatch_script = "/tmp/gene_transformer_script.sbatch"
    with open(sbatch_script, "w") as f:
        f.write(submit_script)

    if template_name == "perlmutter":
        subprocess.run(f"sbatch {sbatch_script}".split())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-T", "--template", default="perlmutter")
    parser.add_argument("-a", "--allocation", default="m3957_g")
    parser.add_argument("-q", "--queue", default="regular")
    parser.add_argument("-t", "--time", default="4:00:00")
    parser.add_argument("-n", "--nodes", default=4, type=int)
    parser.add_argument("-j", "--job_name", default="")
    parser.add_argument("-c", "--config", required=True, type=Path)
    args = parser.parse_args()

    # Set default job name if not specified
    if not args.job_name:
        args.job_name = args.config.with_suffix("").name

    settings = HPCSettings(
        allocation=args.allocation,
        queue=args.queue,
        time=args.time,
        nodes=args.nodes,
        job_name=args.job_name,
        config=args.config,
    )
    format_and_submit(args.template, settings)
