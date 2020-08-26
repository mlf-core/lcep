import click
import tempfile
import mlflow
import subprocess
import os
import numpy as np
import random


def set_general_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python general
    np.random.seed(seed)  # Numpy random
    random.seed(seed)  # Python random


def log_sys_intel_conda_env(framework: str):
    reports_output_dir = tempfile.mkdtemp()
    log_system_intelligence(reports_output_dir)
    log_conda_environment(reports_output_dir, framework)


def log_system_intelligence(reports_output_dir: str):
    # Scoped import to prevent issues like RuntimeError: Numba cannot operate on non-primary CUDA context
    from system_intelligence.query import query_and_export

    click.echo(click.style(f'Writing reports locally to {reports_output_dir}\n', fg='blue'))
    click.echo(click.style('Running system-intelligence', fg='blue'))
    query_and_export(query_scope=list(('all',)),
                     verbose=False,
                     export_format='json',
                     generate_html_table=True,
                     output=f'{reports_output_dir}/system_intelligence.json')
    click.echo(click.style('Uploading system-intelligence report as a run artifact...', fg='blue'))
    mlflow.log_artifacts(reports_output_dir, artifact_path='reports')


def log_conda_environment(reports_output_dir: str, framework: str):
    click.echo(click.style('Exporting conda environment...', fg='blue'))
    conda_env_filehandler = open(f'{reports_output_dir}/{framework}_env.yml', "w")
    subprocess.call(['conda', 'env', 'export', '--name', f'{framework}'], stdout=conda_env_filehandler)
    click.echo(click.style('Uploading conda environment report as a run artifact...', fg='blue'))
    mlflow.log_artifact(f'{reports_output_dir}/{framework}_env.yml', artifact_path='reports')
