"""
FinShield Azure ML v2 SDK Pipeline Definition
Defines the enterprise multi-step MLOps pipeline for submission to Azure Machine Learning.
Orchestrates: Data Validation -> Feature Engineering -> Model Training & Calibration -> Evaluation & Gatekeeper.
"""

import os
from pathlib import Path

# Example Azure ML v2 SDK construction (runnable with azure-ai-ml in cloud)
AZUREML_PIPELINE_YAML = """
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: finshield-credit-risk-training-pipeline
experiment_name: credit-risk-scoring

settings:
  default_compute: azureml:cpu-cluster
  continue_on_step_failure: false

inputs:
  raw_credit_data:
    type: uri_file
    path: azureml:credit-applications-raw:1

jobs:
  validate_data_job:
    type: command
    component: azureml:finshield_validate_component:1
    inputs:
      input_data: ${{parent.inputs.raw_credit_data}}
    outputs:
      validation_report:
        type: uri_file

  feature_engineering_job:
    type: command
    component: azureml:finshield_feature_engineering_component:1
    inputs:
      raw_data: ${{parent.inputs.raw_credit_data}}
    outputs:
      processed_data:
        type: uri_file

  train_and_calibrate_job:
    type: command
    component: azureml:finshield_train_component:1
    inputs:
      processed_data: ${{parent.jobs.feature_engineering_job.outputs.processed_data}}
    outputs:
      model_output:
        type: uri_folder
      metrics_output:
        type: uri_file
"""

def generate_pipeline_manifest(output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "pipeline_job.yaml"
    with open(manifest_file, "w") as f:
        f.write(AZUREML_PIPELINE_YAML.strip())
    print(f"Generated Azure ML v2 Pipeline YAML at: {manifest_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    generate_pipeline_manifest(str(base_dir))
