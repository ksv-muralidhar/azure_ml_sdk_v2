import os
import logging
import tempfile
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
azure_identity_logger = logging.getLogger('azure.identity')
azure_identity_logger.setLevel(logging.ERROR)

subscription_id = os.getenv("SUBSCRIPTION_ID")
resource_group = os.getenv("RESOURCE_GROUP")
workspace_name = os.getenv("WORKSPACE_NAME")
workspace_location = os.getenv("WORKSPACE_LOCATION")
env_name = os.getenv('ENVIRONMENT_NAME')
requirements_file_path = "requirements.txt"
conda_python_version = os.getenv('CONDA_PYTHON_VERSION')

ml_client = MLClient(DefaultAzureCredential(exclude_shared_token_cache_credential=True), subscription_id, resource_group, 
                     workspace_name=workspace_name, workspace_location=workspace_location)

def create_environment(requirements_file_path, env_name):
    try:
        logging.info("Entering create_environment()")
        dependencies = []
        with open(requirements_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.find("==") != -1:
                        package, version = line.split('==')
                        dependencies.append(f'    - {package}=={version}')
                    else:
                        dependencies.append(f'    - {line}')

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, 'conda.yml'), 'w') as f:
                f.write("name: azureml_environment\n")
                f.write("channels:\n")
                f.write("  - conda-forge\n")
                f.write("  - defaults\n")
                f.write("dependencies:\n")
                f.write(f"  - python=={conda_python_version}\n")
                f.write(f"  - pip\n")
                f.write(f"  - pip:\n")
                f.write('\n'.join(dependencies))

            env_docker_conda = Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file=os.path.join(temp_dir, 'conda.yml'),
            name=env_name,
            description="Environment created from a Docker image plus Conda environment.",
                                        )
            ml_client.environments.create_or_update(env_docker_conda)

            logging.info("Succesfully created/updated environment")
            logging.info("Exiting create_environment()")
    except Exception as e:
        logging.critical(f"Error in create_environment(): {e}")
        raise

if __name__ == "__main__":
    create_environment(requirements_file_path=requirements_file_path, env_name=env_name)
