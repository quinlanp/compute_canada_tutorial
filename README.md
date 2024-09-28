# SETUP

For the compute canada systems most of the information you will need can be found here:

https://docs.alliancecan.ca/wiki/AI_and_Machine_Learning

There are several different file systems on each cluster:

| Filesystem     | Default Quota                      | Lustre-based | Backed up | Purged                               | Available by Default | Mounted on Compute Nodes |
|----------------|------------------------------------|--------------|-----------|--------------------------------------|----------------------|---------------------------|
| Home Space     | 50 GB and 500K files per user[1]   | Yes          | Yes       | No                                   | Yes                  | Yes                       |
| Scratch Space  | 20 TB and 1M files per user        | Yes          | No        | Files older than 60 days are purged.[2] | Yes                  | Yes                       |
| Project Space  | 1 TB and 500K files per group[3]   | Yes          | Yes       | No                                   | Yes                  | Yes                       |
| Nearline Space | 2 TB and 5000 files per group      | Yes          | Yes       | No                                   | Yes                  | No                        |

[1] Additional quota available upon request.
[2] Purge policies may vary based on system configuration.
[3] Group quotas may vary depending on project allocation.


In this setup it is generally advisible to use your Scratch space to store large datasets , models etc.. There is more than enough storage here for almost anything you may need.


When it comes to setup it is generally advisible to setup your environments on the actual compute node and not the login node. Compute Canada has preinstalled most of the packages you may need.


## Download Models

To start we will download huggingface models and datasets. To do that you need huggingface-cli installed

```bash
python -m venv local_env

source ./local_env/bin/activate

pip install --upgrade huggingface_hub[cli]

```

Hugging face requires login to download llama 3 models, make sure that you are loogged in before running the following command:
```bash
huggingface-cli login
huggingface-cli whoami
```

Now Download the Model

```bash
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct --local-dir /home/[username]/scratch/llama3_1_70b_instruct
```

Download the data
```bash

```


## Download Code

```bash
git clone https://github.com/meta-llama/llama-recipes.git

```


## Run interactive terminal

This gives about 1/4 of a node for testing if your code runs. I reccomend pasting in the batch script that you intend on using to ensure the environment is configured the same. Also note that you may need to change the bash script to match the resources you requested.

```bash
salloc --cpus-per-task=8 --gpus-per-task=1 --nodes=1 --ntasks-per-node=1 --time=1:00:0 --mem=128000 --account=rrg-zhu2048
```

## submit script

Note that this has the environemnt setup built in. Compute canada does not have every package required for this library so I went ahead and removed the ones we didnt have since they are not needed for this finetuning tutorial.
```bash
sbatch slurm_script.sh
```

Note that after training you will need to convert your FSDP shards to a hf checkpoint with this script

In order to run this command you will need to setup your environment as in the slurm_script ON A COMPUTE NODE

```bash
python convert_checkpoint.py \
  --fsdp_checkpoint_path /home/quinlanp/scratch/llama3-checkpoints/paper/TS/ \
  --consolidated_model_path /home/quinlanp/scratch/llama3-checkpoints/paper/TS/hf_checkpoint \
  --HF_model_path_or_name /home/quinlanp/scratch/llama3-checkpoints/paper/TS/ \
```
