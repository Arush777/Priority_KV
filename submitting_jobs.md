Skip to main content
Cognitive Compute Cluster
 (CCC)
Search


Submitting Jobs
Contents:
Submit job to specific node
Submit job with gpu
Submit parallel jobs
Submit job with resource requirement
Submit job with time limit
Submit job with priority
Submit job as other user
Submit interactive job
GPU Job Submission
There are three different GPU types in the CCC compute nodes in the cluster. These are:
Familiar GPU Name	Formal Name
V100	TeslaV100_SXM2_32GB
A100-40GB	NVIDIAA100_SXM4_40GB
A100-80GB	NVIDIAA100_SXM4_80GB
H100	NVIDIAH10080GBHBM3

When you submit a job with bsub, you probably use a string like
-gpu num=8:mode=exclusive_process
to request GPUs. In this particular example, 8 GPUs are being requested.

If you want to request a particular type of GPUs, you must specify. one of the “formal” names above. So, if you want A100-80GB GPUs, your GPU specification string would look like:
-gpu num=8:mode=exclusive_process:gmodel=NVIDIAA100_SXM4_80GB

You can get a list of compute nodes and their GPU types with command:
lsload -w -gpuload
Job Termination Policy
Any Job that requested GPU and stayed idle without consuming the requested GPU will be terminated in 2 hours.
Interactive queue has 6 hours limit, Hence any job running in interactive queue exceeding the limit will be terminated.
Any process running on login node exceeding the 500% cpu limit get terminated.
Any process running on login node in batch of small task may run upto 800% CPU limit before it get terminated.
Submit Job to a Specific Node.
To submit a job to a specific node, use the -m option to specify the node name.
$ bsub -m node01 -J job1 ./job_script.sh
# This submits job_script.sh to node01, with the job name job1.

$ bsub -m "node01!node02" -J job_exclude_node ./task.sh
# This excludes node02 from the job submission, sending the job to node01

$ bsub -m "node01!node02" -J job_exclude_node command
# This excludes node02 from the job submission, sending the job to node01



Show more
Submit Job with GPU Resource Requirements,
To request a GPU, use the -R option with the rusage flag to specify the number of GPUs needed.
$ bsub -gpu num=1 -R "rusage[ngpus=1]" -J gpu_job1 ./gpu_task.sh
# This requests 1 GPU for gpu_task.sh, with job name gpu_job1.

$ bsub -gpu num=1 -R "rusage[ngpus=1]" -J gpu_job1 command
# This requests 1 GPU for gpu_task.sh, with job name gpu_job1.

$ bsub -gpu num=4  -R "rusage[ngpus=4]" -J gpu_job1 ./gpu_task.sh
# This requests 4 GPUs for gpu_task.sh

$ bsub -gpu num=2 -R "rusage[ngpus=2, cpu=4]" -J gpu_job ./your_script.sh
# This requests 2 GPUs and 4 CPUs per task

$ bsub -gpu num=2 -R "rusage[ngpus=2, cpu=4]" -J gpu_job -env "VAR1=value1,VAR2=value2" ./your_script.sh
# This requests 2 GPUs and 4 CPUs per task

$ bsub -gpu num=2 -n 2 -R "rusage[ngpus=2, cpus=4]" -J parallel_gpu_job ./your_script.sh
# This request run a parallel job where each task requests 2 GPUs and 4 CPUs


Show less
Submit a Parallel Job
specify the number of processors using the -n option. Use the -R option to control how processors are distributed across nodes.
$ bsub -n 4 -J myjob command
# This request 4 processor for 4 task

$ bsub -n 8 -R "span[ptile=4]" -J parallel_job1 ./parallel_task.sh
# This requests 8 processors with 4 processors per node.

$ bsub -n 8 -R "span[ptile=4]" -J parallel_job1 command
# This requests 8 processors with 4 processors per node.


Resource Requirements (Memory, CPU)
You can request specific resources like memory or CPU using the -R option with the rusage flag.
$ bsub -R "rusage[mem=8GB]" -J mem_job1 ./memory_task.sh
# This requests 8GB of memory for memory_task.sh.

$ bsub -R "rusage[mem=16GB]" -J mem_job2 ./memory_task.sh
# This requests 16GB of memory.

$ bsub -R "rusage[mem=32GB]" -J mem_job3 ./memory_task.sh
# This requests 32GB of memory.

$ bsub -R "rusage[cpu=4]" -J cpu_job1 ./cpu_task.sh
# This requests 4 CPUs for cpu_task.sh.

$ bsub -R "rusage[cpu=8]" -J cpu_job2 ./cpu_task.sh
# This requests 8 CPUs.

$ bsub -R "rusage[cpu=16]" -J cpu_job3 ./cpu_task.sh
# This requests 16 CPUs.

$ bsub -R "rusage[mem=8GB, cpu=4]" -J combined_job1 ./combined_task.sh
# This requests 8GB memory and 4 CPUs.

$ bsub -R "rusage[mem=16GB, cpu=8]" -J combined_job2 ./combined_task.sh
# This requests 16GB memory and 8 CPUs.


Show less
Submit Job with Specific Time Limit
Use the -W option to set the maximum runtime.
$ bsub -W 1:00 -J timed_job1 ./timed_task.sh
# This sets a 1-hour limit for timed_task.sh.

$ bsub -W 2:00 -J timed_job2 ./timed_task.sh
# This sets a 2-hour limit.

$ bsub -W 3:00 -J timed_job3 ./timed_task.sh
# This sets a 3-hour limit.

$ bsub -W 4:00 -J timed_job4 ./timed_task.sh
# This sets a 4-hour limit.

$ bsub -W 5:00 -J timed_job5 ./timed_task.sh
# This sets a 5-hour limit.


Show less
Submit Job with Priority
To set the priority of a job, use the -q option to specify the queue and adjust job priority.
$ bsub -q high_priority -P 10 -J high_prio_job1 ./high_priority_task.sh
# This submits to the high_priority queue with priority 10.

$ bsub -q medium_priority -P 5 -J medium_prio_job2 ./medium_priority_task.sh
# This submits to the medium_priority queue with priority 5.

$ bsub -q low_priority -P 1 -J low_prio_job3 ./low_priority_task.sh
# This submits to the low_priority queue with priority 1.

$ bsub -q urgent_queue -P 20 -J urgent_job4 ./urgent_task.sh
# This submits to an urgent_queue with priority 20.

$ bsub -q default -P 15 -J default_prio_job5 ./default_task.sh
# This submits to the default queue with priority 15


Show less
Submit Job as a Specific User
To submit a job as a different user, use the -u option with the username(linux user).
$ bsub -u username -J user_job1 ./user_task.sh
# This submits as alice.

Submit Interactive Job
Interactive jobs allow you to run jobs in an interactive session. Use the -I option to submit an interactive job.
$ bsub -I -J interactive_job1 ./interactive_task.sh
# This runs interactive_task.sh as an interactive job.

$ bsub -I -R "rusage[cpu=4, mem=8GB]" -J interactive_job ./interactive_task.sh
# This requests requests 4CPUs and 8GB of memory for the job

$ bsub -I -gpu num=2 -R "rusage[ngpus=2, cpu=4, mem=8GB]" -J interactive_job ./interactive_task.sh
# This requests requests 2GPUs, 4CPUs and 8GB of memory for the job

$ bsub -I -gpu num=2 -R "rusage[ngpus=2, cpu=4, mem=8GB]" -J interactive_job -env "VAR=myvalue" ./interactive_task.sh
# This requests requests 2GPUs, 4CPUs and 8GB of memory for the job setting the environment variable for job

$ bsub -I -gpu num=2 -n 2 -R "rusage[ngpus=2, cpu=4, mem=8GB]" -J interactive_job ./interactive_task.sh
# This requests requests 2GPUs, 4CPUs and 8GB of memory for each job for 2 tasks


Show less
Edit this page on GitHub
Previous
Job Scheduling, Managing and Monitoring: Working with Jobs
Next
Job Scheduling, Managing and Monitoring: Managaing Jobs
Privacy
Terms of Use
Support via AskETE
Reporting Feedback
Have questions? Open an issue in GitHub.

cognitivecompcluster.

Last updated 03 February 2026
Copyright © 2026 IBM
Navigated to Submitting Jobs
