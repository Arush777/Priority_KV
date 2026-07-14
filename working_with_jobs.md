Skip to main content
Cognitive Compute Cluster
 (CCC)
Search


Working with Jobs
Table of contents:
Introduction
LSF Job Types
LSF Job Statuses
LSF Job Interaction Commands
LSF Job Errors and Exit Codes
bsub
Command flgas to run the jobs
Frequently Asked Questions - FAQ
Introduction of LSF Jobs:
In IBM Spectrum LSF, a job is a unit of work or a task that is submitted to the LSF system for execution. LSF jobs can range from simple commands to complex applications that require multiple tasks to be distributed across available compute resources in a cluster. Jobs can be of different types, such as interactive jobs, batch jobs, or parallel jobs.
LSF provides a rich set of commands for job management, ranging from job submission to querying job status, and modifying job parameters. By understanding commands, job statuses and errors, you can efficiently create, schedule, monitor, and manage jobs in your LSF environment.
For more detailed usage, consult the official LSF documentation or refer to FAQ section to explore individual command options.
LSF Job Types:
Batch Jobs: These are jobs that run without user interaction and are usually submitted to be executed at a later time.
Interactive Jobs: These jobs are launched by users to run interactively on LSF-managed nodes.
Parallel Jobs: These jobs are executed in parallel on multiple compute nodes, typically used for applications like simulations or data processing that require distributed computing.
LSF Job Statuses:
PEND: The job is waiting for resources or conditions to be met before execution.
RUN: The job is actively running on a compute node.
DONE: The job has successfully completed.
EXIT: The job terminated with an error. Check the exit code for details.
PSUSP: The job is in the process of being suspended.
USUSP: The job was manually suspended by a user.
SSUSP: The job was suspended by the system due to resource limitations.
SSUSP (suspended): The job is not running and can be manually resumed.
TIMEOUT: The job exceeded its time limit and was terminated.
WAITING: The job is in a queue, awaiting resources or its turn to execute.
PREEMP: The job was preempted to free up resources for higher-priority jobs.
ZOMBIE: The job has completed, but the system has not yet cleaned up resources.
STARTING: The job is in the process of starting execution.
RUNLIMIT: The job is running but has hit a resource limit (e.g., memory or CPU time).
LSF Job Interaction Commands:
bsub: Submit a job to LSF.
Example: bsub -q normal -o output.txt my_script.sh
bjobs: View the status of jobs.
Example: bjobs
bkill: Terminate a running or pending job.
Example: bkill 12345
bpeek: View the output or error of a running job.
Example: bpeek 12345
bmod: Modify the parameters (e.g., queue or priority) of a running or pending job.
Example: bmod -q high_priority 12345
bqueues: View the status of available queues.
Example: bqueues
bexit: Exit an interactive job session.
Example: bexit 12345
bstrap: Bootstrap an interactive job to set up the environment.
Example: bstrap -q interactive 12345
bresume: Resume a suspended job.
Example: bresume 12345
bstop: Suspend a running job.
Example: bstop 12345
brsvs Check Reservation and hosts allocated to it. Example: brsvs
LSF Job Errors and Exit Codes
Job cannot be submitted: The job could not be submitted to the queue due to invalid job parameters or missing resources.
Job exited with non-zero exit code: The job ran but encountered an error, indicated by a non-zero exit code.
Common exit codes:
1: General error (often due to incorrect parameters or script errors).
2: Command not found or command failed.
127: Command or script is not executable.
137: Job was killed due to exceeding memory limits.
139: Job was killed due to segmentation fault or other critical error.
Queue is full: The specified queue has reached its capacity and cannot accept new jobs.
Cannot find requested resources: The requested resources (e.g., CPU, memory) are not available to run the job.
Job suspended due to resource constraints: The job was suspended automatically due to insufficient system resources like memory or CPU.
Job preempted: The job was preempted to free resources for higher-priority jobs.
Job timed out: The job exceeded its maximum allowed execution time and was automatically terminated.
Exit code: 124 (if time limit exceeded).
Job could not be scheduled: The job is pending and cannot be scheduled due to insufficient resources or conflicts with other jobs.
Cannot connect to LSF server: The client cannot establish a connection with the LSF server, possibly due to network issues.
Invalid job parameters: Incorrect or invalid parameters were provided when submitting the job, such as an invalid queue or resource request.
Job not found: The job ID provided does not exist or has already completed or been removed.
bsub Flags for creating jobs
Flag	Description	Example
-q <queue_name>	Specify the queue to submit the job to (default: normal)	-q normal
-o <file>	Redirect standard output to a file (default: ~/lsf)	-o output.txt
-e <file>	Redirect standard error to a file (default: ~/lsf)	-e error.txt
-J <job_name>	Assign a name to the job	-J my_job
-I	Request an interactive job	-I
-Is	Request an interactive job and start a shell	-Is
-u <user>	Submit the job on behalf of another user	-u user123
-w <condition>	Specify a condition to wait for job completion	-w "done"
-env <env_var=value>	Set environment variables for the job	-env "MY_VAR=hello"
-M <max_mem>	Set the maximum memory for the job	-M 8GB
-n <num_processors>	Request a specific number of processors	-n 4
-gpu <num_gpus>	Request a specific number of GPUs	-gpu 1
-R <resource_request>	Specify resource requirements (memory, CPU, GPU)	-R "rusage[mem=4GB, cpu=2]" -gpu num=1
-R "select[<constraints>]"	Specify resource constraints for job selection	-R "select[type==x86_64]"
-L <login_shell>	Specify the login shell for interactive jobs	-L /bin/bash
-p <priority>	Specify job priority (1–1000, higher = higher priority)	-p 100
-P <project_name>	Specify the project for the job	-P my_project
-f	Force job submission even if the queue is full	-f
-A <account_name>	Specify an account name for billing	-A my_account
-h	Display help information	-h
-m <host_list>	Request specific hosts	-m "host1 host2"
-r	Allow resubmission if the job fails	-r
-X	Export the environment for interactive jobs	-X
-d <directory>	Specify job execution directory (default: home dir)	-d /path/to/directory
-x <execution_host>	Execute job on specific host or group	-x "host01"
-B <block_name>	Set a job block name	-B blockA
-H	Hold job after submission	-H
-h <host>	Direct a job to a specific host (no queues)	-h "host_name"
-M <memory>	Set job memory limit	-M 16GB
-T <time_limit>	Set max execution time (hh:mm format)	-T 02:00
-t <time_range>	Define a time window to run the job	-t "12:00-14:00"
-z	Enable job checkpointing (recommended for long jobs)	-z
-B	Send mail when job starts	-B
-l	Specify job limits (e.g. memory, CPU time)	-l "mem=4GB" -l "cput=4" -l "time=01:00"
-O <output>	Specify output file	-O output.log
Edit this page on GitHub
Previous
Know your Storage and Datasets: Storage and Data FAQ
Next
Job Scheduling, Managing and Monitoring: Submitting Jobs
Privacy
Terms of Use
Support via AskETE
Reporting Feedback
Have questions? Open an issue in GitHub.

cognitivecompcluster.

Last updated 03 February 2026
Copyright © 2026 IBM
Navigated to Working with Jobs
