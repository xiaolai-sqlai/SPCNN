# nohup srun -u --partition=Gvlab-S1-32 --gres=gpu:4 -n1 --ntasks=1 --ntasks-per-node=1 --quotatype=spot --job-name=scnn --kill-on-bad-exit=1 bash run.sh &
# nohup srun -u --partition=Gveval-S1 --gres=gpu:4 -n1 --ntasks=1 --ntasks-per-node=1 --job-name=spcnn --kill-on-bad-exit=1 bash run.sh &
nohup bash -u run.sh &
