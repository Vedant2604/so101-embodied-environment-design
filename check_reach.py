@echo off
call conda activate lerobot
start "p0_g0" cmd /k python so101\training\train.py --mode episodic   --steps 300000 --run p0_g0 --seed 0
start "p0_g1" cmd /k python so101\training\train.py --mode episodic   --steps 300000 --run p0_g1 --seed 1
start "p0_g2" cmd /k python so101\training\train.py --mode episodic   --steps 300000 --run p0_g2 --seed 2
start "p1_g0" cmd /k python so101\training\train.py --mode reset_free --steps 300000 --run p1_g0 --seed 0