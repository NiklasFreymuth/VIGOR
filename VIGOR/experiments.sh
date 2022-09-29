############################
###  panda reacher exps  ###
############################
python main.py configs/panda_reacher.yaml -e panda_reacher_vigor -o
python main.py configs/panda_reacher_drex.yaml -e panda_reacher_drex -o
python TrajectoryBC_main.py configs/panda_reacher_bct.yaml -e panda_reacher_bct -o
python TrajectoryBC_main.py configs/panda_reacher_bct.yaml -e panda_reacher_mbct -o

#############################
###  planar reacher exps  ###
#############################
python main.py configs/planar_reacher.yaml -e planar_reacher_vigor -o
python main.py configs/planar_reacher_drex.yaml -e planar_reacher_drex -o
python TrajectoryBC_main.py configs/planar_reacher_bct.yaml -e planar_reacher_bct -o
python TrajectoryBC_main.py configs/planar_reacher_bct.yaml -e planar_reacher_mbct -o
