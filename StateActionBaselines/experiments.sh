#############################
###  planar reacher exps  ###
#############################

python baselines_main.py configs/planar_reacher.yaml -e planar_reacher_baselines -o -s --nocodecopy

############################
###  panda reacher exps  ###
############################

python baselines_main.py configs/panda_reacher.yaml -e panda_reacher_baselines -o -s --nocodecopy
