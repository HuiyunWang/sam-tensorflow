

# msvd soft
python msvd_soft_main.py --gpu_id=0 --step --centers_num=64 --lr=0.0001 --epoch=20 --d_w2v=512 --output_dim=512 --reduction_dim=512 

# msvd hard
python msvd_hard_main.py --gpu_id=0 --step --centers_num=64 --lr=0.0001 --epoch=20 --d_w2v=512 --output_dim=512 --reduction_dim=512 

# mvad soft
python mvad_soft_main.py --gpu_id=0 --centers_num=64 --lr=0.0001 --epoch=20 --d_w2v=512 --output_dim=512 --reduction_dim=512 

# mvad hard
python mvad_hard_main.py --gpu_id=0 --centers_num=64 --lr=0.0001 --epoch=20 --d_w2v=512 --output_dim=512 --reduction_dim=512 