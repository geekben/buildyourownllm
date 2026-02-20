#nvidia-smi
Fri Feb 20 22:29:38 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:40:00.0 Off |                    0 |
| N/A   39C    P0             26W /   70W |       1MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla T4                       Off |   00000000:E4:00.0 Off |                    0 |
| N/A   40C    P0             28W /   70W |       1MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

#time python3 babygpt_v11_hyper_params.py
step 0: train loss 8.0526, val loss 8.0509, speed: 25899.50 tokens/sec
step 200: train loss 5.2778, val loss 5.4912, speed: 21211.41 tokens/sec
step 400: train loss 4.8039, val loss 5.0707, speed: 21078.90 tokens/sec
step 600: train loss 4.5455, val loss 4.8311, speed: 21028.08 tokens/sec
step 800: train loss 4.3568, val loss 4.6639, speed: 20997.30 tokens/sec
step 1000: train loss 4.1690, val loss 4.5240, speed: 20983.05 tokens/sec
step 1200: train loss 4.0124, val loss 4.3883, speed: 20975.86 tokens/sec
step 1400: train loss 3.9135, val loss 4.3108, speed: 20966.88 tokens/sec
step 1600: train loss 3.8019, val loss 4.2420, speed: 20961.49 tokens/sec
step 1800: train loss 3.7151, val loss 4.1837, speed: 20958.10 tokens/sec
step 2000: train loss 3.6351, val loss 4.1463, speed: 20957.06 tokens/sec
step 2200: train loss 3.5787, val loss 4.1225, speed: 20955.94 tokens/sec
step 2400: train loss 3.5008, val loss 4.0783, speed: 20956.00 tokens/sec
step 2600: train loss 3.4424, val loss 4.0511, speed: 20920.90 tokens/sec
step 2800: train loss 3.3838, val loss 4.0397, speed: 20889.93 tokens/sec
step 3000: train loss 3.3253, val loss 4.0256, speed: 20865.50 tokens/sec
step 3200: train loss 3.2782, val loss 4.0174, speed: 20847.04 tokens/sec
step 3400: train loss 3.2284, val loss 4.0082, speed: 20834.47 tokens/sec
step 3600: train loss 3.1765, val loss 4.0016, speed: 20840.23 tokens/sec
step 3800: train loss 3.1347, val loss 3.9962, speed: 20845.72 tokens/sec
step 4000: train loss 3.0908, val loss 3.9940, speed: 20852.17 tokens/sec
step 4200: train loss 3.0424, val loss 3.9876, speed: 20857.49 tokens/sec
step 4400: train loss 3.0034, val loss 3.9846, speed: 20861.96 tokens/sec
step 4600: train loss 2.9654, val loss 3.9815, speed: 20865.50 tokens/sec
step 4800: train loss 2.9329, val loss 3.9839, speed: 20868.65 tokens/sec
春江总无情意。
待欲倩归来醉里。
夜深贪见君何事。
千点红麟香不定。
一饷清歌帘里去。
休说幸有别离多，何似秋光凝睇。

浣溪沙 赵长卿
柳气飘萧碧水仙。
洞房相映小庭扃。
于门风飐只车。
过眼清寒能几许
----------
往事乐倾国艳，乐历万壑烟斜。
一醉使君同好。
倩风雨调任他，月明无际，长是芳菲。
碧涧苍烟里，红叶无端。
问是昌工事业，千古酒高哉。
年年此夜费鸱夷。
问水流清彻，日变绿园桥。
肥橘可储良夜，唯有阿爷胎妃
----------

real    65m40.442s
user    50m33.513s
sys     15m19.869s
