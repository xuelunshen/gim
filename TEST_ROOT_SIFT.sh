#! /bin/bash
gpus=$1
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests            GL3D
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests           KITTI --img_size 1240
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests          GTASfM
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests         ICLNUIM
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests        MultiFoV
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests      BlendedMVS
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests        SceneNet
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests          ETH3DI --img_size 1600
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests          ETH3DO --img_size 1600
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests   RobotcarNight
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests  RobotcarSeason --max_samples 2000
python test.py --gpus gpus --weight root_sift --version vanilla --test --batch_size 1 --tests RobotcarWeather
