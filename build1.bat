@echo off
chcp 65001 > nul
echo ========================================
echo   正在搭建 Conda（约一分钟）...
echo ========================================
echo.

conda create -n pde_pinn_env --override-channels ^
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ ^
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ ^
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ ^
    -c conda-forge -y
