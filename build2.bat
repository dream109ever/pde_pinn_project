@echo off
chcp 65001 > nul
echo ========================================
echo   正在配置 Python 环境（约两分钟）...
echo ========================================
echo.

conda install python=3.9 --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ -c conda-forge -y

