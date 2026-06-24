@echo off
chcp 65001 > nul
echo ========================================
echo   正在安装核心依赖包（约需 10 分钟）...
echo ========================================
echo.

python -m pip install --upgrade pip
pip install torch==1.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.13.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install d2l==0.17.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install jupyter==1.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install matplotlib==3.9.4 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install notebook==7.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install ipympl==0.9.8 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sympy scipy numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyqt5 pyqt5-tools -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install qdarkstyle -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ========================================
echo   安装完成！
echo ========================================
pause