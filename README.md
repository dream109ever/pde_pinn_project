# 基于机器学习的偏微分求解（大一立项）

本项目基于物理信息神经网络（PINN）实现对**一阶至六阶常/变系数线性常微分方程**及**二阶偏微分方程**的求解，并通过 PyQt5 图形界面实时可视化呈现损失曲线与解曲面。  
网络搭建基于 `PyTorch` 库，图形界面基于 `PyQt5` 库，方程解析解基于 `numpy`、`sympy`、`scipy` 等数学库。  
项目中期已完成图形界面主体搭建与核心求解引擎的初步集成，当前为第二版代码（已重构），支持更清晰的模块化结构与更稳定的求解流程。  
**（当前代码部分内容还存在些许漏洞，但整体上达到了预期，不再安排改进，内容与设计思路仅供学习参考。）**

## 1. 文件结构
```
.
├── .github/workflows/                  # GitHub Actions 工作流
│   └── docs.yml                        # 自动部署文档到 GitHub Pages
├── config/                             # 配置文件目录
│   └── save_counter.json               # 训练结果保存计数（废弃）
├── docs/                               # Sphinx 文档源文件
│   ├── _build/html/index.html          # API 网页文件
│   ├── modules/api.rst                 # API 参考手册
│   ├── conf.py                         # Sphinx 配置
│   └── index.rst                       # 文档首页
├── images/                             # 图片资源文件（README 插图）
├── notebooks/                          # jupyter 脚本目录
│   └── train_demo.ipynb                # 训练演示脚本 
├── results/                            # 训练结果保存目录（废弃）
├── scripts/                            # 测试脚本
│   ├── count.py                        # 统计当前总文件数和代码量
│   ├── gui_test.py                     # 使用固定方程测试可视化界面的脚本
│   └── train_pinn.py                   # 无可视化的训练测试脚本
├── src/                                # 核心源代码
│   ├── __init__.py                     # 包入口
│   ├── data_utils.py                   # 采样点生成器（内部点、边界点）
│   ├── function_factory.py             # PDE 解析、损失函数、基准解生成
│   ├── network_factory.py              # 神经网络自动构建与复杂度分析
│   ├── plotting_core.py                # 纯绘图逻辑（无后端依赖）
│   ├── plotting_qt.py                  # Qt 可视化控件
│   ├── trainer.py                      # PINN 训练器
│   └── visualization.py                # Notebook 环境可视化
├── ui/                                 # 图形界面模块
│   ├── icons/                          # 图标资源
│   ├── pages/                          # 页面控件
│   │   ├── base_widgets.py             # 基础控件类
│   │   ├── mode_selection_page.py      # 模式选择页
│   │   ├── pinn_input_page.py          # PINN 输入页
│   │   ├── pinn_plot_page.py           # PINN 训练与绘图页
│   │   ├── plot_window_page.py         # 独立绘图窗口
│   │   ├── settings_dialog_page.py     # 设置对话框
│   │   └── solver_page.py              # 精确解析解求解页
│   ├── __init__.py                     # UI 包入口
│   ├── app_config.py                   # 全局配置管理
│   ├── main_window.py                  # 主窗口
│   └── theme_manager.py                # 主题管理
├── .gitignore                          # git 配置文件
├── README.md                           # 本文件
├── build1.bat                          # 环境搭建脚本
├── build2.bat                          # Python 安装脚本
├── build3.bat                          # 库安装脚本
├── logo.ico                            # 桌面图标
└── main.py                             # 程序入口
```

## 2. 运行方式

**1. 环境配置**：

- 把全部文件拉取到本地，进入 Conda 的 base 环境，进入项目根目录，输入运行 `build1.bat`，搭建新环境
- 输入 `conda activate pde_pinn_env`，进入新建环境，输入运行 `build2.bat`，安装 Python
- 输入运行 `build3.bat`，安装依赖库，该过程需要十分钟左右，过程中出现报错是正常现象，等待出现安装成功提示即可

**2. 启动程序**：

- 输入运行 `python main.py` 即可使用完整功能

**3. 打包程序（分发时用）**：

- 在项目根目录输入 `pyinstaller main.py --noconsole --hidden-import PyQt5.QtXml --icon="logo.ico"`，等待约十分钟
- 项目根目录下会出现 `build/` 和 `dist/` 两个文件夹，打包好的可执行文件在 `dist/main/` 文件夹中
- 若运行时提示缺失文件，需要手动把项目根目录中的 `ui/` 文件夹复制到 `dist/main/_internal/` 文件夹中

## 3. 使用说明

### 1. 模式选择
直接运行 Python 文件或打包好的可执行文件启动程序后进入模式选择页，提供两种工作模式：

- 精确解析解模式：基于 sympy 与 scipy 求解线性 ODE/PDE，输出解析表达式或数值基准解
- PINN 神经网络模式：基于物理信息神经网络进行训练，实时可视化求解过程

点击右下角退出程序；点击右上角可选择主题配色；运行过程中任意时刻可通过右上角关闭窗口。  
<img src="./images/page1.0.png" alt="page1.0" width="100%" />

主题配色共四种：  
<img src="./images/page1.2.png" alt="page1.2" width="100%" />

不同配色主界面展示如下：  

<table align="center">
  <tr>
    <td align="center">
      <img src="./images/page1.0.png" alt="冰川蓝" width="100%" />
      <br />
      <b>冰川蓝 (默认)</b>
    </td>
    <td align="center">
      <img src="./images/page1.3.png" alt="深海曜石" width="100%" />
      <br />
      <b>深海曜石 (暗黑模式)</b>
    </td>
    <td align="center">
      <img src="./images/page1.4.png" alt="薄荷翡翠" width="100%" />
      <br />
      <b>薄荷翡翠 (清新)</b>
    </td>
    <td align="center">
      <img src="./images/page1.5.png" alt="幻境紫罗兰" width="100%" />
      <br />
      <b>幻境紫罗兰</b>
    </td>
  </tr>
</table>

### 2. 精确解析解模式

在主界面选择第一个模式进入精确解展示界面，如下：  
<img src="./images/page2.1.png" alt="page2.1" width="100%" />

按照方程类型分为四类，其中一维含时的阶数可在 1 ~ 6 之间选择，其他三种阶数固定 2：  
<img src="./images/page2.2.png" alt="page2.2" width="100%" />

通过添加系数和条件来配置方程，配置好的方程会生成预览：   
<table align="center">
  <tr>
    <td align="center">
      <img src="./images/page2.3.png" alt="page2.3" width="100%" />
      <br />
    </td>
    <td align="center">
      <img src="./images/page2.4.png" alt="page2.4" width="100%" />
      <br />
    </td>
  </tr>
</table>

填写完成所有方程配置后，可点击开始求解，程序会自动分析配置并求出解析解（仅部分支持），有时会有轻微卡顿或卡退（因某些漏洞未完全修复），若输入不合规，部分会弹窗提示修改。  
<table align="center">
  <tr>
    <td align="center">
      <img src="./images/page2.5.png" alt="page2.5" width="100%" />
      <br />
    </td>
    <td align="center">
      <img src="./images/page2.6.png" alt="page2.6" width="100%" />
      <br />
    </td>
  </tr>
</table>

求解完毕后可点击绘制解函数，绘制当前函数的图像（仅部分支持）：  
<img src="./images/page2.7.png" alt="page2.7" width="100%" />


### 3. PINN 神经网络模式

在主界面选择第二个模式进入神经网络展示界面：  
<img src="./images/page3.1.png" alt="page3.1" width="100%" />

添加系数和条件同上：  
<img src="./images/page3.2.png" alt="page3.2" width="100%" />

点击下一页进入训练可视化界面。可选择配置网络，也可以直接开始（默认配置是根据方程特征自动生成的）：  
<img src="./images/page3.3.png" alt="page3.3" width="100%" />

训练过程中会每隔一定轮数更新损失曲线和预测图，展现训练可行性：   
<table align="center">
  <tr>
    <td align="center">
      <img src="./images/page3.4.png" alt="page3.4" width="100%" />
      <br />
    </td>
    <td align="center">
      <img src="./images/page3.5.png" alt="page3.5" width="100%" />
      <br />
    </td>
  </tr>
</table>

点击停止训练，稍等片刻，会停止训练线程并展示最后结果（此处若频繁操作可能引起 bug）：    
<img src="./images/page3.6.png" alt="page3.6" width="100%" />

### 4. 部分异常情况说明

- 若输入表达式过于复杂，可能导致误差极大，解析解精度丧失，卡顿；
- 当前对自定义边界系数的支持有限，可能无法正常训练；
- 随着训练轮数或采样点增加，会逐渐出现卡顿现象；
- 隐藏层参数输入缺乏校验，非法输入导致解析失败或崩溃；
- 源项对填入表达式校检不足，可能会出现除零错误；
- 定义域输入框和边界条件系数输入框未校验是否为合法数值，NaN 传入后污染采样坐标；
- 部分输入错误无法识别，传入求解导致崩溃；
- 对于源项较为复杂的方程，网络无法学习到有效特征；
- 对于一维稳态问题，若方程只有最高次数项且系数为 1、源项不为 1，会导致崩溃，其他一些特殊系数组合也会导致崩溃；
- 在二维含时、方程含有二维时间时绘图，会导致极其卡顿甚至崩溃，且绘制异常；
- 训练绘制界面第一次返回时无法正确保存配置，随后正常；
- 求解与停止按钮频繁操作时，线程紊乱，无法正常工作；
- 结果展示的绘制图形位置不正确，暂时使用的每次重绘的方法替代；
- 待补充。

## 4. 进度提要

- 项目于 2025 年 10 月底立项，小组成员在前期完成了《动手学深度学习 PyTorch 版》前三章的学习，掌握了线性神经网络、多层感知机及自动微分的基础。
- 2026 年 3 月完成第一阶段，实现了对一维泊松方程的 PINN 求解并在 jupyter 脚本可视化。
- 2026 年 4 月完成对《数学物理方法》的学习，对常见二阶偏微分方程的类型及其级数解的求法有了初步了解。
- 2026 年 5 月进行系统培训，同时制定中期及结题主要目标。
- 2026 年 6 月完成 PyQt5 图形界面三页面（欢迎页、方程配置页、训练可视化页）的初步集成，支持实时绘图与训练控制。
- 2026 年 7 ~ 8 月完成第一版代码重构，引入模块化设计与主题管理，重新绘制界面并提高可扩展性，完成 Sphinx 文档配置，支持 API 自动生成与 GitHub Pages 自动部署。

## 5. 参考资料

- 主要参考：《动手学深度学习 PyTorch 版》、《数学物理方法顾樵版》
- PyQt5 开发：Qt 官方文档、PyQt5 示例代码、白月黑羽官方学习网站
- 辅助工具：DeepSeek（代码生成与调试指导）、CSDN 社区（环境配置与问题排查）、GitHub Copilot

## 6. 问题与解决

### 6.1 符号表达式解析与变量管理

- **问题**：对输入表达式的解析硬编码，可扩展性极差，后期调试费时费力。
- **解决**：根据方程类型自动呈现对应的系数类型，且仅展示可行部分。

### 6.2 采样器对时间维度的统一处理

- **问题**：处理初始条件和边界条件的组合时，无法合理分开判定逻辑，为简化模型，暂时将时间 `t` 当作第二个空间维度（即“y”）处理，但出现很多其他问题。
- **解决**：在输入时把时间和空间分离，分为 一维/二维 x 稳态/含时 四种组合，逻辑上更清晰。

### 6.3 可视化界面设计与逻辑处理

- **问题**：当前界面存在大量逻辑不完备的地方，对多种极端输入暂未测试和处理。
- **解决**：对界面进行重绘，提高了逻辑性，但极端输入仍未处理。

### 6.4 多线程处理

- **问题**：为了保持界面流畅，需要把训练和求解过程放在单独的线程，目前有简要实现，但未达预期。
- **解决**：已经实现线程分离，但在多次创建和删除线程时会卡顿。

### 6.5 解析解扩展

- **问题**：目前只对极少部分方程进行了解析解处理，还需扩展。
- **解决**：已经完成对更多常见类型方程的解析解扩展。

## 7. 下一步规划

本项目完成了基于 PINN 的 PDE 求解器开发，涵盖了从方程配置、解析解求解到神经网络训练与可视化展示的完整流程。代码结构经过两轮重构，形成了相对清晰的模块分层，并完成了 Sphinx 文档自动生成与 GitHub Pages 部署。考虑到大一年度项目的时间跨度与团队知识积累，当前系统在功能层面已达到项目预期。  
当然，在项目推进过程中也逐渐认识到，PINN 作为一种通用 PDE 求解方法在实际工程场景中仍面临诸多挑战（如训练稳定性、高维问题的计算效率等），该方向更适合作为机器学习的练手项目，而非长期的研究主线。这也是我们选择在此节点为项目画上句号的主要原因。  
由于项目周期、团队规模与知识储备的限制，当前版本在稳定性与异常处理方面方面仍有不足（详见第 6 节），这些问题需要更系统的输入校验机制与更稳健的多线程设计才能根本解决，已超出本阶段的能力范围。但作为一个大一学年的探索性项目，现有代码框架为后续学习和开发留下了可扩展的基础，若后续有兴趣继续推进，可以考虑从以下方向深入：
- 完善输入校验与异常处理机制
- 优化多线程训练时的界面响应流畅度
- 扩展解析解的自动生成能力
- 引入更高效的自适应采样策略
- 学习更先进的求解策略

## 8. 文件和函数结构说明

### 函数和库调用关系图（概略，待完善）

```mermaid
flowchart TD
    %% ============================================================
    %% 第一层：主入口
    %% ============================================================
    MAIN["main.py<br>程序入口"]
    UI_INIT["ui/__init__.py<br>run_gui()"]

    %% ============================================================
    %% 第二层：UI 包
    %% ============================================================
    MAIN_WINDOW["ui/main_window.py<br>MainWindow"]
    THEME_MANAGER["ui/theme_manager.py<br>ThemeManager"]
    APP_CONFIG["ui/app_config.py<br>AppConfig"]

    %% ============================================================
    %% 第三层：UI 页面
    %% ============================================================
    MODE_PAGE["ui/pages/mode_selection_page.py<br>ModeSelectionPage"]
    SOLVER_PAGE["ui/pages/solver_page.py<br>SolverPage"]
    PINN_INPUT_PAGE["ui/pages/pinn_input_page.py<br>PinnInputPage"]
    PINN_PLOT_PAGE["ui/pages/pinn_plot_page.py<br>PinnPlotPage"]
    PLOT_WINDOW["ui/pages/plot_window_page.py<br>PlotWindow"]
    SETTINGS_DIALOG["ui/pages/settings_dialog_page.py<br>SettingsDialog"]
    BASE_WIDGETS["ui/pages/base_widgets.py<br>BasePage / BaseDialog / SolverThread"]

    %% ============================================================
    %% 第四层：src 核心模块
    %% ============================================================
    FUNCTION_FACTORY["src/function_factory.py<br>solve_pde()"]
    NETWORK_FACTORY["src/network_factory.py<br>build_model() / suggest_network()"]
    DATA_UTILS["src/data_utils.py<br>DomainSampler"]
    TRAINER["src/trainer.py<br>PINNTrainer"]
    PLOTTING_CORE["src/plotting_core.py<br>prepare_* / draw_*"]
    PLOTTING_QT["src/plotting_qt.py<br>Steady1DPlotWidget / Transient1DPlotWidget<br>Steady2DPlotWidget / Transient2DPlotWidget"]
    VISUALIZATION["src/visualization.py<br>plot_*（Notebook 版本）"]

    %% ============================================================
    %% 第五层：外部依赖库
    %% ============================================================
    TORCH["PyTorch<br>torch / torch.nn / torch.optim"]
    NUMPY["NumPy<br>numpy"]
    SYMPY["SymPy<br>sympy"]
    SCIPY["SciPy<br>scipy.integrate / scipy.optimize"]
    MATPLOTLIB["Matplotlib<br>matplotlib.pyplot / mpl_toolkits"]
    PYQT5["PyQt5<br>QtWidgets / QtCore / QtGui"]

    %% ============================================================
    %% 调用关系
    %% ============================================================

    %% ---- 主入口到 UI ----
    MAIN -->|"from ui import run_gui"| UI_INIT
    UI_INIT -->|"创建实例"| MAIN_WINDOW
    UI_INIT -->|"ThemeManager.instance()"| THEME_MANAGER
    THEME_MANAGER -->|"读取/写入"| APP_CONFIG

    %% ---- MainWindow 到各页面 ----
    MAIN_WINDOW -->|"mode_selected → switch_mode()"| MODE_PAGE
    MAIN_WINDOW -->|"精确模式"| SOLVER_PAGE
    MAIN_WINDOW -->|"PINN 模式"| PINN_INPUT_PAGE
    MAIN_WINDOW -->|"equation_configured → go_to_solver_config()"| PINN_PLOT_PAGE

    %% ---- 精确解模式依赖 ----
    SOLVER_PAGE -->|"继承"| BASE_WIDGETS
    SOLVER_PAGE -->|"启动 SolverThread"| BASE_WIDGETS
    SOLVER_PAGE -->|"点击绘制解函数 → 打开"| PLOT_WINDOW
    BASE_WIDGETS -->|"SolverThread.run() → 调用"| FUNCTION_FACTORY

    PLOT_WINDOW -->|"继承"| BASE_WIDGETS
    PLOT_WINDOW -->|"导入并使用"| PLOTTING_QT
    PLOT_WINDOW -->|"get_exact_callable() → sympy"| SYMPY
    PLOT_WINDOW -->|"apply_theme() → ThemeManager"| THEME_MANAGER

    %% ---- PINN 模式依赖 ----
    PINN_INPUT_PAGE -->|"继承"| BASE_WIDGETS
    PINN_INPUT_PAGE -->|"suggest_network()"| NETWORK_FACTORY

    PINN_PLOT_PAGE -->|"继承"| BASE_WIDGETS
    PINN_PLOT_PAGE -->|"LossPlotWidget / Steady*PlotWidget"| PLOTTING_QT
    PINN_PLOT_PAGE -->|"启动 SolverThread"| BASE_WIDGETS
    PINN_PLOT_PAGE -->|"启动 PINNTrainerThread"| TRAINER
    PINN_PLOT_PAGE -->|"build_model()"| NETWORK_FACTORY

    %% ---- src 内部依赖 ----
    FUNCTION_FACTORY -->|"调用"| INPUT_PARSER["InputParser"]
    FUNCTION_FACTORY -->|"调用"| LOSS_GENERATOR["LossGenerator"]
    FUNCTION_FACTORY -->|"调用"| ANALYTICAL_SOLVER["AnalyticalSolverHub"]
    FUNCTION_FACTORY -->|"数据交换"| DATA_UTILS

    NETWORK_FACTORY -->|"调用"| COMPLEXITY_ANALYZER["ComplexityAnalyzer"]
    NETWORK_FACTORY -->|"调用"| NETWORK_CONFIG["NetworkConfigGenerator"]
    NETWORK_FACTORY -->|"使用"| TORCH

    TRAINER -->|"使用"| TORCH
    TRAINER -->|"采样"| DATA_UTILS

    PLOTTING_CORE -->|"使用"| NUMPY
    PLOTTING_CORE -->|"使用"| MATPLOTLIB

    PLOTTING_QT -->|"使用"| PYQT5
    PLOTTING_QT -->|"继承"| PLOTTING_CORE
    PLOTTING_QT -->|"使用"| MATPLOTLIB

    VISUALIZATION -->|"依赖"| PLOTTING_CORE
    VISUALIZATION -->|"使用"| MATPLOTLIB

    %% ---- 外部库依赖 ----
    FUNCTION_FACTORY -->|"使用"| SYMPY
    FUNCTION_FACTORY -->|"使用"| SCIPY
    FUNCTION_FACTORY -->|"使用"| TORCH
    FUNCTION_FACTORY -->|"使用"| NUMPY

    TRAINER -->|"使用"| NUMPY
    TRAINER -->|"使用"| TORCH

    DATA_UTILS -->|"使用"| TORCH

    %% ============================================================
    %% 样式定义
    %% ============================================================
    classDef main fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef ui fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef page fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef core fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    classDef lib fill:#fce4ec,stroke:#c62828,stroke-width:1.5px,stroke-dasharray: 5 5
    classDef hidden fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 3 3

    class MAIN,UI_INIT main
    class MAIN_WINDOW,THEME_MANAGER,APP_CONFIG ui
    class MODE_PAGE,SOLVER_PAGE,PINN_INPUT_PAGE,PINN_PLOT_PAGE,PLOT_WINDOW,SETTINGS_DIALOG,BASE_WIDGETS page
    class FUNCTION_FACTORY,NETWORK_FACTORY,DATA_UTILS,TRAINER,PLOTTING_CORE,PLOTTING_QT,VISUALIZATION core
    class INPUT_PARSER,LOSS_GENERATOR,ANALYTICAL_SOLVER,COMPLEXITY_ANALYZER,NETWORK_CONFIG hidden
    class TORCH,NUMPY,SYMPY,SCIPY,MATPLOTLIB,PYQT5 lib
```

### 界面关联图（概略）
```mermaid
flowchart LR
    %% ===== 主入口 =====
    A["主界面<br>ModeSelectionPage"]

    %% ===== 设置（独立分支） =====
    F["⚙ 设置对话框<br>SettingsDialog"]

    %% ===== 精确解模式分支 =====
    subgraph Exact[精确解析解模式]
        direction LR
        B["精确求解页<br>SolverPage"]
        C["绘图窗口<br>PlotWindow"]
        B -->|"求解成功 → 点击绘制"| C
        C -->|"关闭"| B
    end

    %% ===== PINN 模式分支 =====
    subgraph PINN[PINN 神经网络模式]
        direction LR
        D["输入页<br>PinnInputPage"]
        E["训练与绘图页<br>PinnPlotPage"]
        D -->|"配置校验通过 → 点击下一页"| E
        E -->|"点击返回"| D
    end

    %% ===== 顶层跳转 =====
    A -->|"点击精确解析解模式"| B
    A -->|"点击PINN神经网络模式"| D
    A -->|"点击⚙设置"| F
    F -->|"关闭"| A
    B -->|"返回主菜单"| A
    D -->|"返回"| A
```

## 9. 许可证

本项目采用 MIT 许可证开源。详见项目根目录下的 `LICENSE` 文件。

Copyright (c) 2026 dream109ever

特此授予任何人免费获得本软件及相关文档的许可，不受限制地处理本软件，包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或销售副本的权利。

## 10. 联系方式

如有问题反馈、合作意愿或改进建议，欢迎通过以下方式联系：

- **邮箱**：189211640@qq.com

欢迎交流指正。
