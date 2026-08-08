# src/data_utils.py
"""
采样点生成模块。

提供矩形定义域（1D/2D 空间，含/不含时间）内的采样点生成功能。
支持内部点、边界点（按边返回）、初始条件点采样，以及一次性生成所有点。
"""
import torch
from typing import Optional, Tuple, Dict, List, Union

class DomainSampler:
    """
    采样点生成器，支持矩形定义域（1D/2D 空间，含/不含时间）。

    功能：
        - sample_interior: 内部点采样
        - sample_boundary: 边界点采样（按边返回字典）
        - sample_initial: 初始条件点采样
        - sample_all: 一次性生成所有需要的点（内部 + 边界 + 初始）

    输出格式统一为 torch.Tensor，形状为 (n_points, dim)

    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: Tuple[float, float]

    :param y_range: y 轴范围 (y_min, y_max)，若为 None 表示一维空间
    :type y_range: Optional[Tuple[float, float]]

    :param t_range: t 轴范围 (t_min, t_max)，若为 None 表示稳态问题
    :type t_range: Optional[Tuple[float, float]]
    """
    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Optional[Tuple[float, float]] = None,
        t_range: Optional[Tuple[float, float]] = None,
    ):
        self.x_min, self.x_max = x_range
        self.has_y = y_range is not None
        if self.has_y:
            self.y_min, self.y_max = y_range
        self.has_t = t_range is not None
        if self.has_t:
            self.t_min, self.t_max = t_range
        self.spatial_dim = 2 if self.has_y else 1
        self.total_dim = self.spatial_dim + (1 if self.has_t else 0)
        self.side_labels = []
        if self.has_y:
            self.side_labels = ["bottom", "top", "left", "right"]
        else:
            self.side_labels = ["left", "right"]
    @property
    def dim(self) -> int:
        """返回点的总维度。"""
        return self.total_dim
    def _random_in_range(self, n: int, low: float, high: float) -> torch.Tensor:
        """
        生成 [low, high] 范围内的随机点。

        :param n: 采样点数
        :type n: int
        :param low: 下界
        :type low: float
        :param high: 上界
        :type high: float
        :return: (n, 1) 形状的张量
        :rtype: torch.Tensor
        """
        return torch.rand(n, 1) * (high - low) + low
    def _full_value(self, n: int, value: float) -> torch.Tensor:
        """
        生成全为 value 的张量。

        :param n: 采样点数
        :type n: int
        :param value: 填充值
        :type value: float
        :return: (n, 1) 形状的张量
        :rtype: torch.Tensor
        """
        return torch.full((n, 1), value)
    def _build_points(self, components: List[torch.Tensor]) -> torch.Tensor:
        """
        拼接多个分量张量为完整坐标张量。

        :param components: 各维度分量张量列表
        :type components: List[torch.Tensor]
        :return: (n_points, dim) 形状的坐标张量
        :rtype: torch.Tensor
        """
        return torch.cat(components, dim=1)
    # ============ 内部点采样 ============
    def sample_interior(self, n_points: int) -> torch.Tensor:
        """
        在定义域内部随机采样。

        :param n_points: 采样点数
        :type n_points: int
        :return: (n_points, dim) 坐标张量
        :rtype: torch.Tensor
        """
        components = [self._random_in_range(n_points, self.x_min, self.x_max)]
        if self.has_y:
            components.append(self._random_in_range(n_points, self.y_min, self.y_max))
        if self.has_t:
            components.append(self._random_in_range(n_points, self.t_min, self.t_max))
        return self._build_points(components)
    # ============ 边界点采样（按边返回） ============
    def sample_boundary(
        self,
        n_points_per_side: int = 50,
        sides: Optional[List[str]] = None,
        with_time: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        在矩形边界上按边采样。

        :param n_points_per_side: 每条边的采样点数
        :type n_points_per_side: int

        :param sides: 要采样的边列表，如 ["left", "right"]，默认为所有边
        :type sides: Optional[List[str]]

        :param with_time: 若含时间，是否在时间维度上随机采样（True）还是固定在 t_min（False）
        :type with_time: bool

        :return: 键为边名，值为 (n_points, dim) 坐标张量的字典
        :rtype: Dict[str, torch.Tensor]
        """
        if sides is None:
            sides = self.side_labels
        result = {}
        for side in sides:
            if side == "bottom" and self.has_y:
                x = self._random_in_range(n_points_per_side, self.x_min, self.x_max)
                y = self._full_value(n_points_per_side, self.y_min)
                components = [x, y]
            elif side == "top" and self.has_y:
                x = self._random_in_range(n_points_per_side, self.x_min, self.x_max)
                y = self._full_value(n_points_per_side, self.y_max)
                components = [x, y]
            elif side == "left":
                y = self._random_in_range(n_points_per_side, self.y_min, self.y_max) if self.has_y else None
                x = self._full_value(n_points_per_side, self.x_min)
                components = [x, y] if self.has_y else [x]
            elif side == "right":
                y = self._random_in_range(n_points_per_side, self.y_min, self.y_max) if self.has_y else None
                x = self._full_value(n_points_per_side, self.x_max)
                components = [x, y] if self.has_y else [x]
            else:
                continue
            if self.has_t:
                if with_time:
                    t = self._random_in_range(n_points_per_side, self.t_min, self.t_max)
                else:
                    t = self._full_value(n_points_per_side, self.t_min)
                components.append(t)
            result[side] = self._build_points(components)
        return result
    # ============ 初始条件点采样 ============
    def sample_initial(self, n_points: int) -> torch.Tensor:
        """
        采样初始条件点 (t = t_min)。

        :param n_points: 采样点数
        :type n_points: int
        :return: (n_points, dim) 坐标张量
        :rtype: torch.Tensor
        """
        if not self.has_t:
            raise ValueError("没有时间维度，无法采样初始条件。")
        components = [self._random_in_range(n_points, self.x_min, self.x_max)]
        if self.has_y:
            components.append(self._random_in_range(n_points, self.y_min, self.y_max))
        components.append(self._full_value(n_points, self.t_min))
        return self._build_points(components)
    # ============ 按轴采样（用于 Neumann 边界） ============
    def sample_axis(
        self,
        axis: str,
        value: float,
        n_points: int,
    ) -> torch.Tensor:
        """
        采样 axis = value 的超平面上的点，用于 Neumann 边界条件或固定边界。

        :param axis: 轴名称，'x' | 'y' | 't'
        :type axis: str

        :param value: 轴上的固定值
        :type value: float

        :param n_points: 采样点数
        :type n_points: int

        :return: (n_points, dim) 坐标张量
        :rtype: torch.Tensor
        """
        if axis == 'x':
            components = [self._full_value(n_points, value)]
            if self.has_y:
                components.append(self._random_in_range(n_points, self.y_min, self.y_max))
            if self.has_t:
                components.append(self._random_in_range(n_points, self.t_min, self.t_max))
        elif axis == 'y':
            if not self.has_y:
                raise ValueError("y 轴不存在")
            components = [self._random_in_range(n_points, self.x_min, self.x_max)]
            components.append(self._full_value(n_points, value))
            if self.has_t:
                components.append(self._random_in_range(n_points, self.t_min, self.t_max))
        elif axis == 't':
            if not self.has_t:
                raise ValueError("t 轴不存在")
            components = [self._random_in_range(n_points, self.x_min, self.x_max)]
            if self.has_y:
                components.append(self._random_in_range(n_points, self.y_min, self.y_max))
            components.append(self._full_value(n_points, value))
        else:
            raise ValueError(f"不支持的轴: {axis}")
        return self._build_points(components)
    # ============ 一次性生成所有采样点 ============
    def sample_all(
        self,
        n_interior: int = 1000,
        n_boundary_per_side: int = 50,
        n_initial: int = 200,
        return_structured: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]:
        """
        一次性生成内部点、边界点、初始条件点。

        :param n_interior: 内部点数量
        :type n_interior: int

        :param n_boundary_per_side: 每条边的边界点数量
        :type n_boundary_per_side: int

        :param n_initial: 初始条件点数量（仅含时间时有效）
        :type n_initial: int

        :param return_structured: True 返回字典，False 返回元组
        :type return_structured: bool

        :return:
            - 若 return_structured=True::

                {
                    'interior': points,
                    'boundary': {'left': points, 'right': points, ...},
                    'initial': points or None
                }

            - 若 return_structured=False: (interior_points, boundary_points, initial_points)
              boundary_points 为字典
        :rtype: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]]
        """
        result = {
            'interior': self.sample_interior(n_interior),
            'boundary': self.sample_boundary(n_boundary_per_side),
            'initial': self.sample_initial(n_initial) if self.has_t else None,
        }
        if return_structured:
            return result
        return result['interior'], result['boundary'], result['initial']

def get_interior_points(
    n_points: int,
    x_range: Tuple[float, float],
    y_range: Optional[Tuple[float, float]] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """
    便捷函数：获取内部点。

    :param n_points: 采样点数
    :type n_points: int
    :param x_range: x 轴范围
    :type x_range: Tuple[float, float]
    :param y_range: y 轴范围，可选
    :type y_range: Optional[Tuple[float, float]]
    :param t_range: t 轴范围，可选
    :type t_range: Optional[Tuple[float, float]]
    :return: (n_points, dim) 坐标张量
    :rtype: torch.Tensor
    """
    sampler = DomainSampler(x_range, y_range, t_range)
    return sampler.sample_interior(n_points)
def get_boundary_points(
    n_points_per_side: int,
    x_range: Tuple[float, float],
    y_range: Optional[Tuple[float, float]] = None,
    t_range: Optional[Tuple[float, float]] = None,
    sides: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：获取边界点（按边返回字典）。

    :param n_points_per_side: 每条边的采样点数
    :type n_points_per_side: int
    :param x_range: x 轴范围
    :type x_range: Tuple[float, float]
    :param y_range: y 轴范围，可选
    :type y_range: Optional[Tuple[float, float]]
    :param t_range: t 轴范围，可选
    :type t_range: Optional[Tuple[float, float]]
    :param sides: 要采样的边列表，默认为所有边
    :type sides: Optional[List[str]]
    :return: 键为边名，值为坐标张量的字典
    :rtype: Dict[str, torch.Tensor]
    """
    sampler = DomainSampler(x_range, y_range, t_range)
    return sampler.sample_boundary(n_points_per_side, sides=sides)
def get_initial_points(
    n_points: int,
    x_range: Tuple[float, float],
    y_range: Optional[Tuple[float, float]] = None,
    t_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    便捷函数：获取初始条件点。

    :param n_points: 采样点数
    :type n_points: int
    :param x_range: x 轴范围
    :type x_range: Tuple[float, float]
    :param y_range: y 轴范围，可选
    :type y_range: Optional[Tuple[float, float]]
    :param t_range: t 轴范围，默认为 (0.0, 1.0)
    :type t_range: Tuple[float, float]
    :return: (n_points, dim) 坐标张量
    :rtype: torch.Tensor
    """
    sampler = DomainSampler(x_range, y_range, t_range)
    return sampler.sample_initial(n_points)
def get_all_points(
    n_interior: int = 1000,
    n_boundary_per_side: int = 50,
    n_initial: int = 200,
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Optional[Tuple[float, float]] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    便捷函数：一次性获取所有采样点。

    :param n_interior: 内部点数量
    :type n_interior: int
    :param n_boundary_per_side: 每条边的边界点数量
    :type n_boundary_per_side: int
    :param n_initial: 初始条件点数量
    :type n_initial: int
    :param x_range: x 轴范围
    :type x_range: Tuple[float, float]
    :param y_range: y 轴范围，可选
    :type y_range: Optional[Tuple[float, float]]
    :param t_range: t 轴范围，可选
    :type t_range: Optional[Tuple[float, float]]
    :return: 包含 'interior', 'boundary', 'initial' 的字典
    :rtype: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
    """
    sampler = DomainSampler(x_range, y_range, t_range)
    return sampler.sample_all(n_interior, n_boundary_per_side, n_initial)

if __name__ == "__main__":
    print("=" * 60)
    print("DomainSampler 使用示例")
    print("=" * 60)
    # ---------- 示例 1: 1D 稳态（只有 x） ----------
    print("\n[示例 1] 1D 稳态: x ∈ [0, 1]")
    sampler_1d = DomainSampler(x_range=(0.0, 1.0))
    print(f"  维度: {sampler_1d.dim}D, 含时间: {sampler_1d.has_t}, 边界: {sampler_1d.side_labels}")
    interior = sampler_1d.sample_interior(5)
    print(f"  内部点 (5个):\n{interior}")
    boundary = sampler_1d.sample_boundary(n_points_per_side=3)
    print(f"  边界点 (每条边3个):")
    for side, pts in boundary.items():
        print(f"    {side}: {pts.squeeze().tolist()}")
    # ---------- 示例 2: 2D 稳态（x, y） ----------
    print("\n[示例 2] 2D 稳态: x ∈ [0, 1], y ∈ [-1, 1]")
    sampler_2d = DomainSampler(x_range=(0.0, 1.0), y_range=(-1.0, 1.0))
    print(f"  维度: {sampler_2d.dim}D, 边界: {sampler_2d.side_labels}")
    interior = sampler_2d.sample_interior(5)
    print(f"  内部点 (5个):\n{interior}")
    boundary = sampler_2d.sample_boundary(n_points_per_side=2)
    print(f"  边界点 (每条边2个):")
    for side, pts in boundary.items():
        print(f"    {side}: {pts.squeeze().tolist()}")
    # ---------- 示例 3: 1D 含时间（x, t） ----------
    print("\n[示例 3] 1D 含时间: x ∈ [0, 1], t ∈ [0, 2]")
    sampler_1dt = DomainSampler(x_range=(0.0, 1.0), t_range=(0.0, 2.0))
    print(f"  维度: {sampler_1dt.dim}D, 含时间: {sampler_1dt.has_t}, 边界: {sampler_1dt.side_labels}")
    interior = sampler_1dt.sample_interior(5)
    print(f"  内部点 (5个):\n{interior}")
    initial = sampler_1dt.sample_initial(3)
    print(f"  初始条件点 (t=0, 3个):\n{initial}")
    # 边界采样：with_time=True（默认，时间随机）
    boundary = sampler_1dt.sample_boundary(n_points_per_side=2, sides=["left", "right"])
    print(f"  边界点 (with_time=True, 每条边2个):")
    for side, pts in boundary.items():
        print(f"    {side}: {pts.squeeze().tolist()}")
    # 边界采样：with_time=False（时间固定在 t_min）
    boundary_fixed = sampler_1dt.sample_boundary(n_points_per_side=2, sides=["left", "right"], with_time=False)
    print(f"  边界点 (with_time=False, 每条边2个):")
    for side, pts in boundary_fixed.items():
        print(f"    {side}: {pts.squeeze().tolist()}")
    # ---------- 示例 4: 2D 含时间（x, y, t） ----------
    print("\n[示例 4] 2D 含时间: x ∈ [0, 1], y ∈ [0, 1], t ∈ [0, 1]")
    sampler_2dt = DomainSampler(x_range=(0.0, 1.0), y_range=(0.0, 1.0), t_range=(0.0, 1.0))
    print(f"  维度: {sampler_2dt.dim}D, 边界: {sampler_2dt.side_labels}")
    interior = sampler_2dt.sample_interior(5)
    print(f"  内部点 (5个):\n{interior}")
    initial = sampler_2dt.sample_initial(3)
    print(f"  初始条件点 (t=0, 3个):\n{initial}")
    # ---------- 示例 5: sample_axis 按轴采样 ----------
    print("\n[示例 5] sample_axis 按轴采样")
    # 5a: 采样 x=0 平面（用于左边界）
    pts_x0 = sampler_2dt.sample_axis(axis='x', value=0.0, n_points=3)
    print(f"  x=0 平面 (3个点):\n{pts_x0}")
    # 5b: 采样 t=0 平面（用于初始条件）
    pts_t0 = sampler_2dt.sample_axis(axis='t', value=0.0, n_points=3)
    print(f"  t=0 平面 (3个点):\n{pts_t0}")
    # ---------- 示例 6: sample_all 一次性采样 ----------
    print("\n[示例 6] sample_all 一次性采样 (2D 稳态)")
    all_pts = sampler_2d.sample_all(n_interior=10, n_boundary_per_side=3, n_initial=0)
    print(f"  内部点数量: {all_pts['interior'].shape[0]}")
    print(f"  边界点数量: {sum(pts.shape[0] for pts in all_pts['boundary'].values())}")
    print(f"  初始条件点: {all_pts['initial']}")
    # ---------- 示例 7: 便捷函数 ----------
    print("\n[示例 7] 便捷函数")
    interior = get_interior_points(5, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    print(f"  get_interior_points: {interior.shape}")
    boundary = get_boundary_points(3, x_range=(0.0, 1.0), y_range=(0.0, 1.0), sides=["left", "right"])
    print(f"  get_boundary_points: {list(boundary.keys())}")
    initial = get_initial_points(3, x_range=(0.0, 1.0), t_range=(0.0, 1.0))
    print(f"  get_initial_points: {initial.shape}")
    all_pts = get_all_points(n_interior=10, n_boundary_per_side=3, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    print(f"  get_all_points: 内部 {all_pts['interior'].shape[0]}, 边界 {sum(pts.shape[0] for pts in all_pts['boundary'].values())}")
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
