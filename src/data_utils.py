import torch

class DomainSampler:
    """
    采样点生成器，支持矩形定义域（2D 空间或 1D 空间+时间）。
    功能：生成内部点、边界点、初始条件点。
    """
    def __init__(self, x_range, y_range=None, t_range=None):
        """
        参数:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max) 或 None（1D 问题）
            t_range: (t_min, t_max) 或 None（稳态问题）
        """
        self.x_min, self.x_max = x_range
        self.has_y = y_range is not None
        if self.has_y:
            self.y_min, self.y_max = y_range
        self.has_t = t_range is not None
        if self.has_t:
            self.t_min, self.t_max = t_range
    def sample_interior(self, n_points):
        """
        在定义域内部随机采样，形状根据是否有 y 和时间确定。
        返回坐标张量，形状 (n_points, dim)，dim 为空间维度（+1 若含时间）。
        """
        points = []
        points.append(torch.rand(n_points, 1) * (self.x_max - self.x_min) + self.x_min)
        if self.has_y:
            points.append(torch.rand(n_points, 1) * (self.y_max - self.y_min) + self.y_min)
        if self.has_t:
            points.append(torch.rand(n_points, 1) * (self.t_max - self.t_min) + self.t_min)
        return torch.cat(points, dim=1)
    def sample_boundary(self, n_points_per_segment=None):
        """
        在矩形边界上采样（若为 2D），返回边界点坐标和对应的边界部分标识。
        若为 1D，则直接返回两个端点。
        """
        if not self.has_y:
            points = torch.tensor([[self.x_min], [self.x_max]], dtype=torch.float32)
            boundary_ids = torch.tensor([0, 1])
            return points, boundary_ids
        if n_points_per_segment is None:
            n_points_per_segment = 50
        x_mid = torch.rand(n_points_per_segment, 1) * (self.x_max - self.x_min) + self.x_min
        y_bottom = torch.full((n_points_per_segment, 1), self.y_min)
        y_top = torch.full((n_points_per_segment, 1), self.y_max)
        y_mid = torch.rand(n_points_per_segment, 1) * (self.y_max - self.y_min) + self.y_min
        x_left = torch.full((n_points_per_segment, 1), self.x_min)
        x_right = torch.full((n_points_per_segment, 1), self.x_max)
        bottom = torch.cat([x_mid, y_bottom], dim=1)
        top = torch.cat([x_mid, y_top], dim=1)
        left = torch.cat([x_left, y_mid], dim=1)
        right = torch.cat([x_right, y_mid], dim=1)
        points = torch.cat([bottom, top, left, right], dim=0)
        boundary_ids = torch.cat([
            torch.full((n_points_per_segment,), 0),   # bottom
            torch.full((n_points_per_segment,), 1),   # top
            torch.full((n_points_per_segment,), 2),   # left
            torch.full((n_points_per_segment,), 3),   # right
        ], dim=0)
        return points, boundary_ids
    # def sample_initial(self, n_points):
    #     """
    #     采样初始条件点 (t = t_min)，返回坐标张量。
    #     仅当 has_t 为 True 时可用。
    #     """
    #     if not self.has_t:
    #         raise ValueError("No time dimension, cannot sample initial condition.")
    #     points = []
    #     points.append(torch.rand(n_points, 1) * (self.x_max - self.x_min) + self.x_min)
    #     if self.has_y:
    #         points.append(torch.rand(n_points, 1) * (self.y_max - self.y_min) + self.y_min)
    #     points.append(torch.full((n_points, 1), self.t_min))
    #     return torch.cat(points, dim=1)
    # def sample_boundary_by_axis(self, axis, value, n_points):
    #     """
    #     在矩形域上采样 axis = value 的边界点。
    #     axis: 'x' 或 'y'（2D）或 't'（时间）
    #     """
    #     if axis == 'x':
    #         y = torch.rand(n_points, 1) * (self.y_max - self.y_min) + self.y_min
    #         x = torch.full((n_points, 1), value)
    #         return torch.cat([x, y], dim=1)
    #     elif axis == 'y':
    #         x = torch.rand(n_points, 1) * (self.x_max - self.x_min) + self.x_min
    #         y = torch.full((n_points, 1), value)
    #         return torch.cat([x, y], dim=1)
    #     else:
    #         raise ValueError(f"Unsupported axis: {axis}")
    def sample_boundary_by_axis(self, axis, value, n_points):
        """
        在矩形域上采样 axis = value 的边界点。
        axis: 'x' 或 'y'（若 has_y 为 True）或 't'（若 has_t 为 True）
        """
        if axis == 'x':
            if self.has_y:
                y = torch.rand(n_points, 1) * (self.y_max - self.y_min) + self.y_min
                return torch.cat([torch.full((n_points, 1), value), y], dim=1)
            else:
                # 一维空间（只有 x），若还有 t，则 x 固定，t 随机
                if self.has_t:
                    t = torch.rand(n_points, 1) * (self.t_max - self.t_min) + self.t_min
                    return torch.cat([torch.full((n_points, 1), value), t], dim=1)
                else:
                    return torch.full((n_points, 1), value)
        elif axis == 'y':
            if not self.has_y:
                raise ValueError("y 轴不存在，无法采样")
            x = torch.rand(n_points, 1) * (self.x_max - self.x_min) + self.x_min
            return torch.cat([x, torch.full((n_points, 1), value)], dim=1)
        elif axis == 't':
            if not self.has_t:
                raise ValueError("t 轴不存在，无法采样")
            # 采样 t = value 时刻的空间点（x 随机，若还有 y 则 y 也随机）
            x = torch.rand(n_points, 1) * (self.x_max - self.x_min) + self.x_min
            if self.has_y:
                y = torch.rand(n_points, 1) * (self.y_max - self.y_min) + self.y_min
                return torch.cat([x, y, torch.full((n_points, 1), value)], dim=1)
            else:
                return torch.cat([x, torch.full((n_points, 1), value)], dim=1)
        else:
            raise ValueError(f"不支持的轴: {axis}")
    def sample_initial(self, n_points):
        """采样初始条件点 (t = t_min)，返回坐标张量。仅当 has_t 为 True 时可用。"""
        if not self.has_t:
            # raise ValueError("No time dimension, cannot sample initial condition.")
            return torch.empty(n_points, 0)
        points = []
        points.append(torch.rand(n_points, 1) * (self.x_max - self.x_min) + self.x_min)
        if self.has_y:
            points.append(torch.rand(n_points, 1) * (self.y_max - self.y_min) + self.y_min)
        points.append(torch.full((n_points, 1), self.t_min))
        return torch.cat(points, dim=1)

def get_interior_points(n_points, x_range, y_range=None, t_range=None):
    sampler = DomainSampler(x_range, y_range, t_range)
    return sampler.sample_interior(n_points)
def get_boundary_points(n_points_per_segment, x_range, y_range=None, t_range=None):
    boundary_sampler = DomainSampler(x_range, y_range, t_range)
    return boundary_sampler.sample_boundary(n_points_per_segment)
def get_initial_points(n_points, x_range, y_range=None, t_range=(0, 1)):
    sampler = DomainSampler(x_range, y_range, t_range)
    return sampler.sample_initial(n_points)
