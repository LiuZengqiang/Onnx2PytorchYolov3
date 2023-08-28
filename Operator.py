import torch
import torch.nn as nn


class Split(nn.Module):
    def __init__(self, axis: int, split: list[int]) -> None:
        super(Split, self).__init__()
        self.axis = axis
        self.split = split

    def forward(self, x: torch.Tensor):
        x = x.split(split_size=self.split, dim=self.axis)
        return x


class Pow(nn.Module):
    def __init__(self) -> None:
        super(Pow, self).__init__()
        self.Y = 1.0

    def forward(self, x: torch.Tensor):
        x = x.pow(self.Y)
        return x


class Floor(nn.Module):
    def __init__(self) -> None:
        super(Floor, self).__init__()
        self.x = 0

    def forward(self):
        return torch.floor(self.x)


class Slice(nn.Module):
    def __init__(self) -> None:
        super(Slice, self).__init__()
        self.starts = 0
        self.ends = 0
        self.axes = 0

    def forward(self, x: torch.Tensor):
        indexes = []
        for i in range(self.starts, self.ends):
            indexes.append(i)
        x = x.index_select(dim=int(self.axes), index=torch.tensor(indexes))
        return x


class Unsqueeze(nn.Module):
    def __init__(self, axes: int = 0) -> None:
        super(Unsqueeze, self).__init__()
        self.axes = axes

    def forward(self, x):
        if (type(x) != torch.Tensor):
            x = torch.tensor(x)
        if x.dim() <= 0:
            x = x.reshape([1])
        return x


class Resize(nn.Module):
    def __init__(self, coord_trans_mode: str = 'asymmetric', cubic_coeff_a: float = -0.75, mode: str = 'nearest', nearest_mode: str = 'floor') -> None:
        super(Resize, self).__init__()
        self.coord_mode = coord_trans_mode
        self.cubic_coeff_a = cubic_coeff_a
        self.mode = mode
        self.nearest_mode = nearest_mode

    def forward(self, x: torch.Tensor, size: torch.Tensor):
        x = nn.functional.interpolate(x, list(size)[2:4], mode=self.mode)
        return x


class Mul(nn.Module):
    def __init__(self) -> None:
        super(Mul, self).__init__()
        self.factor = 1.0

    def forward(self, x0, x1=None):
        if (x1 == None):
            return torch.mul(x0, self.factor)
        else:
            return torch.mul(x0, x1)


class Add(nn.Module):
    def __init__(self) -> None:
        super(Add, self).__init__()
        self.factor = 0.0

    def forward(self, x0, x1=None):
        if (x1 == None):
            return torch.add(x0, self.factor)
        else:
            return torch.add(x0, x1)


class Concat(nn.Module):
    def __init__(self, dim: int = 0) -> None:
        super(Concat, self).__init__()
        self.dim: int = dim

    def forward(self, x: list[torch.Tensor]):
        return torch.cat(x, dim=self.dim)


class Shape(nn.Module):
    def __init__(self) -> None:
        super(Shape, self).__init__()

    def forward(self, x: torch.Tensor):
        return torch.tensor(x.shape)


class Cast(nn.Module):
    def __init__(self, to: int = 7) -> None:
        super(Cast, self).__init__()
        self.to = to

    def forward(self, x: torch.Tensor):
        if (self.to == 7):
            return x.int()
        elif (self.to == 1):
            return x.float()
        else:
            return x


class Reshape(nn.Module):
    def __init__(self) -> None:
        super(Reshape, self).__init__()
        self.shape = None

    def forward(self, x: torch.Tensor):
        return x.reshape(list(self.shape))


class Transpose(nn.Module):
    def __init__(self, perm: list[int]) -> None:
        super(Transpose, self).__init__()
        self.perm = perm

    def forward(self, x: torch.Tensor):
        ori_order = []
        for i in range(x.dim()):
            ori_order.append(i)
        index = 0
        while index < len(ori_order):
            if (ori_order[index] == self.perm[index]):
                index += 1
            else:
                dim_0 = ori_order[index]
                dim_1 = self.perm.index(dim_0)
                x = x.transpose(index, dim_1)
                ori_order[index] = dim_1
                ori_order[dim_1] = dim_0
        return x
