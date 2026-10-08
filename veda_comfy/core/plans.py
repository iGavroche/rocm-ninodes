"""Tile plans: which tile shape every (layer, head) uses on one geometry.

A plan is searched offline for one target token grid (T, H, W) and ships
inside the predictor bundle. Plans are only exact for the grid they were
searched on; any other grid can still use a plan (tile shapes pad any grid),
it is just outside what the predictor was trained on. `PlanTable.select`
therefore says how it chose, so the UI can tell the user.
"""

from __future__ import annotations

import dataclasses
import math

import torch

from . import tiling


@dataclasses.dataclass(frozen=True)
class HeadGroup:
    """Heads of one layer that share a tile shape."""

    shape: tiling.TileShape
    heads: torch.Tensor  # [H'] int64


@dataclasses.dataclass
class TilePlan:
    """Per-(layer, head) tile shapes for one geometry.

    Attributes:
        name: Geometry name, e.g. '16x9_t37' (aspect and latent frames).
        grid: Target token grid (T, H, W) the plan was searched on.
        shapes: Distinct shapes referenced by `head_shape`.
        head_shape: [num_layers][num_heads] index into `shapes`.
    """

    name: str
    grid: tuple[int, int, int]
    shapes: list[tiling.TileShape]
    head_shape: list[list[int]]

    def __post_init__(self):
        self._groups: dict[tuple[int, str], list[HeadGroup]] = {}

    @property
    def num_layers(self) -> int:
        return len(self.head_shape)

    @property
    def num_heads(self) -> int:
        return len(self.head_shape[0]) if self.head_shape else 0

    def head_groups(self, layer: int,
                    device: torch.device | str) -> list[HeadGroup]:
        """Groups of heads sharing a shape (cached per layer and device)."""
        key = (layer, str(device))
        if key not in self._groups:
            row = torch.tensor(self.head_shape[layer])
            self._groups[key] = [
                HeadGroup(self.shapes[i],
                          torch.nonzero(row == i).view(-1).to(device))
                for i in sorted(set(self.head_shape[layer]))
            ]
        return self._groups[key]

    def transposed(self) -> TilePlan:
        """The H<->W mirrored plan (for portrait / landscape fallback)."""
        t, h, w = self.grid
        return TilePlan(self.name + '_T', (t, w, h),
                        [s.transposed() for s in self.shapes],
                        [list(r) for r in self.head_shape])

    @classmethod
    def from_json(cls, data: dict) -> TilePlan:
        return cls(name=data['geometry'], grid=tuple(data['grid']),
                   shapes=[tiling.TileShape.parse(s) for s in data['shapes']],
                   head_shape=[list(r) for r in data['head_shape']])


@dataclasses.dataclass(frozen=True)
class PlanChoice:
    """The plan picked for a grid and how it was picked.

    Attributes:
        plan: The plan to use.
        exact: The plan was searched on exactly this grid.
        how: Human-readable description, shown to the user.
    """

    plan: TilePlan
    exact: bool
    how: str


FPS = 24


def frames_from_latent_t(latent_t: int) -> int:
    """H3's video frame count for a latent length (17n + 5 frames)."""
    return 17 * ((latent_t - 2) // 5) + 5 if latent_t >= 2 else 1


def describe_grid(grid) -> str:
    """'1344x768 · 5.2 s' for a token grid (T, H, W)."""
    seconds = frames_from_latent_t(grid[0]) / FPS
    return f'{grid[2] * 32}x{grid[1] * 32} · {seconds:.1f} s'


def _aspect(grid) -> float:
    return grid[2] / grid[1]


class PlanTable:
    """All plans of a bundle; picks the plan for a target grid."""

    def __init__(self, plans: list[TilePlan]):
        if not plans:
            raise ValueError('a predictor bundle needs at least one plan')
        self.plans = {p.name: p for p in plans}
        self._choices: dict[tuple[int, int, int], PlanChoice] = {}

    def select(self, grid: tuple[int, int, int]) -> PlanChoice:
        """Plan for a target token grid (T, H, W): nearest aspect ratio,
        then nearest duration (latent frames), then least padding.

        Plans of the H<->W transposed geometry are candidates too (they
        only win when no plan has a closer aspect ratio). Never fails: a
        plan of another size still tiles any grid, it is just outside what
        the predictor was trained on, which `exact` and `how` report.
        """
        grid = tuple(int(g) for g in grid)
        if grid in self._choices:
            return self._choices[grid]
        candidates = list(self.plans.values())
        candidates += [p.transposed() for p in candidates]
        target = math.log(_aspect(grid))

        def cost(p):
            padding = sum(s.num_tiles(grid) for s in p.shapes)
            return (round(abs(math.log(_aspect(p.grid)) - target), 3),
                    abs(p.grid[0] - grid[0]), padding)

        plan = min(candidates, key=cost)
        exact = plan.grid == grid
        if exact:
            how = f'trained for this size ({describe_grid(grid)})'
        else:
            how = (f'nearest trained size: {describe_grid(plan.grid)} '
                   f'(this video: {describe_grid(grid)})')
        choice = PlanChoice(plan, exact, how)
        self._choices[grid] = choice
        return choice

    def summary(self) -> str:
        """e.g. '16:9, 9:16, 1:1, 4:3 x 5.2 / 10.1 / 14.4 s'."""
        aspects, lengths = [], set()
        for name in sorted(self.plans):
            aspect, _, t = name.partition('_t')
            label = aspect.replace('x', ':')
            if label not in aspects:
                aspects.append(label)
            if t.isdigit():
                lengths.add(int(t))
        seconds = ' / '.join(f'{frames_from_latent_t(t) / FPS:.1f}'
                             for t in sorted(lengths))
        return f'{", ".join(aspects)} x {seconds} s'
