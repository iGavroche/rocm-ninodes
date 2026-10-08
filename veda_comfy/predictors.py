"""The released predictor bundles: where they live and how to get them.

Metadata only. This module performs no network access and reads no
environment: the predictor reaches `models/veda` either through ComfyUI's
own missing-model dialog, which the example workflows drive with
`properties.models`, or because the user put the file there. The node
turns a missing file into an error that says exactly where to get it (see
`nodes._predictor_path`).

That is a deliberate narrowing. An earlier version fetched the bundle
itself over plain HTTPS, resuming and verifying its digest. It worked,
but a custom node that opens connections and reads an auth token out of
the environment trips the Comfy Registry's security scan, and fetching
models is something ComfyUI already does for us. See
docs/features/packaging_release.md.
"""

from __future__ import annotations

import dataclasses

HUGGINGFACE = 'https://huggingface.co'


@dataclasses.dataclass(frozen=True)
class KnownPredictor:
    """A released bundle, pinned to an exact revision and hash.

    `sha256` and `size` are kept even though nothing here verifies them:
    they pin which artefact a release means, and let a user who placed the
    file by hand check it (`shasum -a 256`).

    Attributes:
        filename: Name in models/veda, and the name shown in the node.
        repo: Hugging Face repository id.
        revision: Exact commit, so a release is reproducible.
        sha256: Digest of the file at that revision.
        size: Size in bytes.
    """

    filename: str
    repo: str
    revision: str
    sha256: str
    size: int

    @property
    def url(self) -> str:
        """Direct download URL, as used in `properties.models`."""
        return (f'{HUGGINGFACE}/{self.repo}/resolve/{self.revision}/'
                f'{self.filename}')

    @property
    def page(self) -> str:
        return f'{HUGGINGFACE}/{self.repo}'

    @property
    def megabytes(self) -> int:
        """Decimal MB, matching what Hugging Face and the README quote."""
        return round(self.size / 10**6)


KNOWN_PREDICTORS = {
    p.filename: p for p in [
        KnownPredictor(
            filename='minimax_h3_t2va_veda_8nfe_600step_preview_fp8'
                     '.safetensors',
            repo='Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview',
            revision='9a1fd3a41b4a754a7886e64e82edbddf599fd1bd',
            sha256='2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd'
                   '76573a',
            size=275415648),
    ]
}

DEFAULT_PREDICTOR = next(iter(KNOWN_PREDICTORS))


def how_to_get(known: KnownPredictor, folder: str) -> str:
    """What to tell a user whose predictor is not on disk yet."""
    return (f'{known.filename} ({known.megabytes} MB) is not in {folder}.\n'
            'Either open a Veda template (Workflow -> Browse Templates -> '
            'Veda-on-ComfyUI) and let ComfyUI download it from the missing '
            'models dialog, or download it by hand and restart ComfyUI:\n'
            f'  {known.url}\n'
            f'and put it in {folder}\n'
            f'打分器 {known.filename}（{known.megabytes} MB）不在 {folder} 里。'
            '打开 Veda 模板（工作流 -> 浏览模板 -> Veda-on-ComfyUI），'
            '在缺失模型对话框里一键下载；或者从上面的地址手动下载，'
            '放进该目录后重启 ComfyUI。')
