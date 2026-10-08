"""Released predictor metadata, and that nothing here reaches out.

The module is deliberately inert: no network, no environment. Those are
asserted, not just intended, because the reason they were removed (the
Comfy Registry's security scan) will not be visible to whoever adds the
next predictor.
"""

import ast
import os
import re

import pytest

from veda_comfy import predictors

_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def test_every_release_is_pinned():
    assert predictors.KNOWN_PREDICTORS
    for name, known in predictors.KNOWN_PREDICTORS.items():
        assert name == known.filename
        assert known.filename.endswith('.safetensors')
        assert _SHA256.fullmatch(known.sha256), known.sha256
        assert len(known.revision) == 40, 'pin a full commit, not a branch'
        assert known.size > 0


def test_default_is_one_of_them():
    assert predictors.DEFAULT_PREDICTOR in predictors.KNOWN_PREDICTORS


def test_urls_point_at_the_pinned_revision():
    known = predictors.KNOWN_PREDICTORS[predictors.DEFAULT_PREDICTOR]
    assert known.url.startswith('https://huggingface.co/')
    assert known.revision in known.url
    assert known.url.endswith(known.filename)
    assert known.page == f'https://huggingface.co/{known.repo}'
    assert known.megabytes == 275  # decimal MB, as the README quotes


def test_how_to_get_names_the_file_the_url_and_the_folder():
    known = predictors.KNOWN_PREDICTORS[predictors.DEFAULT_PREDICTOR]
    text = predictors.how_to_get(known, '/models/veda')
    for expected in (known.filename, known.url, '/models/veda',
                     'Browse Templates'):
        assert expected in text


@pytest.mark.parametrize('banned', [
    'os.environ', 'getenv', 'urlopen', 'socket', 'subprocess', 'requests',
])
def test_module_source_stays_inert(banned):
    """No network and no environment access, not even in a comment.

    The registry's scanner matches text, so a mention is a finding.
    """
    with open(predictors.__file__, encoding='utf-8') as f:
        assert banned not in f.read()


def test_module_imports_only_the_standard_library():
    with open(predictors.__file__, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split('.')[0])
    assert imported == {'__future__', 'dataclasses'}, imported


def test_the_package_has_no_downloader_left():
    """A stray import of the removed module would fail at load time."""
    root = os.path.dirname(predictors.__file__)
    assert not os.path.exists(os.path.join(root, 'downloads.py'))
