"""Builds example_workflows/ from ComfyUI's official MiniMax-H3 templates.

    python tools/make_example_workflows.py [--templates DIR]

The examples are the official templates (package
`comfyui-workflow-templates`, ComfyUI's template browser) with exactly
these changes, so they stay as close as possible to what users know:

  * a "Veda Sparse Attention (MiniMax H3)" node between the model switch
    (base / Turbo LoRA) and the guider, in its own group, with the
    predictor's download URL in `properties.models` (ComfyUI's missing-model
    dialog offers it);
  * the resolution set to 1344x768 (the trained 16:9 plan);
  * T2VA: the 8-step Turbo LoRA switched on (Veda's predictor was trained
    on an 8-step Turbo trajectory);
  * a note explaining Veda.

Run it again when the official templates change.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from veda_comfy import predictors  # noqa: E402

OUT = os.path.join(ROOT, 'example_workflows')
NODE_TYPE = 'VedaSparseAttention'
REGISTRY_ID = 'veda-sparse-attention'
# 1344x768 through ResolutionSelector (megapixels are MiB-based).
MEGAPIXELS = 0.98

NOTE = """## ⚡ Veda sparse attention

The **Veda Sparse Attention (MiniMax H3)** node (group "Veda") makes H3
compute only the attention tiles a learned predictor marks as important
(90% sparse by default): roughly **2-3x faster end to end** on long videos,
the speed-up growing with length.

* Placement: it is an attention override, so it sits on the MODEL wire
  after the model and any LoRA loaders, last before the guider.
* First run: ComfyUI offers the predictor (~275 MB) in the missing models
  dialog; it goes into `models/veda`. The kernels then compile once, and
  the node shows which one it uses.
* Compare: select the Veda node and press **Ctrl+B** (bypass) to render the
  same seed with full attention.
* Trained sizes: 1344x768, 768x1344, 768x768, 1024x768 at 5 / 10 / 14 s.
  Other sizes work with the plan of the nearest aspect ratio and duration
  (the node says so).
* Tuning (advanced inputs): generated / reference sparsity ("90%" or a
  tile count such as "24"), 0-based full-attention layers and steps,
  verbose diagnostics.
* The sparse kernel installs with the node; nothing else to set up.
"""


def _templates_dir(path: str | None) -> str:
    if path:
        return path
    spec = importlib.util.find_spec('comfyui_workflow_templates_json')
    if spec is None:
        sys.exit('pip install comfyui-workflow-templates, or pass --templates')
    return os.path.join(spec.submodule_search_locations[0], 'templates')


def _veda_node(node_id: int, pos, in_link: int, out_link: int) -> dict:
    known = predictors.KNOWN_PREDICTORS[predictors.DEFAULT_PREDICTOR]
    return {
        'id': node_id, 'type': NODE_TYPE, 'pos': list(pos),
        'size': [420, 110], 'flags': {}, 'order': 0, 'mode': 0,
        'title': '⚡ Veda Sparse Attention (MiniMax H3)',
        'inputs': [
            {'localized_name': 'model', 'name': 'model', 'type': 'MODEL',
             'link': in_link},
            {'localized_name': 'predictor', 'name': 'predictor',
             'type': 'COMBO', 'widget': {'name': 'predictor'},
             'link': None},
        ],
        'outputs': [{'localized_name': 'MODEL', 'name': 'MODEL',
                     'type': 'MODEL', 'links': [out_link]}],
        'properties': {
            'cnr_id': REGISTRY_ID, 'ver': _version(),
            'Node name for S&R': NODE_TYPE,
            # ComfyUI's missing-model dialog reads this, and it is now
            # the only thing that fetches the predictor for the user.
            'models': [{'name': known.filename, 'url': known.url,
                        'directory': 'veda'}],
        },
        # Positional, in schema order: predictor, generated_sparsity,
        # reference_sparsity, full_attention_layers, full_attention_steps,
        # verbose. One value per widget -- a stale extra entry does not
        # error, it shifts every later widget by one.
        'widgets_values': [known.filename, '90%', '90%', '', '', False],
    }


def _note(node_id: int, pos) -> dict:
    return {'id': node_id, 'type': 'MarkdownNote', 'pos': list(pos),
            'size': [520, 460], 'flags': {}, 'order': 0, 'mode': 0,
            'inputs': [], 'outputs': [], 'title': 'Veda',
            'properties': {}, 'widgets_values': [NOTE],
            'color': '#432', 'bgcolor': '#653'}


def _version() -> str:
    with open(os.path.join(ROOT, 'pyproject.toml'), encoding='utf-8') as f:
        for line in f:
            if line.startswith('version'):
                return line.split('=')[1].strip().strip('"')
    return '0.0.0'


def _group(group_id: int, pos) -> dict:
    return {'id': group_id, 'title': '⚡ Veda',
            'bounding': [pos[0] - 20, pos[1] - 60, 460, 190],
            'color': '#3f789e', 'font_size': 24, 'flags': {}}


def _set_resolution(nodes) -> None:
    for node in nodes:
        if node['type'] == 'ResolutionSelector':
            node['widgets_values'][1] = MEGAPIXELS


def _insert_dict_links(graph: dict, guider_type: str, new_id: int,
                       link_id: int, pos) -> None:
    """Subgraph format: links are dicts. Reroutes guider.model via Veda."""
    nodes = {n['id']: n for n in graph['nodes']}
    guider = next(n for n in graph['nodes'] if n['type'] == guider_type)
    model_in = next(i for i in guider['inputs'] if i['name'] == 'model')
    old = next(link for link in graph['links']
               if link['id'] == model_in['link'])
    source = nodes[old['origin_id']]
    # source -> veda (new link), veda -> guider (the old link, re-origined)
    graph['links'].append({'id': link_id, 'origin_id': source['id'],
                           'origin_slot': old['origin_slot'],
                           'target_id': new_id, 'target_slot': 0,
                           'type': 'MODEL'})
    source['outputs'][old['origin_slot']]['links'].remove(old['id'])
    source['outputs'][old['origin_slot']]['links'].append(link_id)
    old['origin_id'], old['origin_slot'] = new_id, 0
    graph['nodes'].append(_veda_node(new_id, pos, link_id, old['id']))


def _insert_list_links(graph: dict, guider_type: str, new_id: int,
                       link_id: int, pos) -> None:
    """Top-level format: links are [id, from, slot, to, slot, type]."""
    nodes = {n['id']: n for n in graph['nodes']}
    guider = next(n for n in graph['nodes'] if n['type'] == guider_type)
    model_in = next(i for i in guider['inputs'] if i['name'] == 'model')
    old = next(link for link in graph['links']
               if link[0] == model_in['link'])
    source = nodes[old[1]]
    graph['links'].append([link_id, source['id'], old[2], new_id, 0,
                           'MODEL'])
    source['outputs'][old[2]]['links'].remove(old[0])
    source['outputs'][old[2]]['links'].append(link_id)
    old[1], old[2] = new_id, 0
    graph['nodes'].append(_veda_node(new_id, pos, link_id, old[0]))


def make_t2va(template: dict) -> dict:
    wf = copy.deepcopy(template)
    sub = wf['definitions']['subgraphs'][0]
    node_id = max(wf['last_node_id'], sub['state']['lastNodeId']) + 1
    link_id = max(wf['last_link_id'], sub['state']['lastLinkId']) + 1
    top = min(g['bounding'][1] for g in sub['groups'])
    pos = (-400, top - 200)
    _insert_dict_links(sub, 'BasicGuider', node_id, link_id, pos)
    sub['groups'].append(_group(sub['state']['lastGroupId'] + 1, pos))
    sub['state'].update(lastNodeId=node_id + 1, lastLinkId=link_id,
                        lastGroupId=sub['state']['lastGroupId'] + 1)
    # Promoted subgraph widgets: switch the Turbo LoRA on (8 steps).
    instance = next(n for n in wf['nodes']
                    if n['type'] == sub['id'])
    names = [i['name'] for i in sub['inputs'] if i['name'] not in (
        'first_frame', 'last_frame')]
    index = names.index('value')
    if not isinstance(instance['widgets_values'][index], bool):
        sys.exit('T2V template changed: Turbo LoRA switch not found')
    instance['widgets_values'][index] = True
    _set_resolution(wf['nodes'])
    wf['nodes'].append(_note(node_id + 1, (instance['pos'][0],
                                           instance['pos'][1] - 520)))
    wf['last_node_id'], wf['last_link_id'] = node_id + 1, link_id
    return wf


def make_r2va(template: dict) -> dict:
    wf = copy.deepcopy(template)
    node_id, link_id = wf['last_node_id'] + 1, wf['last_link_id'] + 1
    top = min(g['bounding'][1] for g in wf['groups'])
    pos = (-770, top - 200)
    _insert_list_links(wf, 'BasicGuider', node_id, link_id, pos)
    wf['groups'].append(_group(max(g.get('id', 0) for g in wf['groups']) + 1,
                               pos))
    _set_resolution(wf['nodes'])
    wf['nodes'].append(_note(node_id + 1, (-1520, top - 560)))
    wf['last_node_id'], wf['last_link_id'] = node_id + 1, link_id
    return wf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--templates', help='official templates directory')
    args = parser.parse_args()
    folder = _templates_dir(args.templates)
    os.makedirs(OUT, exist_ok=True)
    for source, build, target in (
            ('video_minimax_h3_t2v.json', make_t2va,
             'Veda MiniMax H3 T2VA (text to audio-video).json'),
            ('video_minimax_h3_r2v.json', make_r2va,
             'Veda MiniMax H3 R2VA (reference to audio-video).json')):
        with open(os.path.join(folder, source), encoding='utf-8') as f:
            workflow = build(json.load(f))
        with open(os.path.join(OUT, target), 'w', encoding='utf-8') as f:
            json.dump(workflow, f, indent=1, ensure_ascii=False)
            f.write('\n')
        print('wrote', os.path.join('example_workflows', target))


if __name__ == '__main__':
    main()
