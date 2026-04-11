import pygame
from collections import defaultdict
import moderngl
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1280, 720

running = True
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Graphics")

ctx = moderngl.create_context()
prog = ctx.program(
    vertex_shader="""
        #version 330 core
        in vec2 in_vert;
        uniform vec2 resolution;
        void main() {
            vec2 ndc = (in_vert / resolution) * 2.0 - 1.0;
            ndc.y = -ndc.y;
            gl_Position = vec4(ndc, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330 core
        uniform vec3 color;
        out vec4 fragColor;
        void main() { fragColor = vec4(color, 1.0); }
    """,
)
prog['resolution'].value = (WIDTH, HEIGHT)

text_prog = ctx.program(
    vertex_shader="""
        #version 330 core
        in vec2 in_vert;
        in vec2 in_uv;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            uv = in_uv;
        }
    """,
    fragment_shader="""
        #version 330 core
        uniform sampler2D tex;
        in vec2 uv;
        out vec4 fragColor;
        void main() { fragColor = texture(tex, uv); }
    """,
)

font = pygame.font.SysFont('monospace', 16)

HINTS = [
    "H  toggle hints",
    "C  clear all",
    "P  toggle control points",
    "U  curve from selected",
    "D  delete",
    "LMB  add point",
    "RMB on point  draw line",
    "RMB on empty  box select",
    "Drag point  move",
    "ESC  quit",
]

def make_hint_overlay():
    line_h, pad = 22, 10
    rendered = [font.render(h, True, (220, 220, 220)) for h in HINTS]
    w = max(s.get_width() for s in rendered) + pad * 2
    th = len(rendered) * line_h + pad * 2
    surf = pygame.Surface((w, th), pygame.SRCALPHA)
    surf.fill((20, 20, 20, 200))
    for i, ts in enumerate(rendered):
        surf.blit(ts, (pad, pad + i * line_h))
    data = pygame.image.tostring(surf, 'RGBA', True)
    tex = ctx.texture((w, th), 4, data)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    x1 = -1.0 + w / WIDTH * 2
    y1 = 1.0 - th / HEIGHT * 2
    verts = np.array([
        -1.0, 1.0, 0.0, 1.0,
         x1,  1.0, 1.0, 1.0,
        -1.0,  y1, 0.0, 0.0,
         x1,   y1, 1.0, 0.0,
    ], dtype='f4')
    vbo = ctx.buffer(verts.tobytes())
    vao = ctx.vertex_array(text_prog, [(vbo, '2f 2f', 'in_vert', 'in_uv')])
    return tex, vao, vbo

hint_overlay = None

_draw_vbo = None
_draw_vao = None
_draw_capacity = 0

def draw(mode, verts, color, point_size=10, line_width=1):
    global _draw_vbo, _draw_vao, _draw_capacity
    if not verts: return
    data = np.array(verts, dtype='f4').tobytes()
    n_bytes = len(data)
    ctx.point_size = point_size
    ctx.line_width = line_width
    prog['color'].value = color
    if n_bytes > _draw_capacity:
        if _draw_vbo is not None:
            _draw_vao.release()
            _draw_vbo.release()
        _draw_capacity = n_bytes * 2
        _draw_vbo = ctx.buffer(reserve=_draw_capacity)
        _draw_vao = ctx.simple_vertex_array(prog, _draw_vbo, 'in_vert')
    _draw_vbo.write(data)
    _draw_vao.render(mode, vertices=len(verts) // 2)

class BaseCache:
    def __init__(self):
        self.vbo = None
        self.vao = None
        self.cap = 0
        self.n = 0

    def upload(self, data_list):
        if not data_list:
            self.n = 0
            return
        data = np.array(data_list, dtype='f4').tobytes()
        nb = len(data)
        if nb > self.cap:
            if self.vbo:
                self.vao.release()
                self.vbo.release()
            self.cap = nb * 2
            self.vbo = ctx.buffer(reserve=self.cap)
            self.vao = ctx.simple_vertex_array(prog, self.vbo, 'in_vert')
        self.vbo.write(data)
        self.n = len(data_list) // 2

    def render(self, mode, color, point_size=10, line_width=1):
        if not self.n: return
        ctx.point_size = point_size
        ctx.line_width = line_width
        prog['color'].value = color
        self.vao.render(mode, vertices=self.n)

pts_cache = BaseCache()
lns_cache = BaseCache()
crv_cache = BaseCache()
dirty = True

sel_pts_cache = BaseCache()
sel_lns_cache = BaseCache()
sel_crv_cache = BaseCache()
sel_dirty = False

def arc_verts(p1, ctrl, p3, segments=32):
    t = np.linspace(0, 1, segments + 1)
    mt = 1 - t
    x = mt*mt * p1[0] + 2*mt*t * ctrl[0] + t*t * p3[0]
    y = mt*mt * p1[1] + 2*mt*t * ctrl[1] + t*t * p3[1]
    return [v for pair in zip(x, y) for v in pair]

def strip_to_lines(verts):
    a = np.array(verts, dtype='f4').reshape(-1, 2)
    pairs = np.stack([a[:-1], a[1:]], axis=1)
    return pairs.reshape(-1).tolist()


_uid_counter = 0
def new_uid():
    global _uid_counter
    _uid_counter += 1
    return _uid_counter

all_points = {}
all_lines  = {}
all_curves = {}
points = defaultdict(list)
ctrl_points = defaultdict(list)

selected_point = None
selected_lines = []
selected_curves = []

all_selected_points = []
sel_points_map = {}
all_selected_lines = []
all_selected_curves = []

selection_in_progress = False
line_in_progress = False
curve_in_progress = False

dragging_point = False
dragging_selection = False
dragging_ctrl = False
dragging_ctrl_uid = None
ctrl_hovered_uid = None
show_ctrl_points = False
show_hints = False

line_start = None
line_start_uid = None
curve_start = None
curve_start_uid = None
curve_control = None

all_line_points = []
all_dragged_curves = []

mp_quadrant = 1


def _ctrl_quad(x, y):
    if x*2 < WIDTH and y*2 < HEIGHT: return 1
    elif x*2 >= WIDTH and y*2 < HEIGHT: return 2
    elif x*2 < WIDTH and y*2 >= HEIGHT: return 3
    else: return 4

def add_ctrl_point(curve_uid):
    ctrl = all_curves[curve_uid]['ctrl']
    quad = _ctrl_quad(ctrl[0], ctrl[1])
    ctrl_points[quad].append([ctrl[0], ctrl[1], quad, curve_uid])

def remove_ctrl_point(curve_uid):
    for q in ctrl_points.values():
        for entry in q:
            if entry[3] == curve_uid:
                q.remove(entry)
                return

def update_ctrl_point(curve_uid, x, y):
    for q in ctrl_points.values():
        for entry in q:
            if entry[3] == curve_uid:
                q.remove(entry)
                new_quad = _ctrl_quad(x, y)
                ctrl_points[new_quad].append([x, y, new_quad, curve_uid])
                return

def _other_uid(line_or_curve, this_uid):
    return line_or_curve['p2_uid'] if line_or_curve['p1_uid'] == this_uid else line_or_curve['p1_uid']

def _find_point(uid):
    return all_points.get(uid) or sel_points_map.get(uid)

def _remove_point_lines_curves(pt):
    for line_uid in list(pt['line_uids']):
        line = all_lines.pop(line_uid, None)
        if line is None:
            continue
        other = _find_point(_other_uid(line, pt['uid']))
        if other:
            other['line_uids'].discard(line_uid)

    for curve_uid in list(pt['curve_uids']):
        curve = all_curves.pop(curve_uid, None)
        if curve is None:
            continue
        other = _find_point(_other_uid(curve, pt['uid']))
        if other:
            other['curve_uids'].discard(curve_uid)
        remove_ctrl_point(curve_uid)


clock = pygame.time.Clock()

while running:

    ctx.clear(0, 0, 0, 1)

    mp = pygame.mouse.get_pos()
    if mp[0]*2 < WIDTH and mp[1]*2 < HEIGHT:
        mp_quadrant = 1
    elif mp[0]*2 >= WIDTH and mp[1]*2 < HEIGHT:
        mp_quadrant = 2
    elif mp[0]*2 < WIDTH and mp[1]*2 >= HEIGHT:
        mp_quadrant = 3
    elif mp[0]*2 >= WIDTH and mp[1]*2 >= HEIGHT:
        mp_quadrant = 4

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_c:
                all_points.clear()
                all_lines.clear()
                all_curves.clear()
                points.clear()
                ctrl_points.clear()
                dirty = True
                selected_point = None
                selected_lines = []
                selected_curves = []
                all_selected_points = []
                sel_points_map.clear()
                all_selected_lines = []
                all_selected_curves = []
                sel_dirty = True
                selection_in_progress = False
                line_in_progress = False
                curve_in_progress = False
                dragging_point = False
                line_start = None
                line_start_uid = None
                curve_start = None
                curve_start_uid = None
            elif event.key == pygame.K_h:
                show_hints = not show_hints
            elif event.key == pygame.K_p:
                show_ctrl_points = not show_ctrl_points
            elif event.key == pygame.K_u:
                if selected_point is not None and not curve_in_progress and not line_in_progress:
                    curve_in_progress = True
                    curve_start_uid = selected_point['uid']
                    curve_start = (selected_point['x'], selected_point['y'])
                    if selected_point['line_uids']:
                        any_line = all_lines[next(iter(selected_point['line_uids']))]
                        neighbor = all_points[_other_uid(any_line, selected_point['uid'])]
                        lx, ly = neighbor['x'], neighbor['y']
                        dx, dy = curve_start[0] - lx, curve_start[1] - ly
                        length = np.hypot(dx, dy) or 1
                        px, py = -dy / length, dx / length
                        curve_control = (curve_start[0] + px * 5, curve_start[1] + py * 5)
                    else:
                        curve_control = (curve_start[0] + 5, curve_start[1] + 5)
            elif event.key == pygame.K_d:
                if selected_point is not None:
                    points[selected_point['quad']].remove(selected_point)
                    del all_points[selected_point['uid']]
                    _remove_point_lines_curves(selected_point)
                    dirty = True
                    selected_point = None
                    selected_lines = []
                    selected_curves = []
                elif all_selected_lines or all_selected_points or all_selected_curves:
                    all_selected_lines.clear()
                    all_selected_points.clear()
                    sel_points_map.clear()
                    all_selected_curves.clear()
                    sel_dirty = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if ctrl_hovered_uid is not None:
                    dragging_ctrl = True
                    dragging_ctrl_uid = ctrl_hovered_uid
                elif mp not in points[mp_quadrant] and all_selected_points == [] and all_selected_lines == [] and all_selected_curves == [] and not line_in_progress and not dragging_point and selected_point is None:
                    uid = new_uid()
                    new_point = {'uid': uid, 'x': mp[0], 'y': mp[1], 'quad': mp_quadrant,
                                 'line_uids': set(), 'curve_uids': set()}
                    all_points[uid] = new_point
                    points[mp_quadrant].append(new_point)
                    dirty = True
                elif (all_selected_points or all_selected_lines or all_selected_curves) and not dragging_selection:
                    dragging_selection = True
                    dragging_selection_start = mp
                elif not dragging_point and selected_point is not None and not line_in_progress and not curve_in_progress:
                    dragging_point = True
                    pt = selected_point
                    selected_point = None
                    points[pt['quad']].remove(pt)
                    del all_points[pt['uid']]

                    all_line_points = []
                    for line_uid in list(pt['line_uids']):
                        line = all_lines.pop(line_uid)
                        other_uid = _other_uid(line, pt['uid'])
                        all_points[other_uid]['line_uids'].discard(line_uid)
                        all_line_points.append(other_uid)

                    all_dragged_curves = []
                    for curve_uid in list(pt['curve_uids']):
                        curve = all_curves.pop(curve_uid)
                        other_uid = _other_uid(curve, pt['uid'])
                        all_points[other_uid]['curve_uids'].discard(curve_uid)
                        remove_ctrl_point(curve_uid)
                        all_dragged_curves.append((curve['ctrl'], other_uid))

                    dirty = True

            elif event.button == 3:
                if selected_point is not None and not line_in_progress:
                    line_in_progress = True
                    line_start = (selected_point['x'], selected_point['y'])
                    line_start_uid = selected_point['uid']
                elif selected_point is None and not line_in_progress:
                    selection_in_progress = True
                    if all_selected_points:
                        for p in all_selected_points:
                            all_points[p['uid']] = p
                            points[p['quad']].append(p)
                        all_selected_points.clear()
                        sel_points_map.clear()
                    if all_selected_lines:
                        for line in all_selected_lines:
                            all_lines[line['uid']] = line
                            all_points[line['p1_uid']]['line_uids'].add(line['uid'])
                            all_points[line['p2_uid']]['line_uids'].add(line['uid'])
                        all_selected_lines.clear()
                    if all_selected_curves:
                        for curve in all_selected_curves:
                            all_curves[curve['uid']] = curve
                            all_points[curve['p1_uid']]['curve_uids'].add(curve['uid'])
                            all_points[curve['p2_uid']]['curve_uids'].add(curve['uid'])
                            add_ctrl_point(curve['uid'])
                        all_selected_curves.clear()
                    dirty = True
                    sel_dirty = True
                    selection_start = mp

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_ctrl:
                dragging_ctrl = False
                dragging_ctrl_uid = None
            elif dragging_point:
                uid = new_uid()
                new_point = {'uid': uid, 'x': mp[0], 'y': mp[1], 'quad': mp_quadrant,
                             'line_uids': set(), 'curve_uids': set()}
                all_points[uid] = new_point
                points[mp_quadrant].append(new_point)
                for other_uid in all_line_points:
                    line_uid = new_uid()
                    line = {'uid': line_uid, 'p1_uid': uid, 'p2_uid': other_uid}
                    all_lines[line_uid] = line
                    new_point['line_uids'].add(line_uid)
                    all_points[other_uid]['line_uids'].add(line_uid)
                for ctrl, other_uid in all_dragged_curves:
                    other = all_points[other_uid]
                    _av = arc_verts((mp[0], mp[1]), ctrl, (other['x'], other['y']))
                    curve_uid = new_uid()
                    c = {'uid': curve_uid, 'p1_uid': uid, 'ctrl': ctrl,
                         'p2_uid': other_uid, 'line_verts': strip_to_lines(_av)}
                    all_curves[curve_uid] = c
                    new_point['curve_uids'].add(curve_uid)
                    all_points[other_uid]['curve_uids'].add(curve_uid)
                    add_ctrl_point(curve_uid)
                dirty = True
            elif selection_in_progress:
                selection_in_progress = False
                x1, y1 = selection_start
                x2, y2 = mp
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                quad_points = []
                if x1*2 < WIDTH  and y1*2 < HEIGHT:  quad_points.extend(list(points[1]))
                if x2*2 >= WIDTH and y1*2 < HEIGHT:  quad_points.extend(list(points[2]))
                if x1*2 < WIDTH  and y2*2 >= HEIGHT: quad_points.extend(list(points[3]))
                if x2*2 >= WIDTH and y2*2 >= HEIGHT: quad_points.extend(list(points[4]))
                for p in quad_points:
                    if x1 <= p['x'] <= x2 and y1 <= p['y'] <= y2:
                        if p['uid'] not in sel_points_map:
                            all_selected_points.append(p)
                            sel_points_map[p['uid']] = p
                            points[p['quad']].remove(p)
                            del all_points[p['uid']]
                        for line_uid in list(p['line_uids']):
                            line = all_lines.get(line_uid)
                            if line is None:
                                continue
                            if line not in all_selected_lines:
                                all_selected_lines.append(line)
                                other = _find_point(_other_uid(line, p['uid']))
                                if other:
                                    other['line_uids'].discard(line_uid)
                                del all_lines[line_uid]
                        for curve_uid in list(p['curve_uids']):
                            curve = all_curves.get(curve_uid)
                            if curve is None:
                                continue
                            if curve not in all_selected_curves:
                                all_selected_curves.append(curve)
                                other = _find_point(_other_uid(curve, p['uid']))
                                if other:
                                    other['curve_uids'].discard(curve_uid)
                                remove_ctrl_point(curve_uid)
                                del all_curves[curve_uid]
                dirty = True
                sel_dirty = True
            elif dragging_selection:
                dragging_selection = False
                dx = mp[0] - dragging_selection_start[0]
                dy = mp[1] - dragging_selection_start[1]
                moved_uids = set()

                for p in all_selected_points:
                    p['x'] += dx
                    p['y'] += dy
                    if   p['x']*2 <  WIDTH and p['y']*2 <  HEIGHT: p['quad'] = 1
                    elif p['x']*2 >= WIDTH and p['y']*2 <  HEIGHT: p['quad'] = 2
                    elif p['x']*2 <  WIDTH and p['y']*2 >= HEIGHT: p['quad'] = 3
                    else:                                            p['quad'] = 4
                    moved_uids.add(p['uid'])

                dup_map = {}

                def _get_endpoint_uid(p_uid):
                    if p_uid in moved_uids:
                        return p_uid
                    if p_uid in dup_map:
                        return dup_map[p_uid]
                    p = all_points[p_uid]
                    dup_x = p['x'] + dx
                    dup_y = p['y'] + dy
                    if   dup_x*2 <  WIDTH and dup_y*2 <  HEIGHT: dup_quad = 1
                    elif dup_x*2 >= WIDTH and dup_y*2 <  HEIGHT: dup_quad = 2
                    elif dup_x*2 <  WIDTH and dup_y*2 >= HEIGHT: dup_quad = 3
                    else:                                          dup_quad = 4
                    uid2 = new_uid()
                    dup = {'uid': uid2, 'x': dup_x, 'y': dup_y, 'quad': dup_quad,
                           'line_uids': set(), 'curve_uids': set()}
                    all_selected_points.append(dup)
                    sel_points_map[uid2] = dup
                    dup_map[p_uid] = uid2
                    return uid2

                for line in all_selected_lines:
                    line['p1_uid'] = _get_endpoint_uid(line['p1_uid'])
                    line['p2_uid'] = _get_endpoint_uid(line['p2_uid'])

                for curve in all_selected_curves:
                    curve['p1_uid'] = _get_endpoint_uid(curve['p1_uid'])
                    curve['p2_uid'] = _get_endpoint_uid(curve['p2_uid'])
                    new_ctrl = (curve['ctrl'][0] + dx, curve['ctrl'][1] + dy)
                    p1 = _find_point(curve['p1_uid'])
                    p2 = _find_point(curve['p2_uid'])
                    _av = arc_verts((p1['x'], p1['y']), new_ctrl, (p2['x'], p2['y']))
                    curve['ctrl'] = new_ctrl
                    curve['line_verts'] = strip_to_lines(_av)

                sel_dirty = True
            elif line_in_progress:
                line_in_progress = False
                if selected_point is not None:
                    line_end = (selected_point['x'], selected_point['y'])
                    line_end_uid = selected_point['uid']
                else:
                    line_end = (mp[0], mp[1])
                    uid = new_uid()
                    new_pt = {'uid': uid, 'x': mp[0], 'y': mp[1], 'quad': mp_quadrant,
                              'line_uids': set(), 'curve_uids': set()}
                    all_points[uid] = new_pt
                    points[mp_quadrant].append(new_pt)
                    line_end_uid = uid
                if line_start != line_end:
                    line_uid = new_uid()
                    line = {'uid': line_uid, 'p1_uid': line_start_uid, 'p2_uid': line_end_uid}
                    all_lines[line_uid] = line
                    all_points[line_start_uid]['line_uids'].add(line_uid)
                    all_points[line_end_uid]['line_uids'].add(line_uid)
                dirty = True
                line_start = None
                line_start_uid = None
            elif curve_in_progress:
                curve_in_progress = False
                if selected_point is not None:
                    p3 = (selected_point['x'], selected_point['y'])
                    p3_uid = selected_point['uid']
                else:
                    p3 = (mp[0], mp[1])
                    uid = new_uid()
                    new_pt = {'uid': uid, 'x': mp[0], 'y': mp[1], 'quad': mp_quadrant,
                              'line_uids': set(), 'curve_uids': set()}
                    all_points[uid] = new_pt
                    points[mp_quadrant].append(new_pt)
                    p3_uid = uid
                if p3 != curve_start:
                    _av = arc_verts(curve_start, curve_control, p3)
                    curve_uid = new_uid()
                    c = {'uid': curve_uid, 'p1_uid': curve_start_uid, 'ctrl': curve_control,
                         'p2_uid': p3_uid, 'line_verts': strip_to_lines(_av)}
                    all_curves[curve_uid] = c
                    all_points[curve_start_uid]['curve_uids'].add(curve_uid)
                    all_points[p3_uid]['curve_uids'].add(curve_uid)
                    add_ctrl_point(curve_uid)
                dirty = True
                curve_start = None
                curve_start_uid = None
                curve_control = None
            dragging_point = False
            selected_point = None
            line_in_progress = False
            curve_in_progress = False
            line_start = None
            line_start_uid = None
            curve_start = None
            curve_start_uid = None
            curve_control = None

    if not dragging_point:
        selected_point = None
    selected_lines = []
    selected_curves = []

    if line_in_progress:
        draw(moderngl.LINES, [*line_start, *mp], (1, 0, 0))
    elif curve_in_progress:
        verts = arc_verts(curve_start, curve_control, mp)
        draw(moderngl.LINE_STRIP, verts, (1, 0, 0))
    elif selection_in_progress:
        x, y = selection_start
        draw(moderngl.LINE_LOOP, [x, y, mp[0], y, mp[0], mp[1], x, mp[1]], (1, 0, 0))
    elif dragging_point:
        draw(moderngl.POINTS, [mp[0], mp[1]], (1, 0, 0), point_size=10)
        if all_line_points:
            verts = []
            for neighbor_uid in all_line_points:
                n = all_points[neighbor_uid]
                verts += [mp[0], mp[1], n['x'], n['y']]
            draw(moderngl.LINES, verts, (1, 0, 0))
        dragged_verts = []
        for ctrl, other_uid in all_dragged_curves:
            other = all_points[other_uid]
            dragged_verts += strip_to_lines(arc_verts(mp, ctrl, (other['x'], other['y'])))
        draw(moderngl.LINES, dragged_verts, (1, 0, 0), line_width=2)

    ctrl_hovered_uid = None
    if not dragging_point and not dragging_ctrl and not selection_in_progress and not curve_in_progress and not line_in_progress:
        for entry in ctrl_points[mp_quadrant]:
            if mp[0] - 5 <= entry[0] <= mp[0] + 5 and mp[1] - 5 <= entry[1] <= mp[1] + 5:
                ctrl_hovered_uid = entry[3]
                break

    if dragging_ctrl:
        curve = all_curves[dragging_ctrl_uid]
        p1 = all_points[curve['p1_uid']]
        p2 = all_points[curve['p2_uid']]
        _av = arc_verts((p1['x'], p1['y']), mp, (p2['x'], p2['y']))
        curve['ctrl'] = mp
        curve['line_verts'] = strip_to_lines(_av)
        update_ctrl_point(dragging_ctrl_uid, mp[0], mp[1])
        dirty = True

    if not dragging_point and not dragging_ctrl and not selection_in_progress:
        for p in points[mp_quadrant]:
            if mp[0] - 5 <= p['x'] <= mp[0] + 5 and mp[1] - 5 <= p['y'] <= mp[1] + 5:
                selected_point = p
                selected_lines = [all_lines[uid] for uid in p['line_uids']]
                selected_curves = [all_curves[uid] for uid in p['curve_uids']]
                break

    if dirty:
        pts_cache.upload([c for p in all_points.values() for c in (p['x'], p['y'])])
        lns_cache.upload([c for l in all_lines.values()
                          for p1, p2 in [(all_points[l['p1_uid']], all_points[l['p2_uid']])]
                          for c in (p1['x'], p1['y'], p2['x'], p2['y'])])
        crv_cache.upload([c for curve in all_curves.values() for c in curve['line_verts']])
        dirty = False

    lns_cache.render(moderngl.LINES, (1, 1, 1), line_width=2)
    crv_cache.render(moderngl.LINES, (1, 1, 1), line_width=2)
    pts_cache.render(moderngl.POINTS, (1, 1, 1), point_size=10)

    if sel_dirty:
        def _sel_coords(uid):
            p = all_points.get(uid) or sel_points_map.get(uid)
            return (p['x'], p['y']) if p else (0, 0)
        sel_pts_cache.upload([c for p in all_selected_points for c in (p['x'], p['y'])])
        sel_lns_cache.upload([c for l in all_selected_lines
                               for c in (*_sel_coords(l['p1_uid']), *_sel_coords(l['p2_uid']))])
        sel_crv_cache.upload([c for curve in all_selected_curves for c in curve['line_verts']])
        sel_dirty = False

    sel_lns_cache.render(moderngl.LINES, (0, 1, 0), line_width=2)
    sel_crv_cache.render(moderngl.LINES, (0, 1, 0), line_width=2)
    sel_pts_cache.render(moderngl.POINTS, (0, 1, 0), point_size=10)

    hover_line_verts = []
    for line in selected_lines:
        p1 = all_points[line['p1_uid']]
        p2 = all_points[line['p2_uid']]
        hover_line_verts += [p1['x'], p1['y'], p2['x'], p2['y']]
    hover_curve_verts = [c for curve in selected_curves for c in curve['line_verts']]
    draw(moderngl.LINES, hover_line_verts, (0, 1, 0), line_width=2)
    draw(moderngl.LINES, hover_curve_verts, (0, 1, 0), line_width=2)
    draw(moderngl.POINTS, [selected_point['x'], selected_point['y']] if selected_point else [], (0, 1, 0), point_size=10)

    ctrl_arm_verts = []
    ctrl_pt_verts = []
    hovered_arm_verts = []
    hovered_pt_verts = []
    if show_ctrl_points:
        for uid, curve in all_curves.items():
            p1 = all_points[curve['p1_uid']]
            p2 = all_points[curve['p2_uid']]
            if uid == ctrl_hovered_uid:
                hovered_arm_verts += [p1['x'], p1['y'], *curve['ctrl'], p2['x'], p2['y'], *curve['ctrl']]
                hovered_pt_verts += [*curve['ctrl']]
            else:
                ctrl_arm_verts += [p1['x'], p1['y'], *curve['ctrl'], p2['x'], p2['y'], *curve['ctrl']]
                ctrl_pt_verts += [*curve['ctrl']]
    else:
        for curve in selected_curves:
            p1 = all_points[curve['p1_uid']]
            p2 = all_points[curve['p2_uid']]
            ctrl_arm_verts += [p1['x'], p1['y'], *curve['ctrl'], p2['x'], p2['y'], *curve['ctrl']]
            ctrl_pt_verts += [*curve['ctrl']]
        if ctrl_hovered_uid is not None:
            curve = all_curves[ctrl_hovered_uid]
            p1 = all_points[curve['p1_uid']]
            p2 = all_points[curve['p2_uid']]
            hovered_arm_verts += [p1['x'], p1['y'], *curve['ctrl'], p2['x'], p2['y'], *curve['ctrl']]
            hovered_pt_verts += [*curve['ctrl']]
    draw(moderngl.LINES, ctrl_arm_verts, (1, 1, 0.4), line_width=1)
    draw(moderngl.POINTS, ctrl_pt_verts, (1, 1, 0.4), point_size=8)
    draw(moderngl.LINES, hovered_arm_verts, (1, 0.5, 0), line_width=1)
    draw(moderngl.POINTS, hovered_pt_verts, (1, 0.5, 0), point_size=8)

    if show_hints:
        if hint_overlay is None:
            hint_overlay = make_hint_overlay()
        tex, vao, vbo = hint_overlay
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(0)
        text_prog['tex'].value = 0
        vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
