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


points = defaultdict(list)
lines = defaultdict(set)
curves = defaultdict(set)
curve_lookup = {}  # (p1, ctrl, p2) and (p2, ctrl, p1) -> curve tuple
ctrl_points = defaultdict(list)  # quadrant -> [[x, y, quad, curve_idx], ...]

all_points = []
all_lines = []
all_curves = []

selected_point = None
selected_lines = []
selected_curves = []

all_selected_points = []
all_selected_lines = []
all_selected_curves = []

selection_in_progress = False
line_in_progress = False
curve_in_progress = False

dragging_point = False
dragging_selection = False
dragging_ctrl = False
dragging_ctrl_idx = None
dragging_ctrl_old_ctrl = None
ctrl_hovered_idx = None
show_ctrl_points = False
show_hints = False

line_start = None
curve_start = None
curve_control = None

mp_quadrant = 1
line_quad = 1
line_len = 0
    
def check_lines(point):
    return [((point[0], point[1]), line) for line in lines.get((point[0], point[1]), ())]

def check_curves(point):
    pt = (point[0], point[1])
    return [curve_lookup[(pt, ctrl, other)] for ctrl, other in curves.get(pt, ())]

def _ctrl_quad(x, y):
    if x*2 < WIDTH and y*2 < HEIGHT: return 1
    elif x*2 >= WIDTH and y*2 < HEIGHT: return 2
    elif x*2 < WIDTH and y*2 >= HEIGHT: return 3
    else: return 4

def add_ctrl_point(curve_idx):
    ctrl = all_curves[curve_idx][1]
    quad = _ctrl_quad(ctrl[0], ctrl[1])
    ctrl_points[quad].append([ctrl[0], ctrl[1], quad, curve_idx])

def remove_ctrl_point(curve_idx):
    for q in ctrl_points.values():
        for entry in q:
            if entry[3] == curve_idx:
                q.remove(entry)
                break
    for q in ctrl_points.values():
        for entry in q:
            if entry[3] > curve_idx:
                entry[3] -= 1

def update_ctrl_point(curve_idx, x, y):
    for q in ctrl_points.values():
        for entry in q:
            if entry[3] == curve_idx:
                q.remove(entry)
                new_quad = _ctrl_quad(x, y)
                ctrl_points[new_quad].append([x, y, new_quad, curve_idx])
                return

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
                points.clear()
                lines.clear()
                curves.clear()
                all_points.clear()
                all_lines.clear()
                all_curves.clear()
                curve_lookup.clear()
                ctrl_points.clear()
                dirty = True
                selected_point = None
                selected_lines = []
                selected_curves = []
                all_selected_points = []
                all_selected_lines = []
                all_selected_curves = []
                sel_dirty = True
                selection_in_progress = False
                line_in_progress = False
                curve_in_progress = False
                dragging_point = False
                line_start = None
                curve_start = None
            elif event.key == pygame.K_h:
                show_hints = not show_hints
            elif event.key == pygame.K_p:
                show_ctrl_points = not show_ctrl_points
            elif event.key == pygame.K_u:
                if selected_point is not None and not curve_in_progress and not line_in_progress:
                    curve_in_progress = True
                    curve_start = (selected_point[0], selected_point[1])
                    connected = list(lines.get(curve_start, []))
                    if connected:
                        lx, ly = connected[0]
                        dx, dy = curve_start[0] - lx, curve_start[1] - ly
                        length = np.hypot(dx, dy) or 1
                        px, py = -dy / length, dx / length
                        curve_control = (curve_start[0] + px * 5, curve_start[1] + py * 5)
                    else:
                        curve_control = (curve_start[0] + 5, curve_start[1] + 5)
            elif event.key == pygame.K_d:
                if selected_point != None:
                    points[selected_point[2]].remove(selected_point)
                    all_points.remove(selected_point)
                    if (selected_point[0], selected_point[1]) in lines:
                        for line in lines[(selected_point[0], selected_point[1])]:
                            lines[(line[0], line[1])].discard((selected_point[0], selected_point[1]))
                            try:
                                all_lines.remove(((selected_point[0], selected_point[1]), (line[0], line[1])))
                            except:
                                all_lines.remove(((line[0], line[1]), (selected_point[0], selected_point[1])))
                        del lines[(selected_point[0], selected_point[1])]
                    pt = (selected_point[0], selected_point[1])
                    for curve in check_curves(selected_point):
                        _cidx = all_curves.index(curve)
                        remove_ctrl_point(_cidx)
                        all_curves.remove(curve)
                        curve_lookup.pop((curve[0], curve[1], curve[2]), None)
                        curve_lookup.pop((curve[2], curve[1], curve[0]), None)
                        other = curve[2] if curve[0] == pt else curve[0]
                        curves[other].discard((curve[1], pt))
                    if pt in curves:
                        del curves[pt]
                    dirty = True
                    selected_point = None
                    selected_lines = []
                    selected_curves = []
                elif all_selected_lines != [] or all_selected_points != [] or all_selected_curves != []:
                    all_selected_lines.clear()
                    all_selected_points.clear()
                    all_selected_curves.clear()
                    sel_dirty = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if ctrl_hovered_idx is not None:
                    dragging_ctrl = True
                    dragging_ctrl_idx = ctrl_hovered_idx
                    dragging_ctrl_old_ctrl = all_curves[ctrl_hovered_idx][1]
                elif mp not in points[mp_quadrant] and all_selected_points == [] and all_selected_lines == [] and all_selected_curves == [] and line_in_progress == False and dragging_point == False and selected_point == None:
                    new_point = [mp[0], mp[1], mp_quadrant]
                    points[mp_quadrant].append(new_point)
                    all_points.append(new_point)
                    dirty = True
                elif (all_selected_points or all_selected_lines or all_selected_curves) and dragging_selection == False:
                    dragging_selection = True
                    dragging_selection_start = mp
                elif dragging_point == False and selected_point != None and not line_in_progress and not curve_in_progress:
                    dragging_point = True
                    line_in_progress = False
                    line_start = None
                    points[mp_quadrant].remove(selected_point)
                    all_points.remove(selected_point)
                    all_line_points = []
                    if (selected_point[0], selected_point[1]) in lines:
                        for line in lines[(selected_point[0], selected_point[1])]:
                            all_line_points.append(line)
                            lines[(line[0], line[1])].discard((selected_point[0], selected_point[1]))
                            try:
                                all_lines.remove(((selected_point[0], selected_point[1]), (line[0], line[1])))
                            except:
                                all_lines.remove(((line[0], line[1]), (selected_point[0], selected_point[1])))
                        del lines[(selected_point[0], selected_point[1])]
                    pt = (selected_point[0], selected_point[1])
                    all_dragged_curves = []
                    for curve in check_curves(selected_point):
                        _cidx = all_curves.index(curve)
                        remove_ctrl_point(_cidx)
                        all_curves.remove(curve)
                        curve_lookup.pop((curve[0], curve[1], curve[2]), None)
                        curve_lookup.pop((curve[2], curve[1], curve[0]), None)
                        other = curve[2] if curve[0] == pt else curve[0]
                        curves[other].discard((curve[1], pt))
                        all_dragged_curves.append((curve[1], other))
                    if pt in curves:
                        del curves[pt]
                    dirty = True
            elif event.button == 3:
                if selected_point != None and line_in_progress == False:
                    line_in_progress = True
                    line_start = (selected_point[0], selected_point[1])
                elif selected_point == None and line_in_progress == False:
                    selection_in_progress = True
                    if all_selected_points:
                        for p in all_selected_points:
                            points[p[2]].append(p)
                            all_points.append(p)
                        all_selected_points.clear()
                    if all_selected_lines:
                        for line in all_selected_lines:
                            lines[(line[0][0], line[0][1])].add(line[1])
                            lines[(line[1][0], line[1][1])].add(line[0])
                            all_lines.append((line[0], line[1]))
                        all_selected_lines.clear()
                    if all_selected_curves:
                        for curve in all_selected_curves:
                            all_curves.append(curve)
                            add_ctrl_point(len(all_curves) - 1)
                            curve_lookup[(curve[0], curve[1], curve[2])] = curve
                            curve_lookup[(curve[2], curve[1], curve[0])] = curve
                            curves[curve[0]].add((curve[1], curve[2]))
                            curves[curve[2]].add((curve[1], curve[0]))
                        all_selected_curves.clear()
                    dirty = True
                    sel_dirty = True
                    selection_start = mp
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_ctrl:
                dragging_ctrl = False
                curve = all_curves[dragging_ctrl_idx]
                new_ctrl = curve[1]
                old_ctrl = dragging_ctrl_old_ctrl
                curves[curve[0]].discard((old_ctrl, curve[2]))
                curves[curve[0]].add((new_ctrl, curve[2]))
                curves[curve[2]].discard((old_ctrl, curve[0]))
                curves[curve[2]].add((new_ctrl, curve[0]))
                dragging_ctrl_idx = None
                dragging_ctrl_old_ctrl = None
            elif dragging_point == True:
                new_point = [mp[0], mp[1], mp_quadrant]
                points[mp_quadrant].append(new_point)
                all_points.append(new_point)
                if all_line_points:
                    for line in all_line_points:
                        lines[(new_point[0], new_point[1])].add(line)
                        lines[(line[0], line[1])].add((new_point[0], new_point[1]))
                        all_lines.append(((new_point[0], new_point[1]), (line[0], line[1])))
                new_pt = (new_point[0], new_point[1])
                for ctrl, other in all_dragged_curves:
                    new_verts = arc_verts(new_pt, ctrl, other)
                    c = (new_pt, ctrl, other, new_verts, strip_to_lines(new_verts))
                    all_curves.append(c)
                    add_ctrl_point(len(all_curves) - 1)
                    curve_lookup[(new_pt, ctrl, other)] = c
                    curve_lookup[(other, ctrl, new_pt)] = c
                    curves[new_pt].add((ctrl, other))
                    curves[other].add((ctrl, new_pt))
                dirty = True
            elif selection_in_progress == True:
                selection_in_progress = False
                x1, y1 = selection_start
                x2, y2 = mp
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                mid_x, mid_y = WIDTH // 2, HEIGHT // 2
                quad_points = []
                if x1*2 < WIDTH and y1*2 < HEIGHT: quad_points.extend(points[1])
                if x2*2 >= WIDTH and y1*2 < HEIGHT: quad_points.extend(points[2])
                if x1*2 < WIDTH and y2*2 >= HEIGHT: quad_points.extend(points[3])
                if x2*2 >= WIDTH and y2*2 >= HEIGHT: quad_points.extend(points[4])
                for p in quad_points:
                    if x1 <= p[0] <= x2 and y1 <= p[1] <= y2:
                        if p not in all_selected_points:
                            all_selected_points.append(p)
                            points[p[2]].remove(p)
                            all_points.remove(p)
                        selected_lines = check_lines(p)
                        for line in selected_lines:
                            if line not in all_selected_lines and (line[1], line[0]) not in all_selected_lines:
                                all_selected_lines.append(line)
                                lines[(line[0][0], line[0][1])].discard(line[1])
                                lines[(line[1][0], line[1][1])].discard(line[0])
                                try:
                                    all_lines.remove((line[0], line[1]))
                                except:
                                    all_lines.remove((line[1], line[0]))
                        for curve in check_curves(p):
                            if curve not in all_selected_curves:
                                all_selected_curves.append(curve)
                                _cidx = all_curves.index(curve)
                                remove_ctrl_point(_cidx)
                                all_curves.remove(curve)
                                curve_lookup.pop((curve[0], curve[1], curve[2]), None)
                                curve_lookup.pop((curve[2], curve[1], curve[0]), None)
                                curves[curve[0]].discard((curve[1], curve[2]))
                                curves[curve[2]].discard((curve[1], curve[0]))
                dirty = True
                sel_dirty = True
            elif dragging_selection == True:
                dragging_selection = False
                dx = mp[0] - dragging_selection_start[0]
                dy = mp[1] - dragging_selection_start[1]
                if all_selected_points:
                    np_all_selected_points = np.array(all_selected_points)
                    np_all_selected_points[:, 0] += dx
                    np_all_selected_points[:, 1] += dy
                    all_selected_points = np_all_selected_points.tolist()
                for p in all_selected_points:
                    if p[0]*2 < WIDTH and p[1]*2 < HEIGHT: p[2] = 1
                    elif p[0]*2 >= WIDTH and p[1]*2 < HEIGHT: p[2] = 2
                    elif p[0]*2 < WIDTH and p[1]*2 >= HEIGHT: p[2] = 3
                    else: p[2] = 4
                for i, line in enumerate(all_selected_lines):
                    new_start = (line[0][0] + dx, line[0][1] + dy)
                    new_end = (line[1][0] + dx, line[1][1] + dy)
                    all_selected_lines[i] = (new_start, new_end)
                    for pt in (new_start, new_end):
                        if pt[0]*2 < WIDTH and pt[1]*2 < HEIGHT: quad = 1
                        elif pt[0]*2 >= WIDTH and pt[1]*2 < HEIGHT: quad = 2
                        elif pt[0]*2 < WIDTH and pt[1]*2 >= HEIGHT: quad = 3
                        else: quad = 4
                        if [pt[0], pt[1], quad] not in all_selected_points:
                            all_selected_points.append([pt[0], pt[1], quad])
                for i, curve in enumerate(all_selected_curves):
                    new_p1 = (curve[0][0] + dx, curve[0][1] + dy)
                    new_ctrl = (curve[1][0] + dx, curve[1][1] + dy)
                    new_p3 = (curve[2][0] + dx, curve[2][1] + dy)
                    _av = arc_verts(new_p1, new_ctrl, new_p3)
                    all_selected_curves[i] = (new_p1, new_ctrl, new_p3, _av, strip_to_lines(_av))
                    for pt in (new_p1, new_p3):
                        if pt[0]*2 < WIDTH and pt[1]*2 < HEIGHT: quad = 1
                        elif pt[0]*2 >= WIDTH and pt[1]*2 < HEIGHT: quad = 2
                        elif pt[0]*2 < WIDTH and pt[1]*2 >= HEIGHT: quad = 3
                        else: quad = 4
                        if [pt[0], pt[1], quad] not in all_selected_points:
                            all_selected_points.append([pt[0], pt[1], quad])
                sel_dirty = True
            elif line_in_progress == True:
                line_in_progress = False
                if selected_point is not None:
                    line_end = (selected_point[0], selected_point[1])
                else:
                    line_end = (mp[0], mp[1])
                    new_pt = [mp[0], mp[1], mp_quadrant]
                    points[mp_quadrant].append(new_pt)
                    all_points.append(new_pt)
                if line_start != line_end:
                    lines[(line_start[0], line_start[1])].add(line_end)
                    lines[(line_end[0], line_end[1])].add(line_start)
                    all_lines.append((line_start, line_end))
                dirty = True
                line_start = None
            elif curve_in_progress == True:
                curve_in_progress = False
                if selected_point is not None:
                    p3 = (selected_point[0], selected_point[1])
                else:
                    p3 = (mp[0], mp[1])
                    new_pt = [mp[0], mp[1], mp_quadrant]
                    points[mp_quadrant].append(new_pt)
                    all_points.append(new_pt)
                if p3 != curve_start:
                    _av = arc_verts(curve_start, curve_control, p3)
                    c = (curve_start, curve_control, p3, _av, strip_to_lines(_av))
                    all_curves.append(c)
                    add_ctrl_point(len(all_curves) - 1)
                    curve_lookup[(curve_start, curve_control, p3)] = c
                    curve_lookup[(p3, curve_control, curve_start)] = c
                    curves[curve_start].add((curve_control, p3))
                    curves[p3].add((curve_control, curve_start))
                dirty = True
                curve_start = None
                curve_control = None
            dragging_point = False
            selected_point = None
            line_in_progress = False
            curve_in_progress = False
            line_start = None
            curve_start = None
            curve_control = None
    
    if dragging_point == False:
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
        selected_point[0] = mp[0]
        selected_point[1] = mp[1]
        draw(moderngl.POINTS, [*mp], (1, 0, 0), point_size=10)
        if all_line_points:
            verts = [c for line in all_line_points for c in [*mp, *line]]
            draw(moderngl.LINES, verts, (1, 0, 0))
        dragged_verts = []
        for ctrl, other in all_dragged_curves:
            dragged_verts += strip_to_lines(arc_verts(mp, ctrl, other))
        draw(moderngl.LINES, dragged_verts, (1, 0, 0), line_width=2)

    ctrl_hovered_idx = None
    if not dragging_point and not dragging_ctrl and not selection_in_progress and not curve_in_progress and not line_in_progress:
        for entry in ctrl_points[mp_quadrant]:
            if mp[0] - 5 <= entry[0] <= mp[0] + 5 and mp[1] - 5 <= entry[1] <= mp[1] + 5:
                ctrl_hovered_idx = entry[3]
                break

    if dragging_ctrl:
        curve = all_curves[dragging_ctrl_idx]
        _av = arc_verts(curve[0], mp, curve[2])
        curve_lookup.pop((curve[0], curve[1], curve[2]), None)
        curve_lookup.pop((curve[2], curve[1], curve[0]), None)
        new_curve = (curve[0], mp, curve[2], _av, strip_to_lines(_av))
        all_curves[dragging_ctrl_idx] = new_curve
        update_ctrl_point(dragging_ctrl_idx, mp[0], mp[1])
        curve_lookup[(new_curve[0], new_curve[1], new_curve[2])] = new_curve
        curve_lookup[(new_curve[2], new_curve[1], new_curve[0])] = new_curve
        dirty = True

    if not dragging_point and not dragging_ctrl and not selection_in_progress:
        for p in points[mp_quadrant]:
            if mp[0] - 5 <= p[0] <= mp[0] + 5 and mp[1] - 5 <= p[1] <= mp[1] + 5:
                selected_point = p
                selected_lines = check_lines(selected_point)
                selected_curves = check_curves(selected_point)
                break
    
    if dirty:
        pts_cache.upload([c for p in all_points for c in p[:2]])
        lns_cache.upload([c for l in all_lines for c in [*l[0], *l[1]]])
        crv_cache.upload([c for curve in all_curves for c in curve[4]])
        dirty = False

    lns_cache.render(moderngl.LINES, (1, 1, 1), line_width=2)
    crv_cache.render(moderngl.LINES, (1, 1, 1), line_width=2)
    pts_cache.render(moderngl.POINTS, (1, 1, 1), point_size=10)

    if sel_dirty:
        sel_lns_cache.upload([c for l in all_selected_lines for c in [*l[0], *l[1]]])
        sel_crv_cache.upload([c for curve in all_selected_curves for c in curve[4]])
        sel_pts_cache.upload([c for p in all_selected_points for c in p[:2]])
        sel_dirty = False

    sel_lns_cache.render(moderngl.LINES, (0, 1, 0), line_width=2)
    sel_crv_cache.render(moderngl.LINES, (0, 1, 0), line_width=2)
    sel_pts_cache.render(moderngl.POINTS, (0, 1, 0), point_size=10)

    hover_line_verts = [c for line in selected_lines for c in [*line[0], *line[1]]]
    draw(moderngl.LINES, hover_line_verts, (0, 1, 0), line_width=2)
    hover_curve_verts = [c for curve in selected_curves for c in curve[4]]
    draw(moderngl.LINES, hover_curve_verts, (0, 1, 0), line_width=2)
    draw(moderngl.POINTS, selected_point[:2] if selected_point else [], (0, 1, 0), point_size=10)

    ctrl_arm_verts = []
    ctrl_pt_verts = []
    hovered_arm_verts = []
    hovered_pt_verts = []
    if show_ctrl_points:
        for i, curve in enumerate(all_curves):
            if i == ctrl_hovered_idx:
                hovered_arm_verts += [*curve[0], *curve[1], *curve[2], *curve[1]]
                hovered_pt_verts += [*curve[1]]
            else:
                ctrl_arm_verts += [*curve[0], *curve[1], *curve[2], *curve[1]]
                ctrl_pt_verts += [*curve[1]]
    else:
        for curve in selected_curves:
            ctrl_arm_verts += [*curve[0], *curve[1], *curve[2], *curve[1]]
            ctrl_pt_verts += [*curve[1]]
        if ctrl_hovered_idx is not None:
            curve = all_curves[ctrl_hovered_idx]
            hovered_arm_verts += [*curve[0], *curve[1], *curve[2], *curve[1]]
            hovered_pt_verts += [*curve[1]]
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