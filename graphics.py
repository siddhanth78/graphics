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

def draw(mode, verts, color, point_size=10, line_width=1):
    if not verts: return
    ctx.point_size = point_size
    ctx.line_width = line_width
    prog['color'].value = color
    vbo = ctx.buffer(np.array(verts, dtype='f4').tobytes())
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert')
    vao.render(mode)
    vao.release(); vbo.release()

def arc_verts(p1, ctrl, p3, segments=32):
    t = np.linspace(0, 1, segments + 1)
    mt = 1 - t
    x = mt*mt * p1[0] + 2*mt*t * ctrl[0] + t*t * p3[0]
    y = mt*mt * p1[1] + 2*mt*t * ctrl[1] + t*t * p3[1]
    return [v for pair in zip(x, y) for v in pair]

points = defaultdict(list)
lines = defaultdict(set)
curves = defaultdict(list)

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

line_start = None
curve_start = None
curve_control = None

mp_quadrant = 1
line_quad = 1
line_len = 0
    
def check_lines(point):
    return [((point[0], point[1]), line) for line in lines[(point[0], point[1])]]

def check_curves(point):
    pt = (point[0], point[1])
    return [c for c in all_curves if c[0] == pt or c[2] == pt]

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
                line_start = None
                curve_start = None
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
                        all_curves.remove(curve)
                        other = curve[2] if curve[0] == pt else curve[0]
                        curves[other] = [(c, e) for c, e in curves[other] if not (c == curve[1] and e == pt)]
                    if pt in curves:
                        del curves[pt]
                    selected_point = None
                    selected_lines = []
                    selected_curves = []
                elif all_selected_lines != [] or all_selected_points != [] or all_selected_curves != []:
                    all_selected_lines.clear()
                    all_selected_points.clear()
                    all_selected_curves.clear()

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
                        all_curves.remove(curve)
                        other = curve[2] if curve[0] == pt else curve[0]
                        curves[other] = [(c, e) for c, e in curves[other] if not (c == curve[1] and e == pt)]
                        all_dragged_curves.append((curve[1], other))
                    if pt in curves:
                        del curves[pt]
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
                            curves[curve[0]].append((curve[1], curve[2]))
                            curves[curve[2]].append((curve[1], curve[0]))
                        all_selected_curves.clear()
                    selection_start = mp
        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_ctrl:
                dragging_ctrl = False
                curve = all_curves[dragging_ctrl_idx]
                new_ctrl = curve[1]
                old_ctrl = dragging_ctrl_old_ctrl
                curves[curve[0]] = [(new_ctrl if c == old_ctrl and e == curve[2] else c, e) for c, e in curves[curve[0]]]
                curves[curve[2]] = [(new_ctrl if c == old_ctrl and e == curve[0] else c, e) for c, e in curves[curve[2]]]
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
                    all_curves.append((new_pt, ctrl, other, new_verts))
                    curves[new_pt].append((ctrl, other))
                    curves[other].append((ctrl, new_pt))
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
                                all_curves.remove(curve)
                                curves[curve[0]] = [(c, e) for c, e in curves[curve[0]] if not (c == curve[1] and e == curve[2])]
                                curves[curve[2]] = [(c, e) for c, e in curves[curve[2]] if not (c == curve[1] and e == curve[0])]
            elif dragging_selection == True:
                dragging_selection = False
                dx = mp[0] - dragging_selection_start[0]
                dy = mp[1] - dragging_selection_start[1]
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
                    all_selected_curves[i] = (new_p1, new_ctrl, new_p3, arc_verts(new_p1, new_ctrl, new_p3))
                    for pt in (new_p1, new_p3):
                        if pt[0]*2 < WIDTH and pt[1]*2 < HEIGHT: quad = 1
                        elif pt[0]*2 >= WIDTH and pt[1]*2 < HEIGHT: quad = 2
                        elif pt[0]*2 < WIDTH and pt[1]*2 >= HEIGHT: quad = 3
                        else: quad = 4
                        if [pt[0], pt[1], quad] not in all_selected_points:
                            all_selected_points.append([pt[0], pt[1], quad])
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
                    all_curves.append((curve_start, curve_control, p3, arc_verts(curve_start, curve_control, p3)))
                    curves[curve_start].append((curve_control, p3))
                    curves[p3].append((curve_control, curve_start))
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
        for ctrl, other in all_dragged_curves:
            draw(moderngl.LINE_STRIP, arc_verts(mp, ctrl, other), (1, 0, 0), line_width=2)

    ctrl_hovered_idx = None
    if not dragging_point and not dragging_ctrl and not selection_in_progress and not curve_in_progress and not line_in_progress:
        for i, curve in enumerate(all_curves):
            ctrl = curve[1]
            if mp[0] - 5 <= ctrl[0] <= mp[0] + 5 and mp[1] - 5 <= ctrl[1] <= mp[1] + 5:
                ctrl_hovered_idx = i
                break

    if dragging_ctrl:
        curve = all_curves[dragging_ctrl_idx]
        all_curves[dragging_ctrl_idx] = (curve[0], mp, curve[2], arc_verts(curve[0], mp, curve[2]))

    if not dragging_point and not dragging_ctrl and not selection_in_progress:
        for p in points[mp_quadrant]:
            if mp[0] - 5 <= p[0] <= mp[0] + 5 and mp[1] - 5 <= p[1] <= mp[1] + 5:
                selected_point = p
                selected_lines = check_lines(selected_point)
                selected_curves = check_curves(selected_point)
                break
    
    selected_set = set(selected_lines) | {(b, a) for a, b in selected_lines}

    verts = []
    for l in all_selected_lines: verts += [*l[0], *l[1]]
    for line in all_lines:
        if (line[0], line[1]) in selected_set: verts += [*line[0], *line[1]]
    draw(moderngl.LINES, verts, (0, 1, 0), line_width=2)

    verts = [c for line in all_lines if (line[0], line[1]) not in selected_set for c in [*line[0], *line[1]]]
    draw(moderngl.LINES, verts, (1, 1, 1), line_width=2)

    for curve in all_selected_curves:
        draw(moderngl.LINE_STRIP, curve[3], (0, 1, 0), line_width=2)
    selected_curve_keys = {(c[0], c[1], c[2]) for c in selected_curves}
    for i, curve in enumerate(all_curves):
        color = (0, 1, 0) if (curve[0], curve[1], curve[2]) in selected_curve_keys else (1, 1, 1)
        draw(moderngl.LINE_STRIP, curve[3], color, line_width=2)
        if show_ctrl_points or (curve[0], curve[1], curve[2]) in selected_curve_keys or i == ctrl_hovered_idx:
            ctrl_color = (1, 0.5, 0) if i == ctrl_hovered_idx else (0.4, 0.4, 1)
            draw(moderngl.LINES, [*curve[0], *curve[1], *curve[2], *curve[1]], ctrl_color, line_width=1)
            draw(moderngl.POINTS, [*curve[1]], ctrl_color, point_size=8)

    verts = [c for p in all_selected_points for c in p[:2]]
    if selected_point: verts += selected_point[:2]
    draw(moderngl.POINTS, verts, (0, 1, 0), point_size=10)

    verts = [c for p in all_points if p != selected_point for c in p[:2]]
    draw(moderngl.POINTS, verts, (1, 1, 1), point_size=10)

    pygame.display.flip()

pygame.quit()