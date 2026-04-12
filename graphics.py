import pygame
import moderngl
import numpy as np

pygame.init()

WIDTH, HEIGHT = 1280, 720
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
    "D  delete / deselect",
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

# ── GPU helpers ───────────────────────────────────────────────────────────────

_draw_vbo = None
_draw_vao = None
_draw_cap  = 0

def draw(mode, data, color, point_size=10, line_width=1):
    global _draw_vbo, _draw_vao, _draw_cap
    if data is None: return
    arr = np.asarray(data, dtype=np.float32).ravel()
    if len(arr) == 0: return
    raw = arr.tobytes()
    nb  = len(raw)
    ctx.point_size = point_size
    ctx.line_width = line_width
    prog['color'].value = color
    if nb > _draw_cap:
        if _draw_vbo:
            _draw_vao.release(); _draw_vbo.release()
        _draw_cap = nb * 2
        _draw_vbo = ctx.buffer(reserve=_draw_cap)
        _draw_vao = ctx.simple_vertex_array(prog, _draw_vbo, 'in_vert')
    _draw_vbo.write(raw)
    _draw_vao.render(mode, vertices=nb // 8)

class BaseCache:
    def __init__(self):
        self.vbo = None; self.vao = None; self.cap = 0; self.n = 0

    def upload(self, data):
        arr = np.asarray(data, dtype=np.float32).ravel()
        if len(arr) == 0:
            self.n = 0; return
        raw = arr.tobytes(); nb = len(raw)
        if nb > self.cap:
            if self.vbo: self.vao.release(); self.vbo.release()
            self.cap = nb * 2
            self.vbo = ctx.buffer(reserve=self.cap)
            self.vao = ctx.simple_vertex_array(prog, self.vbo, 'in_vert')
        self.vbo.write(raw)
        self.n = nb // 8

    def render(self, mode, color, point_size=10, line_width=1):
        if not self.n: return
        ctx.point_size = point_size
        ctx.line_width = line_width
        prog['color'].value = color
        self.vao.render(mode, vertices=self.n)

pts_cache     = BaseCache()
lns_cache     = BaseCache()   # holds lines + curves combined
sel_pts_cache = BaseCache()
sel_lns_cache = BaseCache()   # holds selected lines + curves combined
dirty     = True
sel_dirty = True

# ── Data layout ───────────────────────────────────────────────────────────────
#
#  pts        uint32  (N,)       packed: bits 0-10 = x, bits 11-20 = y
#  pt_lines   int16   (N, MCONN) indices of connected lines, -1 = empty
#  pt_curves  int16   (N, MCONN) indices of connected curves, -1 = empty
#  pt_nl      int8    (N,)       line count per point
#  pt_nc      int8    (N,)       curve count per point
#  pt_sel     bool    (N,)       selection mask
#
#  lns        int16   (N, 2)     [p1_idx, p2_idx]
#  ln_sel     bool    (N,)
#
#  crvs       int16   (N, 4)     [p1_idx, p2_idx, ctrl_x, ctrl_y]
#  crv_verts  float32 (N, 128)   baked line-pair verts (SEG*4 floats), direct GPU feed
#  crv_sel    bool    (N,)

MCONN  = 16
SEG    = 32
CVERTS = SEG * 4   # 128 floats: SEG line-pairs × 2 endpoints × 2 coords

def _grow(arr, fill=0):
    sh  = (len(arr) * 2, *arr.shape[1:])
    new = np.full(sh, fill, dtype=arr.dtype)
    new[:len(arr)] = arr
    return new

pts       = np.zeros(64, dtype=np.uint32)
pt_lines  = np.full((64, MCONN), -1, dtype=np.int32)
pt_curves = np.full((64, MCONN), -1, dtype=np.int32)
pt_nl     = np.zeros(64, dtype=np.int8)
pt_nc     = np.zeros(64, dtype=np.int8)
pt_sel    = np.zeros(64, dtype=bool)
pt_n      = 0

lns    = np.zeros((64, 2), dtype=np.int32)
ln_sel = np.zeros(64, dtype=bool)
ln_n   = 0

crvs      = np.zeros((64, 4), dtype=np.int32)
crv_verts = np.zeros((64, CVERTS), dtype=np.float32)
crv_sel   = np.zeros(64, dtype=bool)
crv_n     = 0

# ── Pack / unpack ─────────────────────────────────────────────────────────────

def pack_pt(x, y):
    return np.uint32(max(0, min(2047, int(round(x)))) |
                    (max(0, min(1023, int(round(y)))) << 11))

def unpack_pt(p):
    iv = int(p)
    return float(iv & 0x7FF), float((iv >> 11) & 0x3FF)

# ── Adjacency helpers ─────────────────────────────────────────────────────────

def _adj_add(adj, cnt, pi, idx):
    n = int(cnt[pi])
    if n < MCONN:
        adj[pi, n] = idx
        cnt[pi] = n + 1

def _adj_remove(adj, cnt, pi, idx):
    n = int(cnt[pi])
    for k in range(n):
        if adj[pi, k] == idx:
            adj[pi, k] = adj[pi, n - 1]
            adj[pi, n - 1] = -1
            cnt[pi] = n - 1
            return

def _adj_replace(adj, cnt, pi, old, new_val):
    n = int(cnt[pi])
    for k in range(n):
        if adj[pi, k] == old:
            adj[pi, k] = new_val
            return

# ── Geometry ──────────────────────────────────────────────────────────────────

def arc_verts(p1, ctrl, p3):
    t  = np.linspace(0, 1, SEG + 1, dtype=np.float32)
    mt = 1 - t
    x  = mt*mt*p1[0]   + 2*mt*t*ctrl[0] + t*t*p3[0]
    y  = mt*mt*p1[1]   + 2*mt*t*ctrl[1] + t*t*p3[1]
    return np.stack([x, y], axis=1)   # (SEG+1, 2)

def strip_to_lines(strip):
    return np.stack([strip[:-1], strip[1:]], axis=1).reshape(-1, 2)  # (SEG*2, 2)

def _bake_curve(ci):
    p1i  = int(crvs[ci, 0]); p2i = int(crvs[ci, 1])
    ctrl = (float(crvs[ci, 2]), float(crvs[ci, 3]))
    x1, y1 = unpack_pt(pts[p1i])
    x2, y2 = unpack_pt(pts[p2i])
    crv_verts[ci] = strip_to_lines(arc_verts((x1, y1), ctrl, (x2, y2))).ravel()

# ── CRUD ──────────────────────────────────────────────────────────────────────

def add_point(x, y):
    global pts, pt_lines, pt_curves, pt_nl, pt_nc, pt_sel, pt_n
    if pt_n >= len(pts):
        pts       = _grow(pts,       0)
        pt_lines  = _grow(pt_lines,  -1)
        pt_curves = _grow(pt_curves, -1)
        pt_nl     = _grow(pt_nl,     0)
        pt_nc     = _grow(pt_nc,     0)
        pt_sel    = _grow(pt_sel,    False)
    i = pt_n
    pts[i] = pack_pt(x, y)
    pt_lines[i, :]  = -1
    pt_curves[i, :] = -1
    pt_nl[i] = 0; pt_nc[i] = 0; pt_sel[i] = False
    pt_n += 1
    return i

def move_point(i, x, y):
    pts[i] = pack_pt(x, y)
    for k in range(int(pt_nc[i])):
        _bake_curve(int(pt_curves[i, k]))

def _swap_del_pt(i):
    global pt_n
    last = pt_n - 1
    if i != last:
        pts[i]       = pts[last]
        pt_lines[i]  = pt_lines[last]
        pt_curves[i] = pt_curves[last]
        pt_nl[i]     = pt_nl[last]
        pt_nc[i]     = pt_nc[last]
        pt_sel[i]    = pt_sel[last]
        for k in range(int(pt_nl[i])):
            li = int(pt_lines[i, k])
            if lns[li, 0] == last: lns[li, 0] = i
            if lns[li, 1] == last: lns[li, 1] = i
        for k in range(int(pt_nc[i])):
            ci = int(pt_curves[i, k])
            if crvs[ci, 0] == last: crvs[ci, 0] = i
            if crvs[ci, 1] == last: crvs[ci, 1] = i
    pt_n -= 1

def del_point(i):
    line_list  = [int(pt_lines[i, k])  for k in range(int(pt_nl[i]))]
    curve_list = [int(pt_curves[i, k]) for k in range(int(pt_nc[i]))]
    for j in range(len(line_list)):
        li       = line_list[j]
        old_last = ln_n - 1
        _del_line(li, skip=i)
        if li != old_last:
            for k in range(j + 1, len(line_list)):
                if line_list[k] == old_last: line_list[k] = li
    for j in range(len(curve_list)):
        ci       = curve_list[j]
        old_last = crv_n - 1
        _del_curve(ci, skip=i)
        if ci != old_last:
            for k in range(j + 1, len(curve_list)):
                if curve_list[k] == old_last: curve_list[k] = ci
    _swap_del_pt(i)

def add_line(p1, p2):
    global lns, ln_sel, ln_n
    if ln_n >= len(lns):
        lns    = _grow(lns,    0)
        ln_sel = _grow(ln_sel, False)
    li = ln_n
    lns[li] = [p1, p2]; ln_sel[li] = False
    _adj_add(pt_lines, pt_nl, p1, li)
    _adj_add(pt_lines, pt_nl, p2, li)
    ln_n += 1
    return li

def _swap_del_ln(li):
    global ln_n
    last = ln_n - 1
    if li != last:
        lns[li]    = lns[last]
        ln_sel[li] = ln_sel[last]
        lp1, lp2 = int(lns[li, 0]), int(lns[li, 1])
        _adj_replace(pt_lines, pt_nl, lp1, last, li)
        _adj_replace(pt_lines, pt_nl, lp2, last, li)
    ln_n -= 1

def _del_line(li, skip=-1):
    p1, p2 = int(lns[li, 0]), int(lns[li, 1])
    if p1 != skip: _adj_remove(pt_lines, pt_nl, p1, li)
    if p2 != skip: _adj_remove(pt_lines, pt_nl, p2, li)
    _swap_del_ln(li)

def add_curve(p1, p2, ctrl):
    global crvs, crv_verts, crv_sel, crv_n
    if crv_n >= len(crvs):
        crvs      = _grow(crvs,      0)
        crv_verts = _grow(crv_verts, 0.0)
        crv_sel   = _grow(crv_sel,   False)
    ci = crv_n
    crvs[ci] = [p1, p2, int(round(ctrl[0])), int(round(ctrl[1]))]
    crv_sel[ci] = False
    _adj_add(pt_curves, pt_nc, p1, ci)
    _adj_add(pt_curves, pt_nc, p2, ci)
    _bake_curve(ci)
    crv_n += 1
    return ci

def _swap_del_crv(ci):
    global crv_n
    last = crv_n - 1
    if ci != last:
        crvs[ci]      = crvs[last]
        crv_verts[ci] = crv_verts[last]
        crv_sel[ci]   = crv_sel[last]
        cp1, cp2 = int(crvs[ci, 0]), int(crvs[ci, 1])
        _adj_replace(pt_curves, pt_nc, cp1, last, ci)
        _adj_replace(pt_curves, pt_nc, cp2, last, ci)
    crv_n -= 1

def _del_curve(ci, skip=-1):
    p1, p2 = int(crvs[ci, 0]), int(crvs[ci, 1])
    if p1 != skip: _adj_remove(pt_curves, pt_nc, p1, ci)
    if p2 != skip: _adj_remove(pt_curves, pt_nc, p2, ci)
    _swap_del_crv(ci)

# ── GPU data builders ─────────────────────────────────────────────────────────

def _pt_coords():
    if pt_n == 0: return np.empty((0, 2), dtype=np.float32)
    v  = pts[:pt_n].astype(np.int32)
    return np.stack([v & 0x7FF, (v >> 11) & 0x3FF], axis=1).astype(np.float32)

def gpu_pts(mask=None):
    c = _pt_coords()
    if mask is not None: c = c[mask]
    return c.ravel()

def gpu_lns(mask=None, coords=None):
    if ln_n == 0: return np.empty(0, dtype=np.float32)
    if coords is None: coords = _pt_coords()
    idx = np.where(mask)[0] if mask is not None else np.arange(ln_n)
    if len(idx) == 0: return np.empty(0, dtype=np.float32)
    p1 = lns[idx, 0].astype(np.int32)
    p2 = lns[idx, 1].astype(np.int32)
    return np.stack([frame_coords[p1], frame_coords[p2]], axis=1).reshape(-1, 2).ravel()

def gpu_crvs(mask=None):
    if crv_n == 0: return np.empty(0, dtype=np.float32)
    idx = np.where(mask)[0] if mask is not None else np.arange(crv_n)
    if len(idx) == 0: return np.empty(0, dtype=np.float32)
    return crv_verts[idx].ravel()

# ── Hover ─────────────────────────────────────────────────────────────────────

def hover_point(mx, my):
    if pt_n == 0: return -1
    v  = pts[:pt_n].astype(np.int32)
    px = v & 0x7FF; py = (v >> 11) & 0x3FF
    hits = np.where(
        (np.abs(px - mx) <= 5) & (np.abs(py - my) <= 5) & ~pt_sel[:pt_n]
    )[0]
    return int(hits[0]) if len(hits) else -1

def hover_ctrl(mx, my):
    if crv_n == 0: return -1
    cx = crvs[:crv_n, 2].astype(np.int32)
    cy = crvs[:crv_n, 3].astype(np.int32)
    hits = np.where((np.abs(cx - mx) <= 5) & (np.abs(cy - my) <= 5))[0]
    return int(hits[0]) if len(hits) else -1

# ── State ─────────────────────────────────────────────────────────────────────

sel_pt            = -1
ctrl_hovered      = -1
dragging_ctrl_ci  = -1

show_ctrl_points  = False
show_hints        = False

line_in_progress  = False
line_start        = None
line_start_idx    = -1

curve_in_progress = False
curve_start       = None
curve_start_idx   = -1
curve_control     = None

dragging_point    = False
dragging_pt_idx   = -1

selection_in_progress = False
selection_start       = (0, 0)

dragging_selection    = False
dragging_sel_start    = (0, 0)

running       = True
render_needed = True
prev_sel_pt   = -1
prev_ctrl_hov = -1
prev_mp       = (-1, -1)
clock         = pygame.time.Clock()

# ── Main loop ─────────────────────────────────────────────────────────────────

while running:
    mp = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.VIDEOEXPOSE:
            render_needed = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_c:
                pt_n = 0; ln_n = 0; crv_n = 0
                pt_sel[:]  = False; ln_sel[:]  = False; crv_sel[:] = False
                sel_pt = -1; ctrl_hovered = -1; dragging_ctrl_ci = -1
                line_in_progress = False; curve_in_progress = False
                dragging_point = False; dragging_pt_idx = -1
                dirty = True; sel_dirty = True

            elif event.key == pygame.K_h:
                show_hints = not show_hints
                render_needed = True

            elif event.key == pygame.K_p:
                show_ctrl_points = not show_ctrl_points
                render_needed = True

            elif event.key == pygame.K_u:
                if sel_pt >= 0 and not curve_in_progress and not line_in_progress:
                    curve_in_progress = True
                    curve_start_idx   = sel_pt
                    sx, sy = unpack_pt(pts[sel_pt])
                    curve_start = (sx, sy)
                    if pt_nl[sel_pt] > 0:
                        li    = int(pt_lines[sel_pt, 0])
                        other = int(lns[li, 1]) if lns[li, 0] == sel_pt else int(lns[li, 0])
                        lx, ly = unpack_pt(pts[other])
                        dx, dy = sx - lx, sy - ly
                        length = np.hypot(dx, dy) or 1
                        px, py = -dy / length, dx / length
                        curve_control = (sx + px * 5, sy + py * 5)
                    else:
                        curve_control = (sx + 5, sy + 5)

            elif event.key == pygame.K_d:
                if sel_pt >= 0:
                    del_point(sel_pt)
                    sel_pt = -1
                    dirty = True; sel_dirty = True
                elif np.any(pt_sel[:pt_n]) or np.any(ln_sel[:ln_n]) or np.any(crv_sel[:crv_n]):
                    while pt_n > 0 and np.any(pt_sel[:pt_n]):
                        del_point(int(np.where(pt_sel[:pt_n])[0][0]))
                    while ln_n > 0 and np.any(ln_sel[:ln_n]):
                        _del_line(int(np.where(ln_sel[:ln_n])[0][0]))
                    while crv_n > 0 and np.any(crv_sel[:crv_n]):
                        _del_curve(int(np.where(crv_sel[:crv_n])[0][0]))
                    dirty = True; sel_dirty = True

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if dragging_ctrl_ci >= 0:
                    pass  # already dragging
                elif (np.any(pt_sel[:pt_n]) or np.any(ln_sel[:ln_n]) or np.any(crv_sel[:crv_n])) \
                        and not line_in_progress and not curve_in_progress:
                    dragging_selection = True
                    dragging_sel_start = mp
                elif ctrl_hovered >= 0:
                    dragging_ctrl_ci = ctrl_hovered
                elif sel_pt >= 0 and not line_in_progress and not curve_in_progress:
                    dragging_point  = True
                    dragging_pt_idx = sel_pt
                    sel_pt = -1
                elif sel_pt < 0 and not line_in_progress and not curve_in_progress \
                        and not np.any(pt_sel[:pt_n]):
                    add_point(mp[0], mp[1])
                    dirty = True

            elif event.button == 3:
                if sel_pt >= 0 and not line_in_progress:
                    line_in_progress = True
                    sx, sy = unpack_pt(pts[sel_pt])
                    line_start     = (sx, sy)
                    line_start_idx = sel_pt
                elif sel_pt < 0 and not line_in_progress:
                    pt_sel[:pt_n]   = False
                    ln_sel[:ln_n]   = False
                    crv_sel[:crv_n] = False
                    dirty = True; sel_dirty = True
                    selection_in_progress = True
                    selection_start       = mp

        elif event.type == pygame.MOUSEBUTTONUP:
            if dragging_ctrl_ci >= 0 and event.button == 1:
                dragging_ctrl_ci = -1

            elif dragging_point:
                dragging_point = False
                move_point(dragging_pt_idx, mp[0], mp[1])
                dragging_pt_idx = -1
                dirty = True

            elif dragging_selection:
                dragging_selection = False
                dx = mp[0] - dragging_sel_start[0]
                dy = mp[1] - dragging_sel_start[1]
                dup_map = {}

                def _dup_if_needed(pi):
                    if pt_sel[pi]: return pi
                    if pi in dup_map: return dup_map[pi]
                    ox, oy = unpack_pt(pts[pi])
                    ni = add_point(ox + dx, oy + dy)
                    dup_map[pi] = ni
                    return ni

                sel_idx = np.where(pt_sel[:pt_n])[0]
                if len(sel_idx):
                    v  = pts[sel_idx].astype(np.int32)
                    xs = np.clip((v & 0x7FF) + dx, 0, 2047)
                    ys = np.clip(((v >> 11) & 0x3FF) + dy, 0, 1023)
                    pts[sel_idx] = (xs | (ys << 11)).astype(np.uint32)

                if ln_n:
                    lp1 = lns[:ln_n, 0]; lp2 = lns[:ln_n, 1]
                    ls1 = pt_sel[lp1];   ls2 = pt_sel[lp2]
                    for li in np.where(ls1 != ls2)[0]:
                        p1, p2 = int(lns[li, 0]), int(lns[li, 1])
                        if pt_sel[p1]:
                            ni = _dup_if_needed(p2)
                            _adj_remove(pt_lines, pt_nl, p2, li)
                            lns[li, 1] = ni
                            _adj_add(pt_lines, pt_nl, ni, li)
                        else:
                            ni = _dup_if_needed(p1)
                            _adj_remove(pt_lines, pt_nl, p1, li)
                            lns[li, 0] = ni
                            _adj_add(pt_lines, pt_nl, ni, li)

                if crv_n:
                    cp1 = crvs[:crv_n, 0]; cp2 = crvs[:crv_n, 1]
                    cs1 = pt_sel[cp1];     cs2 = pt_sel[cp2]
                    for ci in np.where(cs1 | cs2)[0]:
                        p1, p2 = int(crvs[ci, 0]), int(crvs[ci, 1])
                        s1, s2 = bool(pt_sel[p1]), bool(pt_sel[p2])
                        if s1 and not s2:
                            ni = _dup_if_needed(p2)
                            _adj_remove(pt_curves, pt_nc, p2, ci)
                            crvs[ci, 1] = ni
                            _adj_add(pt_curves, pt_nc, ni, ci)
                        elif s2 and not s1:
                            ni = _dup_if_needed(p1)
                            _adj_remove(pt_curves, pt_nc, p1, ci)
                            crvs[ci, 0] = ni
                            _adj_add(pt_curves, pt_nc, ni, ci)
                        crvs[ci, 2] = int(round(float(crvs[ci, 2]) + dx))
                        crvs[ci, 3] = int(round(float(crvs[ci, 3]) + dy))
                        _bake_curve(ci)

                pt_sel[:pt_n]   = False
                ln_sel[:ln_n]   = False
                crv_sel[:crv_n] = False
                dirty = True; sel_dirty = True

            elif selection_in_progress:
                selection_in_progress = False
                x1, y1 = selection_start; x2, y2 = mp
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                if pt_n > 0:
                    v  = pts[:pt_n].astype(np.int32)
                    px = v & 0x7FF; py = (v >> 11) & 0x3FF
                    in_box = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
                    pt_sel[:pt_n] |= in_box
                for li in range(ln_n):
                    if pt_sel[lns[li, 0]] or pt_sel[lns[li, 1]]:
                        ln_sel[li] = True
                for ci in range(crv_n):
                    if pt_sel[crvs[ci, 0]] or pt_sel[crvs[ci, 1]]:
                        crv_sel[ci] = True
                dirty = True; sel_dirty = True

            elif line_in_progress:
                line_in_progress = False
                end_idx = sel_pt if sel_pt >= 0 else add_point(mp[0], mp[1])
                ex, ey  = unpack_pt(pts[end_idx])
                if (ex, ey) != line_start:
                    add_line(line_start_idx, end_idx)
                line_start = None; line_start_idx = -1
                dirty = True

            elif curve_in_progress:
                curve_in_progress = False
                end_idx = sel_pt if sel_pt >= 0 else add_point(mp[0], mp[1])
                ex, ey  = unpack_pt(pts[end_idx])
                if (ex, ey) != curve_start:
                    add_curve(curve_start_idx, end_idx, curve_control)
                curve_start = None; curve_start_idx = -1; curve_control = None
                dirty = True

    # ── Per-frame: ctrl drag ──────────────────────────────────────────────────

    if dragging_ctrl_ci >= 0:
        crvs[dragging_ctrl_ci, 2] = int(round(mp[0]))
        crvs[dragging_ctrl_ci, 3] = int(round(mp[1]))
        _bake_curve(dragging_ctrl_ci)
        dirty = True

    # ── Per-frame: point drag ─────────────────────────────────────────────────

    if dragging_point and dragging_pt_idx >= 0:
        move_point(dragging_pt_idx, mp[0], mp[1])
        dirty = True

    # ── Per-frame: hover ──────────────────────────────────────────────────────

    if not dragging_point and dragging_ctrl_ci < 0 and not selection_in_progress:
        ctrl_hovered = hover_ctrl(mp[0], mp[1])
        sel_pt = hover_point(mp[0], mp[1]) if ctrl_hovered < 0 else -1
    elif dragging_ctrl_ci >= 0:
        ctrl_hovered = dragging_ctrl_ci
        sel_pt = -1
    else:
        ctrl_hovered = -1
        sel_pt = -1

    # ── Render gate ───────────────────────────────────────────────────────────

    render_needed = (render_needed or dirty or sel_dirty or
                     sel_pt != prev_sel_pt or
                     ctrl_hovered != prev_ctrl_hov or
                     (mp != prev_mp and (line_in_progress or curve_in_progress or
                                         selection_in_progress or dragging_point or
                                         dragging_ctrl_ci >= 0)))
    if not render_needed:
        clock.tick(60)
        continue

    ctx.clear(0, 0, 0, 1)
    render_needed = False
    prev_sel_pt   = sel_pt
    prev_ctrl_hov = ctrl_hovered
    prev_mp       = mp

    # ── Upload caches ─────────────────────────────────────────────────────────

    frame_coords = _pt_coords()

    if dirty:
        unsel_pt  = ~pt_sel[:pt_n]
        unsel_ln  = ~ln_sel[:ln_n]   if ln_n  else np.empty(0, dtype=bool)
        unsel_crv = ~crv_sel[:crv_n] if crv_n else np.empty(0, dtype=bool)
        if dragging_point and dragging_pt_idx >= 0:
            unsel_pt = unsel_pt.copy()
            unsel_pt[dragging_pt_idx] = False
        pts_cache.upload(gpu_pts(unsel_pt))
        lns_cache.upload(np.concatenate([
            gpu_lns(unsel_ln,  frame_coords) if ln_n  else np.empty(0, dtype=np.float32),
            gpu_crvs(unsel_crv)              if crv_n else np.empty(0, dtype=np.float32),
        ]))
        dirty = False

    if sel_dirty:
        sel_pts_cache.upload(gpu_pts(pt_sel[:pt_n]) if pt_n else np.empty(0))
        sel_lns_cache.upload(np.concatenate([
            gpu_lns(ln_sel[:ln_n],   frame_coords) if ln_n  else np.empty(0, dtype=np.float32),
            gpu_crvs(crv_sel[:crv_n])              if crv_n else np.empty(0, dtype=np.float32),
        ]))
        sel_dirty = False

    # ── Render ────────────────────────────────────────────────────────────────

    lns_cache.render(moderngl.LINES,  (1, 1, 1), line_width=2)
    pts_cache.render(moderngl.POINTS, (1, 1, 1), point_size=10)

    sel_lns_cache.render(moderngl.LINES,  (0, 1, 0), line_width=2)
    sel_pts_cache.render(moderngl.POINTS, (0, 1, 0), point_size=10)

    # in-progress previews
    if line_in_progress and line_start:
        draw(moderngl.LINES, [*line_start, mp[0], mp[1]], (1, 0, 0))
    elif curve_in_progress and curve_start:
        draw(moderngl.LINE_STRIP,
             arc_verts(curve_start, curve_control, mp).ravel(), (1, 0, 0))
    elif selection_in_progress:
        x, y = selection_start
        draw(moderngl.LINE_LOOP, [x, y, mp[0], y, mp[0], mp[1], x, mp[1]], (1, 0, 0))

    # dragged point preview
    if dragging_point and dragging_pt_idx >= 0:
        dx2, dy2 = unpack_pt(pts[dragging_pt_idx])
        draw(moderngl.POINTS, [dx2, dy2], (1, 0, 0), point_size=10)
        lv = []
        for k in range(int(pt_nl[dragging_pt_idx])):
            li = int(pt_lines[dragging_pt_idx, k])
            lv += [dx2, dy2, *frame_coords[int(lns[li, 1]) if lns[li, 0] == dragging_pt_idx else int(lns[li, 0])]]
        if lv: draw(moderngl.LINES, lv, (1, 0, 0))
        cv = [crv_verts[int(pt_curves[dragging_pt_idx, k])]
              for k in range(int(pt_nc[dragging_pt_idx]))]
        if cv: draw(moderngl.LINES, np.concatenate(cv), (1, 0, 0), line_width=2)

    # hover highlight
    if sel_pt >= 0:
        sx, sy = unpack_pt(pts[sel_pt])
        draw(moderngl.POINTS, [sx, sy], (0, 1, 0), point_size=10)
        lv = []
        for k in range(int(pt_nl[sel_pt])):
            li = int(pt_lines[sel_pt, k])
            lv += [*frame_coords[int(lns[li, 0])], *frame_coords[int(lns[li, 1])]]
        if lv: draw(moderngl.LINES, lv, (0, 1, 0), line_width=2)
        cv = [crv_verts[int(pt_curves[sel_pt, k])]
              for k in range(int(pt_nc[sel_pt]))]
        if cv: draw(moderngl.LINES, np.concatenate(cv), (0, 1, 0), line_width=2)

    # ctrl points
    if crv_n > 0:
        cx = crvs[:crv_n, 2].astype(np.float32)
        cy = crvs[:crv_n, 3].astype(np.float32)
        ctrl_xy  = np.stack([cx, cy], axis=1)
        p1c = frame_coords[crvs[:crv_n, 0].astype(np.int32)]
        p2c = frame_coords[crvs[:crv_n, 1].astype(np.int32)]
        arms = np.stack([p1c, ctrl_xy, p2c, ctrl_xy], axis=1).reshape(-1, 2)

        if show_ctrl_points:
            if ctrl_hovered >= 0:
                mask = np.ones(crv_n, dtype=bool); mask[ctrl_hovered] = False
                draw(moderngl.LINES,  arms[np.repeat(mask, 4)],  (1, 1, 0.4), line_width=1)
                draw(moderngl.POINTS, ctrl_xy[mask],              (1, 1, 0.4), point_size=8)
                draw(moderngl.LINES,  arms[np.repeat(~mask, 4)], (1, 0.5, 0), line_width=1)
                draw(moderngl.POINTS, ctrl_xy[~mask],             (1, 0.5, 0), point_size=8)
            else:
                draw(moderngl.LINES,  arms,    (1, 1, 0.4), line_width=1)
                draw(moderngl.POINTS, ctrl_xy, (1, 1, 0.4), point_size=8)
        else:
            # show ctrl for hovered point's curves
            if sel_pt >= 0:
                for k in range(int(pt_nc[sel_pt])):
                    ci = int(pt_curves[sel_pt, k])
                    if ci == ctrl_hovered: continue
                    draw(moderngl.LINES,  arms[ci*4:ci*4+4], (1, 1, 0.4), line_width=1)
                    draw(moderngl.POINTS, ctrl_xy[ci:ci+1],  (1, 1, 0.4), point_size=8)
            if ctrl_hovered >= 0:
                ci = ctrl_hovered
                draw(moderngl.LINES,  arms[ci*4:ci*4+4], (1, 0.5, 0), line_width=1)
                draw(moderngl.POINTS, ctrl_xy[ci:ci+1],  (1, 0.5, 0), point_size=8)

    if show_hints:
        if hint_overlay is None:
            hint_overlay = make_hint_overlay()
        tex, vao, vbo = hint_overlay
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        tex.use(0); text_prog['tex'].value = 0
        vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
