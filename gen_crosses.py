from PIL import Image, ImageDraw
import random, math

random.seed(7)

img = Image.open('/Users/jaydeepde/work/Research/A2E/image.png').convert('RGBA')
draw = ImageDraw.Draw(img)

net_x_starts = [198, 360, 528, 699, 869, 1040, 1212, 1380]
subcol_offsets = [17, 50, 83, 116, 149]
row_ys = [int(108 + i * 57.3) for i in range(10)]

def jitter(v, r=4):
    return v + random.randint(-r, r)

def rot_point(cx, cy, x, y, angle_deg):
    a = math.radians(angle_deg)
    dx, dy = x - cx, y - cy
    return cx + dx*math.cos(a) - dy*math.sin(a), cy + dx*math.sin(a) + dy*math.cos(a)

def wobbly_line(draw, x1, y1, x2, y2, color, width):
    mx = (x1+x2)/2 + random.randint(-3, 3)
    my = (y1+y2)/2 + random.randint(-3, 3)
    draw.line([(x1,y1),(mx,my)], fill=color, width=width)
    draw.line([(mx,my),(x2,y2)], fill=color, width=width)

def cross_x(draw, cx, cy, size, color, width=3):
    tilt = random.uniform(-8, 8)
    pts = [rot_point(0,0,p[0],p[1],tilt) for p in [(-size,-size),(size,size),(size,-size),(-size,size)]]
    wobbly_line(draw, cx+pts[0][0]+jitter(0,3), cy+pts[0][1]+jitter(0,3),
                      cx+pts[1][0]+jitter(0,3), cy+pts[1][1]+jitter(0,3), color, width)
    wobbly_line(draw, cx+pts[2][0]+jitter(0,3), cy+pts[2][1]+jitter(0,3),
                      cx+pts[3][0]+jitter(0,3), cy+pts[3][1]+jitter(0,3), color, width)

def cross_thick_x(draw, cx, cy, size, color):
    cross_x(draw, cx, cy, size, color, width=4)
    cross_x(draw, cx+jitter(0,2), cy+jitter(0,2), size, color, width=2)

def cross_plus(draw, cx, cy, size, color, width=3):
    tilt = random.uniform(-8, 8)
    for (dx1,dy1,dx2,dy2) in [(-size,0,size,0),(0,-size,0,size)]:
        p1 = rot_point(cx,cy,cx+dx1,cy+dy1,tilt)
        p2 = rot_point(cx,cy,cx+dx2,cy+dy2,tilt)
        wobbly_line(draw, p1[0]+jitter(0,2), p1[1]+jitter(0,2),
                          p2[0]+jitter(0,2), p2[1]+jitter(0,2), color, width)

def cross_double_x(draw, cx, cy, size, color):
    cross_x(draw, cx,             cy,             size,   color, width=2)
    cross_x(draw, cx+jitter(0,3), cy+jitter(0,3), size-2, color, width=2)

cross_fns = [cross_x, cross_thick_x, cross_plus, cross_double_x]

colors = [
    (15,  15,  90, 230),   # dark blue ink
    (100,  0,   0, 220),   # dark red
    (10,  10,  10, 240),   # near black
    (20,  60, 120, 220),   # blue-black
]

cross_idx = 0
for ni, nx in enumerate(net_x_starts):
    for sci, sco in enumerate(subcol_offsets):
        cx = nx + sco
        max_row = 1 if sci == 0 else 9
        row_i = random.randint(0, max_row)
        cy = row_ys[row_i]
        fn   = cross_fns[cross_idx % len(cross_fns)]
        col  = colors[cross_idx % len(colors)]
        size = random.randint(13, 20)
        fn(draw, cx, cy, size, col)
        cross_idx += 1

img.convert('RGB').save('/Users/jaydeepde/work/Research/A2E/image_crosses.png')
print("Saved image_crosses.png")
