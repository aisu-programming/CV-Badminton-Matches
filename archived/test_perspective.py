# Failed
def matplotlib_way():
    import matplotlib.patches
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.art3d as art3d

    lines = []

    olc = outer_line_corner = {'x':303, 'y':668}
    signs = [ [1,1], [1,-1], [-1,-1], [-1,1], [1,1] ]
    for sign_id in range(len(signs)-1):
        sign_x1, sign_y1 = signs[sign_id]
        sign_x2, sign_y2 = signs[sign_id+1]
        x1, y1 = sign_x1 * olc['x'], sign_y1 * olc['y']
        x2, y2 = sign_x2 * olc['x'], sign_y2 * olc['y']
        lines.append(([x1, x2], [y1, y2], [0,0]))

    sslc = short_service_line_corner = {'x':303, 'y':200}
    for sign_y in [1, -1]:
        x1, y1 = ( 1) * sslc['x'], sign_y * sslc['y']
        x2, y2 = (-1) * sslc['x'], sign_y * sslc['y']
        lines.append(([x1, x2], [y1, y2], [0,0]))

    lsl4dc = long_service_line_4_doubles_corner = {'x':303, 'y':592}
    for sign_y in [1, -1]:
        x1, y1 = ( 1) * lsl4dc['x'], sign_y * lsl4dc['y']
        x2, y2 = (-1) * lsl4dc['x'], sign_y * lsl4dc['y']
        lines.append(([x1, x2], [y1, y2], [0,0]))

    sl4sc = side_line_4_singles_corner = {'x':257, 'y':668}
    for sign_x in [1, -1]:
        x1, y1 = sign_x * sl4sc['x'], ( 1) * sl4sc['y']
        x2, y2 = sign_x * sl4sc['x'], (-1) * sl4sc['y']
        lines.append(([x1, x2], [y1, y2], [0,0]))

    clc = center_line_corner = {'y1':668, 'y2':200}
    for sign_y in [1, -1]:
        y1, y2 = sign_y * clc['y1'], sign_y * clc['y2']
        lines.append(([0, 0], [y1, y2], [0,0]))

    court_corners = [[-350,-350,350,350], [750,-750,-750,750], [0,0,0,0]]
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)
    plt.fill(*court_corners[:2], facecolor="g")
    for ln_x, ln_y, ln_z in lines:
        plt.plot(ln_x, ln_y, marker='', color="w")
    ax.set_xlim(-800, 800)
    ax.set_ylim(-800, 800)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    for ln_x, ln_y, ln_z in lines:
        length_x = max(ln_x) - min(ln_x)
        length_y = max(ln_y) - min(ln_y)
        if length_x == 0:
            ln_x = int(min(ln_x)-4)
            ln_y = min(ln_y)
            length_x = 4
        elif length_y == 0:
            ln_y = int(min(ln_y)-4)
            ln_x = min(ln_x)
            length_y = 4
        line = matplotlib.patches.Rectangle((ln_x, ln_y), length_x, length_y, facecolor="b")
        ax.add_patch(line)
        art3d.pathpatch_2d_to_3d(line, z=10, zdir="z")
    court = matplotlib.patches.Rectangle((-350,-700), 700, 1500, facecolor="g")
    ax.add_patch(court)
    art3d.pathpatch_2d_to_3d(court, z=0, zdir="z")
    ax.set_xlim(-800, 800)
    ax.set_ylim(-800, 800)
    ax.set_zlim(0, 100)
    plt.tight_layout()
    plt.show()