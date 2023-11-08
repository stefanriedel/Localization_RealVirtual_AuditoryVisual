import numpy as np
import matplotlib.pyplot as plt


def cart2sph(x, y, z):
    """Convert cartesian to spherical coordinates.

    Args: unit vector in cart coordinates
        ux (ndarray): x 
        uy (ndarray): y
        uz (ndarray): z

    Returns:
        azi, zen : spherical coordinates in rad
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    azi = np.arctan2(y, x)
    zen = np.arccos(z)

    return azi, zen


def plotTriangulation(hull, ls_xyz):
    """Plot triangulated layout / convex hull.

    Args:
        hull (ConvexHull): convex hull
        ls_xyz (ndarray): LS directions
    """

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d')

    ls_x = ls_xyz[:, 0]
    ls_y = ls_xyz[:, 1]
    ls_z = ls_xyz[:, 2]

    # ax.plot(ls_x, ls_y, ls_z, 'bo', ms=10)
    ax.plot(ls_xyz[hull.vertices, 0],
            ls_xyz[hull.vertices, 1],
            ls_xyz[hull.vertices, 2], 'ko', markersize=4)
    s = ax.plot_trisurf(ls_x, ls_y, ls_z, triangles=hull.simplices,
                        cmap='viridis', alpha=0.9, edgecolor='k')
    # plt.colorbar(s, shrink=0.7)

    plt.show()


def distance_line_to_point(line_base, line_dir, point):
    """Compute distance between a line and a point in space.

    Args: 
        line_base (ndarray): base point of line 
        line_dir (ndarray): unit direction vector of line
        point (ndarray): point in space 

    Returns:
        dist : shortest normal distance between line and point
    """

    pma = point - line_base[None, :]
    proj = np.dot(pma, line_dir[:, np.newaxis])
    return np.linalg.norm((pma) - np.dot(proj, line_dir[np.newaxis, :]), axis=1)


def find_piercing_point(line_base, line_dir, hull, ls_xyz):
    """Compute piercing point of a line and a convex hull

    Args: 
        line_base (ndarray): base point of line (3,)
        line_dir (ndarray): unit direction vector of line (3,)
        hull (object): scipy.spatial.ConvexHull object
        ls_xyz (ndarray): loudspeaker coordinates in 3D space (N x 3)

    Returns:
        point, x_eq : point and solution parameters
    """

    dist = distance_line_to_point(line_base, line_dir, ls_xyz)
    sorted_dist = np.argsort(dist)
    for vertex_candidate in sorted_dist:
        triangle_candidates = np.where(hull.simplices == vertex_candidate)[0]
        for triangle_candidate in triangle_candidates:
            a = ls_xyz[hull.simplices[triangle_candidate, 0], :]
            b = ls_xyz[hull.simplices[triangle_candidate, 1], :]
            c = ls_xyz[hull.simplices[triangle_candidate, 2], :]
            v = line_dir
            p = line_base
            b_eq = p - a
            A_eq = np.array([b-a, c-a, -v]).transpose()
            x_eq = np.linalg.solve(A_eq, b_eq)
            if x_eq[2] >= 0 and x_eq[0] >= 0 and x_eq[1] >= 0 and (x_eq[0] + x_eq[1]) <= 1:
                return p + x_eq[2] * v, x_eq
            else:
                continue


def convert_piercing_to_azi_ele(point, center, degrees=False):
    """Convert piercing point to azi and ele

    Args: 
        point (ndarray): direction vector/point in space (3,)
        center (ndarray): coordinate system center (3,)


    Returns:
        azi, ele: spherical coordinates in deg
    """
    vec = point - center
    unit_vec = vec / np.linalg.norm(vec)
    azi, zen = cart2sph(unit_vec[0], unit_vec[1], unit_vec[2])
    if degrees:
        return azi * 180 / np.pi, (np.pi/2 - zen) * 180 / np.pi
    else:
        return azi, np.pi/2 - zen
