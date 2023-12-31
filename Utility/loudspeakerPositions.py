import numpy as np

xyz = np.array([[1.85, 0.0, 1.20],
                [1.85, 1.03, 1.20],
                [0.0, 1.75, 1.20],
                [-1.84, 1.06, 1.20],
                [-1.84, 0.0, 1.20],
                [-1.84, -1.04, 1.20],
                [0.0, -1.90, 1.20],
                [1.85, -1.05, 1.20],

                [1.85, 0.0, 2.22],
                [1.85, 1.03, 2.22],
                [0.0, 1.75, 2.22],
                [-1.84, 1.06, 2.22],
                [-1.84, 0.0, 2.22],
                [-1.84, -1.04, 2.22],
                [0.0, -1.90, 2.22],
                [1.85, -1.05, 2.22],

                [0.85, 0.05, 3.06],
                [0.0, 1.07, 3.11],
                [-0.88, 0.0, 3.09],
                [0.0, -1.00, 3.10],
                [0.0, 0.0, 3.10],

                [0.0, -1.90, 1.68],
                [1.85, -1.05, 1.72],
                [1.85, 0.0, 1.70],
                [1.85, 0.0, 0.70],
                ])

azi = [0, 30, 90, 150, 180, -150, -90, -30] * 2 + \
    [0, 90, 180, -90] + [0] + [-90, -30, 0] + [0]
ele = [0] * 8 + [30] * 8 + [60] * 4 + [90] + [15] * 3 + [-15]

azi_hull = azi + [30, 90, 150, 180, -150, -90, -30] + [0]
ele_hull = ele + [-15] * 7 + [-90]

add_hull_ls = np.array([[1.85, 1.03,  0.70],
                        [0.0, 1.75,  0.70],
                        [-1.84, 1.06,  0.70],
                        [-1.84, 0.0,  0.70],
                        [-1.84, -1.04,  0.70],
                        [0.0, -1.90,  0.70],
                        [1.85, -1.05,  0.70],
                        [0, 0, 0]])
xyz_hull = np.concatenate((xyz, add_hull_ls), axis=0)
