"""obj file interface
"""
from typing import Tuple
import numpy as np


def save_obj(filename: str, vertices: np.ndarray, faces: np.ndarray = None, colors=(0.5, 0.5, 0.5)):
    """save obj

    Args:
        filename (str): *.obj
        vertices (np.ndarray): (N, 3)
        faces (np.ndarray, optional): (M, 3). Defaults to None.
        colors (tuple, optional): rgb. Defaults to (0.5, 0.5, 0.5).
    """
    if isinstance(colors, list) or isinstance(colors, tuple):
        colors = np.array(colors)
    if colors is not None:
        if len(colors.shape) == 1:
            colors = colors[None]
        if colors.shape[0] == 1:
            colors = colors.repeat(vertices.shape[0], axis=0)

    assert len(vertices.shape) == 2
    assert vertices.shape[-1] == 3

    with open(filename, "w") as fw:
        for i in range(vertices.shape[0]):
            x, y, z = vertices[i]
            if colors is not None:
                r, g, b = colors[i]
                fw.write(f"v {x} {y} {z} {r} {g} {b}\n")
            else:
                fw.write(f"v {x} {y} {z}\n")

        if faces is not None:
            for i in range(faces.shape[0]):
                a = faces[i][0]
                b = faces[i][1]
                c = faces[i][2]
                fw.write(f"f {a+1} {b+1} {c+1}\n")


def load_obj(obj_filename: str) -> Tuple[np.ndarray]:
    v_list = []
    f_list = []

    with open(obj_filename, "r") as fr:
        for line in fr.readlines():
            segs = line.split(" ")
            if segs[0] == "v":
                x = float(segs[1])
                y = float(segs[2])
                z = float(segs[3])
                v_list.append([x, y, z])

            elif segs[0] == "f":
                a = int(segs[1])
                b = int(segs[2])
                c = int(segs[3])
                f_list.append([a, b, c])
    vertices = np.array(v_list, dtype=np.float32)
    if len(f_list) == 0:
        faces = None
    else:
        faces = np.array(f_list, dtype=np.int)
        if np.min(faces) == 1:
            faces -= 1
    return vertices, faces
