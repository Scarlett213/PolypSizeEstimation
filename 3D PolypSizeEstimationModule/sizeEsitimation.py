import numpy as np
import open3d as o3d
import time

def rodrigues_rot(P, n0, n1):
    if P.ndim == 1:
        P = P[np.newaxis, :]
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.dot(n0, n1))
    P_rot = np.zeros((len(P), 3))
    for i in range(len(P)):
        P_rot[i] = P[i] * np.cos(theta) + np.cross(k, P[i]) * np.sin(theta) + k * np.dot(k, P[i]) * (1 - np.cos(theta))

    return P_rot

def fit_circle_2d(x, y, w=[]):
    A = np.array([x, y, np.ones(len(x))]).T
    b = x ** 2 + y ** 2

    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc ** 2 + yc ** 2)
    return xc, yc, r

def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    P_circle = r * np.cos(t)[:, np.newaxis] * u + r * np.sin(t)[:, np.newaxis] * np.cross(n, u) + C
    return P_circle


# https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
def circle_segmentation(cloud):
    P_mean = cloud.mean(axis=0)
    P_centered = cloud - P_mean
    U, s, V = np.linalg.svd(P_centered)
    normal = V[2, :]
    d = -np.dot(P_mean, normal)
    P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
    xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
    t = np.linspace(0, 2 * np.pi, 100)
    xx = xc + r * np.cos(t)
    yy = yc + r * np.sin(t)
    C = rodrigues_rot(np.array([xc, yc, 0]), [0, 0, 1], normal) + P_mean
    C = C.flatten()
    t = np.linspace(0, 2 * np.pi, 1000)
    u = cloud[0] - C
    P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)
    return P_fitcircle, C, r

def ThreeDSize(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.uniform_down_sample(100)
    P = np.asarray(pcd.points)
    # print(P.shape)
    circle, circle_center, radius = circle_segmentation((P))
    return radius*2, pcd

def calSize(ply_path):
    start = time.time()
    size, _ = ThreeDSize(ply_path)
    return size


if __name__ =='__main__':
    ply_path = ''# format ply
    size = calSize(ply_path)
    print('the size of the polyp is {} cm'.format(size))