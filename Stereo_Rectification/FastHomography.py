import numpy as np


def warpHomo(H, source_image):
    h, w = source_image.shape[:2]
    target_image = np.zeros(source_image.shape)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    points = np.hstack((x, y, np.ones(x.shape))).T
    mapped_points = np.dot(H, points)
    mapx, mapy, mapw = mapped_points[0, :], mapped_points[1, :], mapped_points[2, :]

    mapx = np.int32(mapx / mapw)
    mapy = np.int32(mapy / mapw)
    valid_indices = np.where((mapx >= 0) & (mapy >= 0) & (mapx < w) & (mapy < h))[0]
    mapx = mapx[valid_indices]
    mapy = mapy[valid_indices]
    y = y[valid_indices].flatten()
    x = x[valid_indices].flatten()
    # print(mapx.shape, mapy.shape, x.shape, y.shape, source_image.shape, target_image.shape)

    target_image[y, x, :] = source_image[mapy, mapx, :]
    #     print(target_image[y,x,:].shape,source_image[mapy,mapx,:].shape)
    return target_image
