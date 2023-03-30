# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def visualize_counterfactuals(
    edits,
    query_index,
    distractor_index,
    dataset,
    n_pix,
    fname=None,
):
    # load image
    query_img = dataset.__getitem__(query_index)
    height, width = query_img.shape[0], query_img.shape[1]

    # geometric properties of cells
    width_cell = width // n_pix
    height_cell = height // n_pix

    # create plot
    n_edits = len(edits)
    _, axes = plt.subplots(n_edits, 2)
    if n_edits == 1:
        axes = [axes]

    # loop over edits
    for ii, edit in enumerate(edits):
        # show query
        cell_index_query = edit[0]
        row_index_query = cell_index_query // n_pix
        col_index_query = cell_index_query % n_pix

        query_left_box = int(col_index_query * width_cell)
        query_top_box = int(row_index_query * height_cell)

        rect = patches.Rectangle(
            (query_left_box, query_top_box),
            width_cell,
            height_cell,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        axes[ii][0].imshow(query_img)
        axes[ii][0].add_patch(rect)
        axes[ii][0].get_xaxis().set_ticks([])
        axes[ii][0].get_yaxis().set_ticks([])
        if ii == 0:
            axes[ii][0].set_title("Query")

        # show distractor
        cell_index_distractor = edit[1]

        index_distractor = distractor_index[cell_index_distractor // (n_pix**2)]
        img_distractor = dataset.__getitem__(index_distractor)

        cell_index_distractor = cell_index_distractor % (n_pix**2)
        row_index_distractor = cell_index_distractor // n_pix
        col_index_distractor = cell_index_distractor % n_pix

        distractor_left_box = int(col_index_distractor * width_cell)
        distractor_top_box = int(row_index_distractor * height_cell)

        rect = patches.Rectangle(
            (distractor_left_box, distractor_top_box),
            width_cell,
            height_cell,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        axes[ii][1].imshow(img_distractor)
        axes[ii][1].add_patch(rect)
        axes[ii][1].get_xaxis().set_ticks([])
        axes[ii][1].get_yaxis().set_ticks([])
        if ii == 0:
            axes[ii][1].set_title("Distractor")

    # save or view
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()

def smoothstep(e0, e1, x):
    t = max(min((x - e0) / (e1 - e0), 1.0), 0.0)
    return t * t * (3.0 - 2.0 * t)

def circular_crop(image):
    width = image.shape[0]
    height = image.shape[1]
    tolerance = 0.25
    min_radius = 0.5 - tolerance
    max_radius = 0.5
    mask = np.zeros((width, height))
    output = np.zeros((width, height, 4))
    for i in range(width):
        for j in range(height):
            v = [i / width, j / height]
            d = np.linalg.norm(v - np.array([0.5, 0.5]))
            s = smoothstep(min_radius, max_radius, d)
            mask[i, j] = 1.0 - s
            output[i, j] = np.append(image[i, j] / 255, [1.0 - s])
    return output

def visualize_merge(
    edits,
    query_index,
    distractor_index,
    dataset,
    n_pix,
    radius=0.5*np.sqrt(2),
    n_edits=1,
    fname=None,
):
    query_img = dataset.__getitem__(query_index)
    height, width = query_img.shape[0], query_img.shape[1]

    # geometric properties of cells
    width_cell = width // n_pix
    height_cell = height // n_pix

    plt.figure(figsize=(10, 10))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.imshow(query_img, extent=(0, 1, 0, 1))

    for i in range(min(len(edits), n_edits)):
        edit = edits[i]
        cell_index_query = edit[0]
        row_index_query = cell_index_query // n_pix
        col_index_query = cell_index_query % n_pix

        cell_index_distractor = edit[1]

        index_distractor = distractor_index[cell_index_distractor // (n_pix**2)]
        distractor_img = dataset.__getitem__(index_distractor)

        cell_index_distractor = cell_index_distractor % (n_pix**2)
        row_index_distractor = cell_index_distractor // n_pix
        col_index_distractor = cell_index_distractor % n_pix

        crop = [
            max(0, round((row_index_distractor + 0.5 - radius) * height_cell)),
            round((row_index_distractor + 0.5 + radius) * height_cell),
            max(0, round((col_index_distractor + 0.5 - radius) * width_cell)),
            round((col_index_distractor + 0.5 + radius) * width_cell)
        ]
        distractor_img_cropped = distractor_img[crop[0]:crop[1], crop[2]:crop[3]]
        distractor_img_masked = circular_crop(distractor_img_cropped)

        extent = ((col_index_query + 0.5 - radius) / n_pix, (col_index_query + 0.5 + radius) / n_pix, (n_pix - row_index_query - 0.5 - radius) / n_pix, (n_pix - row_index_query - 0.5 + radius) / n_pix)
        
        plt.imshow(distractor_img_masked, extent=extent)
    
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()

def save_image(
    img,
    fname=None,
):
    plt.figure(figsize=(10, 10))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.imshow(img, extent=(0, 1, 0, 1))
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


def visualize_edit(
    img,
    x_coord,
    y_coord,
    n_pix,
    fname
):
    plt.figure(figsize=(10, 10))
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.imshow(img, extent=(0, 1, 0, 1))

    rect = patches.Rectangle(
            (x_coord / n_pix, 1 - (y_coord + 1) / n_pix),
            1 / n_pix,
            1 / n_pix,
            linewidth=5,
            edgecolor="k",
            facecolor="none",
        )
    ax.add_patch(rect)
    plt.tight_layout()
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()

def visualize_edits(
    edits,
    query_index,
    distractor_index,
    dataset,
    n_pix,
    radius=0.5*np.sqrt(2),
    n_edits=1,
    fname=None,
):
    for i in range(1, n_edits + 1):
        visualize_merge(
            edits=edits,
            query_index=query_index,
            distractor_index=distractor_index,
            dataset=dataset,
            n_pix=n_pix,
            radius=radius,
            n_edits=i,
            fname=f"{fname}/merge_{n_edits}.png"
        )
    
    query_img = dataset.__getitem__(query_index)
    save_image(query_img, f"{fname}/query.png")
    for i in range(min(len(edits), n_edits)):
        edit = edits[i]
        cell_index_distractor = edit[1]

        index_distractor = distractor_index[cell_index_distractor // (n_pix**2)]
        distractor_img = dataset.__getitem__(index_distractor)

        save_image(distractor_img, f"{fname}/distractor_{i}.png")
        
        cell_index_query = edit[0]
        row_index_query = cell_index_query // n_pix
        col_index_query = cell_index_query % n_pix

        visualize_edit(query_img, col_index_query, row_index_query, n_pix, f"{fname}/query_edit_{i}.png")     

        cell_index_distractor = cell_index_distractor % (n_pix**2)
        row_index_distractor = cell_index_distractor // n_pix
        col_index_distractor = cell_index_distractor % n_pix        

        visualize_edit(distractor_img, col_index_distractor, row_index_distractor, n_pix, f"{fname}/distractor_edit_{i}.png")     