import pylidc as pl
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours
from pylidc.utils import consensus


# THIS IS NEEDED SINCE THE NUMPY HAS VERSION CONFLICT
np.bool = np.bool_
np.int = int


def displayCTImageByPID(pid: str) -> None:
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    print(f"Patient {pid} has {len(scan.annotations)} annotations.")

    nodules = scan.cluster_annotations()
    print(f"Scan of {pid} has {len(nodules)} nodules.")

    vol = scan.to_volume()
    print(f"Volume shape: {vol.shape}")

    scan.visualize(annotation_groups=nodules)

    # choose mid slice
    mid_slice = vol.shape[2] // 2
    plt.imshow(vol[:, :, mid_slice], cmap='gray')
    plt.title(f"Patient {pid} - Slice {mid_slice}")
    plt.axis('off')
    plt.show()


def displayFourDoctorsAnnotationsByPID(pid: str) -> None:
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = scan.to_volume()

    nodules = scan.cluster_annotations()
    anns = nodules[0]

    cmask, cbbox, masks = consensus(anns, clevel=0.5, pad=[(20, 20), (20, 20), (0, 0)])
    # get center slice
    k = int(0.5 * (cbbox[2].stop - cbbox[2].start))

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(vol[cbbox][:, :, k], cmap=plt.cm.gray, alpha=0.5)

    # label with different color
    added_labels = set()
    colors = ['r', 'g', 'b', 'y']
    for j in range(len(masks)):
        for contour in find_contours(masks[j][:, :, k].astype(float), 0.5):
            label = f"Annotation {j + 1}"
            if label not in added_labels:  # check if already lable
                plt.plot(contour[:, 1], contour[:, 0], colors[j], label=label)
                added_labels.add(label)

    for contour in find_contours(cmask[:, :, k].astype(float), 0.5):
        if '50% Consensus' not in added_labels:
            plt.plot(contour[:, 1], contour[:, 0], '--k', label='50% Consensus')
            added_labels.add('50% Consensus')

    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    "LIDC-IDRI-0066_N001_S568"
    pid = 'LIDC-IDRI-0066'
    displayCTImageByPID(pid)
    displayFourDoctorsAnnotationsByPID(pid)







