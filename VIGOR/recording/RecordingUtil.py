import os
import shutil

import numpy as np
from PyPDF2 import PdfFileMerger

import util.Defaults as d
from util.Functions import format_title

_scalar_ids = {d.TOTAL_LOSS: 0,
               d.ACCURACY: 1,
               d.BINARY_CROSS_ENTROPY: 2,
               d.MEAN_ABSOLUTE_ERROR: 11,
               d.ROOT_MEAN_SQUARED_ERROR: 12,
               d.MEAN_SQUARED_ERROR: 13,
               d.COMPARISON_BASED_LOSS: 14,
               d.STEPWISE_L1_LOSS: 105,
               d.STEPWISE_L2_LOSS: 106,
               }

_scalar_names = {d.TOTAL_LOSS: "Total Loss",
                 d.ACCURACY: "Accuracy",
                 d.BINARY_CROSS_ENTROPY: "Binary Cross Entropy",
                 d.MEAN_ABSOLUTE_ERROR: "Mean Absolute Error",
                 d.ROOT_MEAN_SQUARED_ERROR: "Root Mean Squared Error",
                 d.MEAN_SQUARED_ERROR: "Mean Squared Error",
                 d.COMPARISON_BASED_LOSS: "Comparison Based Loss",
                 d.STEPWISE_L1_LOSS: "Stepwise Target L1 Loss",
                 d.STEPWISE_L2_LOSS: "Stepwise Target L2 Loss",
                 }

DEFAULT_ID = 999
DEFAULT_TITLE = "UnknownTitle"


def scalar_to_title(scalar_name: str, format: bool = False) -> str:
    """

    Args:
        scalar_name: Name of the used scalar
        format: Whether to format the title for plotting or not

    Returns: A title corresponding to the scalar Name

    """
    title = _scalar_names.get(scalar_name, DEFAULT_TITLE)
    if format:
        title = format_title(title=title)
    return title


def scalar_to_id(scalar_name: str) -> int:
    """
    Looks up and returns the id of the given scalar
    Args:
        scalar_name: Name of the used scalar

    Returns: The id corresponding to the scalar Name.

    """
    return _scalar_ids.get(scalar_name, DEFAULT_ID)


def merge_folder_to_file(target_path: str, delete_folder: bool = True, recursion_depth: int = 1,
                         make_video: bool = True) -> bool:
    """
    Merges all .pdf files found in the specified folder into a single file of the name
    folder_name.pdf which is then saved at the same level as the folder.
    If delete_folder, the folder and its contents are deleted afterwards
    Args:
        target_path: Full path to folder to merge pdfs from
        delete_folder: Whether to delte the folder afterwards or not
        recursion_depth: If the target path is a folder, this method is called for all subdirectories of depth
        recursion_depth
        make_video: Whether to make a video of the .pdf files on unix based systems or not.
    Returns: A boolean. True if pdfs were merged in this call, false otherwise

    """

    def _sort_pdf(filename: str):
        filename = filename[:-4]  # remove ".pdf"
        if filename.isnumeric():
            return int(filename)
        else:
            return -1

    has_merged_subfolder = False  # whether or not any child folders have merged pdfs
    if recursion_depth > 0:
        for subdir in [x for x in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, x))]:
            recursive_dir = os.path.join(target_path, subdir)
            is_merged_subfolder = merge_folder_to_file(target_path=recursive_dir, delete_folder=delete_folder,
                                                       recursion_depth=recursion_depth - 1)
            has_merged_subfolder = has_merged_subfolder or is_merged_subfolder

    if not has_merged_subfolder:  # only merge if the child folders have not
        pdfs_to_merge = [x for x in os.listdir(target_path) if x.endswith(".pdf")]
        if len(pdfs_to_merge) > 0:
            pdfs_to_merge = list(
                sorted(pdfs_to_merge, key=_sort_pdf))  # weird hack to have plots that are not numbered first.

            # merge
            merger = PdfFileMerger()
            for next_pdf in pdfs_to_merge:
                merger.append(os.path.join(target_path, next_pdf))

            merger.write(target_path + ".pdf")
            merger.close()

            if os.name == "posix" and make_video:  # try making a video on unix-based systems
                try:
                    pdf_to_avi(path_to_pdf=target_path)
                except Exception as e:
                    print("Error with video: '{}'".format(e))

            if delete_folder:  # only delete folders that were merged, which must be low-level folders
                shutil.rmtree(target_path)
            return True
        else:
            return False
    else:
        return True


def pdf_to_avi(path_to_pdf: str):
    import fitz
    import cv2
    pdf_document = fitz.open(path_to_pdf + ".pdf")

    all_images = []
    for page in pdf_document.pages():
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.getPixmap(matrix=mat)
        imgData = pix.getImageData("png")

        # save image from opencv
        nparr = np.frombuffer(imgData, np.uint8)
        current_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        all_images.append(current_image)

    size = current_image.shape[:2]
    out = cv2.VideoWriter(path_to_pdf + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps=10, frameSize=size[::-1])

    for image in all_images:
        out.write(image)
    out.release()


def parse_last_results(metrics: dict) -> dict:
    """
    Retrieves the last iteration of the recorded metrics and parses them into a csv-friendly format
    Args:
        metrics: The recorded metrics to parse

    Returns: A dict of the recorded metrics where each entry now is the last recording and also parsed
    into a suitable format

    """
    last_results = {k: v[-1] for k, v in metrics.items()}
    for k, v in last_results.items():
        if isinstance(v, np.ndarray):
            last_results[k] = v.tolist()
        if isinstance(v, float):
            last_results[k] = float(v)
        if isinstance(v, (np.int32, np.int64)):
            last_results[k] = int(v)
    return last_results


def is_vigor(algorithm) -> bool:
    from algorithms.VIGOR import VIGOR
    from algorithms.DRex import DRex
    return isinstance(algorithm, VIGOR) and not isinstance(algorithm, DRex)
