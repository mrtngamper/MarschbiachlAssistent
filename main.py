import argparse
import gc
import os
import shutil
from os.path import isfile, join, isdir
from pathlib import Path
import PIL
import PyPDF2
from PIL.ImageDraw import Draw
from PyPDF2 import PdfWriter, PdfReader
import cv2
import numpy as np
import psutil
from PIL import Image, ImageFont
from joblib import Parallel, delayed
from pdf2image import convert_from_path, pdfinfo_from_path

vertical = ["Böhmischer Traum", "Happy Birthday", "Graz"]
to_rotate = ["Böhmischer Traum", "Graz"]

parser = argparse.ArgumentParser(
    prog='Sheet Music Extractor',
    description='This program splits a list of pdfs into images and can then be used to auto rotate the images and upscale them',
    epilog='by Martin Gamper'
)

parser.add_argument('directory',
                    help='Input directory'
                    )
parser.add_argument('-m',
                    '--mode',
                    choices=["split", "align", "combine"],
                    required=True
                    )
parser.add_argument('-o',
                    '--output',
                    help='Output directory',
                    required=True
                    )
parser.add_argument('-p',
                    '--parts',
                    nargs='+',
                    help='Sets the parts which should be aligned or combined to an output file. '
                         'The files the align and combine commands will select are those which contain the names'
                         'provided with this option in their filename',
                    required=False
                    )

args = parser.parse_args()

print(args.directory, args.mode)

# Stimp net wrkl. ober is Seitenverhältnis passt
A5_HEIGHT = 420 * 12
A5_WIDTH = 595 * 12


def change_pdf_page_size(input_pdf_path, output_pdf_path):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    for page in reader.pages:
        page.scale_to(A5_WIDTH, A5_HEIGHT)
        writer.add_page(page)

    with open(output_pdf_path, "wb") as output_pdf:
        writer.write(output_pdf)


def split_file_to_images(destination_directory, file_path, subdir=""):
    os.makedirs(destination_directory, exist_ok=True)
    file_name = Path(file_path).stem

    if file_path.endswith(".pdf"):
        info = pdfinfo_from_path(file_path, userpw=None, poppler_path=None)
        pages = info["Pages"]
        stepsize = 2

        def convert_pdf(page, stepsize):
            print(f"{file_path}: page-{page}-{page + stepsize - 1}")
            images = convert_from_path(file_path, dpi=600, first_page=page, last_page=min(page + stepsize - 1, pages))
            for index in range(len(images)):
                images[index].save(
                    join(destination_directory,
                         f"{file_name}_{f"{subdir}_" if subdir != "" else ""}{page + index}.jpeg"),
                    'JPEG')
            del images
            gc.collect()

        Parallel(n_jobs=2)(
            delayed(convert_pdf)(page, stepsize) for page in range(1, pages + 1, stepsize))
    elif file_path.endswith(".tif"):
        im = Image.open(file_path)
        out = im.convert("RGB")
        out.save(join(destination_directory, f"{file_name}{f"_{subdir}" if subdir != "" else ""}.jpeg"))
        im.close()
    elif file_path.endswith(".jpg"):
        shutil.copy(file_path, join(destination_directory, f"{file_name}{f"_{subdir}" if subdir != "" else ""}.jpeg"))
    else:
        print("unsupported file format !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!: " + file_name)


def split_pdfs(input_dir: str, output_dir: str):
    files = [file for file in os.listdir(input_dir) if isfile(join(input_dir, file))]
    for file in files:
        split_file_to_images(join(output_dir, file), join(input_dir, file))

    sub_dirs = [sub_dir for sub_dir in os.listdir(input_dir) if isdir(join(input_dir, sub_dir))]
    for sub_dir in sub_dirs:
        for file in os.listdir(join(input_dir, sub_dir)):
            sub_dir_path = join(input_dir, sub_dir)
            sub_dir_file_path = join(sub_dir_path, file)

            if isdir(sub_dir_file_path):
                sub_dir_files = [f for f in os.listdir(sub_dir_file_path) if isfile(join(sub_dir_file_path, f))]
                for sub_dir_file in sub_dir_files:
                    split_file_to_images(join(output_dir, sub_dir), join(input_dir, sub_dir, file, sub_dir_file))
            else:
                split_file_to_images(join(output_dir, sub_dir), join(input_dir, sub_dir, file))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))
    return result


def fix_sheet_rotation(image, is_vert):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    (thresh, image_bw) = cv2.threshold(image_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    image_bw_filtered = cv2.erode(image_bw, np.ones((3, 3), np.uint8), iterations=1)
    image_bw_filtered_inverted = cv2.bitwise_not(image_bw_filtered)
    lines = cv2.HoughLinesP(image_bw_filtered_inverted, 1, np.pi / 180, 15, minLineLength=width / 5,
                            maxLineGap=10)

    if lines is not None:

        empty_image = np.zeros(image.shape[0:2], dtype=np.uint8)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(empty_image, (x1, y1), (x2, y2), (255), 1)

        empty_image = cv2.dilate(empty_image, np.ones((3, 3), np.uint8), iterations=5)
        empty_image = cv2.ximgproc.thinning(empty_image)
        if is_vert:
            empty_image = cv2.rotate(empty_image, cv2.ROTATE_90_CLOCKWISE)

        total_labels, labels = cv2.connectedComponents(empty_image, connectivity=8)
        angles = []
        weights = []
        for label in range(0, total_labels - 1):
            mask = (labels == label).astype(np.uint8) * 255
            mask = cv2.resize(mask, np.floor_divide(mask.shape, 4), interpolation=cv2.INTER_CUBIC)
            x, y, w, h = cv2.boundingRect(mask)
            point_list = cv2.findNonZero(mask)
            if point_list is None or len(point_list) < 3:
                continue
            vx, vy, x0, y0 = cv2.fitLine(point_list, cv2.DIST_L2, 0, 0.01, 0.01)
            angles.append(np.arctan2(vy, vx) * 180 / np.pi)
            weights.append(np.linalg.norm([w, h]))

        angles = np.array([angle if angle < 180 else angle - 360 for angle in angles]).squeeze()

        weights = np.array(weights)
        perm = np.argsort(angles)
        weights = weights[perm]
        angles = angles[perm]
        start_index = int(len(angles) * 0.25)
        end_index = int(len(angles) * 0.75)
        angles = angles[start_index:end_index]
        weights = weights[start_index:end_index]
        median_angle = np.average(angles, weights=np.array(weights))

        rotated_image = rotate_image(image, median_angle / 2)
        return median_angle / 2, rotated_image
    return None


def crop_image(image):
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, image_bw) = cv2.threshold(image_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image_bw_filtered = cv2.erode(image_bw, np.ones((3, 3), np.uint8), iterations=4)
    image_bw_filtered = cv2.dilate(image_bw_filtered, kernel=np.ones((5, 5)), iterations=5)

    values = np.column_stack(np.where(image_bw_filtered < 255))
    max_row = np.max(values[:, 0])
    min_row = np.min(values[:, 0])
    max_col = np.max(values[:, 1])
    min_col = np.min(values[:, 1])
    row_diff = max_row - min_row
    col_diff = max_col - min_col

    root2 = 2 ** (1 / 2)
    vertical_margin = (max_row - min_row) * 0.015
    horizontal_margin = (max_col - min_col) * 0.015

    new_height = row_diff + 2 * vertical_margin
    new_width = col_diff + 2 * horizontal_margin
    vertical_margin = max((new_width / root2 - row_diff) / 2, vertical_margin)
    horizontal_margin = max((new_height * root2 - col_diff) / 2, horizontal_margin)

    vertical_margin = int(vertical_margin)
    horizontal_margin = int(horizontal_margin)
    extended_image = cv2.copyMakeBorder(image, vertical_margin, vertical_margin, horizontal_margin,
                                        horizontal_margin, cv2.BORDER_CONSTANT, None, value=(255, 255, 255))
    cropped_image = extended_image[min_row:max_row + 2 * vertical_margin, min_col:max_col + 2 * horizontal_margin]

    return cropped_image


def show_image(image):
    window_name = 'edges'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 1000)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def normalize_angle(angle):
    if angle > 180:
        return angle - 360
    return angle


def align_image(image, is_vert, file_name, dir_name, output_dir):
    if Path(join(output_dir, dir_name, file_name) + ".jpeg").exists():
        return

    # rotate image
    median_angle = 90
    max_iterations = 10
    iteration = 0
    total_angle = 0
    original_image = image
    while abs(normalize_angle(median_angle)) > 0.02 and abs(
            normalize_angle(median_angle + 180)) > 0.02 and iteration < max_iterations:
        median_angle, image = fix_sheet_rotation(image, is_vert)
        total_angle += median_angle
        print(f"{dir_name} {file_name}, rotation step: {median_angle}, iteration: {iteration + 1}/{max_iterations}")
        iteration += 1
    image = rotate_image(original_image, total_angle)

    # Crop image
    image = crop_image(image)
    # write image to file
    os.makedirs(join(output_dir, dir_name), exist_ok=True)
    cv2.imwrite(join(output_dir, dir_name, file_name) + ".jpeg", image)


def align_images(input_dir: str, output_dir: str, parts: list[str]):
    jpeg_dirs = [join(input_dir, file) for file in os.listdir(input_dir) if isdir(join(input_dir, file))]

    def process_dir(jpeg_dir: str):
        jpegs = [join(jpeg_dir, file) for file in os.listdir(jpeg_dir) if
                 isfile(join(jpeg_dir, file)) and file.endswith('.jpeg')]

        def process_image(jpeg: str):
            if parts is not None:
                contained = False
                for part in parts:
                    if jpeg.__contains__(part):
                        contained = True
                        break
                if not contained:
                    return
            print(jpeg)

            is_vert = False
            for vert in vertical:
                if jpeg.__contains__(vert):
                    is_vert = True
                    break

            image = cv2.imread(jpeg)
            for rotate in to_rotate:
                if jpeg.__contains__(rotate):
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    break

            align_image(image, is_vert, Path(jpeg).stem, Path(jpeg_dir).stem, output_dir)

        for jpeg in jpegs:
            process_image(jpeg)

    Parallel(n_jobs=psutil.cpu_count(logical=True))(delayed(process_dir)(jpeg_dir) for jpeg_dir in jpeg_dirs)


def combine_images(input_dir: str, output_dir: str, part: str):
    jpeg_dirs = [join(input_dir, file) for file in os.listdir(input_dir) if isdir(join(input_dir, file))]
    jpeg_dirs.sort()
    images = []

    chapter_titles = []

    for jpeg_dir in jpeg_dirs:
        jpegs = [join(jpeg_dir, file) for file in os.listdir(jpeg_dir) if
                 isfile(join(jpeg_dir, file)) and file.endswith('.jpeg') and file.__contains__(part)]
        jpegs.sort()

        chapter_title = os.path.basename(jpeg_dir)
        chapter_title = chapter_title.replace('_', ' ')


        if len(jpegs) == 0:
            image = PIL.Image.new(mode="RGB", size=(A5_WIDTH, A5_HEIGHT))
            I1 = Draw(image)
            I1.text((50, A5_HEIGHT / 2), "Missing part for: " + jpeg_dir,
                    font=ImageFont.truetype('LiberationSerif-Regular.ttf', 200))
            images.append(image)
            chapter_titles.append(chapter_title)
        for jpeg in jpegs:
            cv_image = cv2.imread(jpeg)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            cv_image = 255 - cv_image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cv_image = clahe.apply(cv_image)
            cv_image, thres = cv2.threshold(cv_image, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

            non_zero_pixels = thres[thres > 0]
            non_zero_pixels.sort()
            median_value = non_zero_pixels[len(non_zero_pixels)*9 // 10]
            large_thres = thres.astype(np.uint32)
            thres[thres > 0] = np.clip(large_thres[ large_thres>0] + 255 - median_value, 0, 255).astype(np.uint8)

            thres = 255 - thres

            image = Image.fromarray(thres)
            images.append(image)
            chapter_titles.append(chapter_title)

    os.makedirs(output_dir, exist_ok=True)
    output_file = join(output_dir, part) + ".pdf"
    images[0].save(
        output_file, "PDF", resolution=100.0, save_all=True, append_images=images[1:]
    )


    change_pdf_page_size(output_file, output_file)

    # Add bookmarks
    pdf_reader = PyPDF2.PdfReader(output_file)
    pdf_writer = PyPDF2.PdfWriter()

    for page in pdf_reader.pages:
        pdf_writer.add_page(page)

    for page, title in enumerate(chapter_titles):
        pdf_writer.add_outline_item(title, page)

    with open(output_file, 'wb') as output_pdf_file:
        pdf_writer.write(output_pdf_file)

    pass


if args.mode == "split":
    split_pdfs(args.directory, args.output)
elif args.mode == "align":
    align_images(args.directory, args.output, args.parts)
elif args.mode == "combine":
    if args.parts is None:
        print("No parts provided")
        exit(1)
    for part in args.parts:
        combine_images(args.directory, args.output, part)
