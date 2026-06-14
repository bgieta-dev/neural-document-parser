import cv2
import fitz
import numpy as np
import math


class ImageProcessor:
    def __init__(self, path, DEBUG_MODE=False):
        self.path = path
        self.debug_mode = DEBUG_MODE
        self.debug_print(f"{20*'-'} {path} {20*'-'}")
        self.load_scan()
        self.orientation_check()
        self.fix_angle()
        self.isolate_table()
        self.extract_cells()

    def debug_print(self, s):
        if self.debug_mode:
            print(s)

    def save_debug(self, name, img):
        if self.debug_mode:
            cv2.imwrite(f"debug/debug_{name}.png", img)

    def pdf_to_numpy(self, path):
        doc = fitz.open(path)
        page = doc[0]
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.img = img

    def load_scan(self):
        if self.path[-3:].lower() == "pdf":
            self.pdf_to_numpy(self.path)
        else:
            self.img = cv2.imread(self.path)

    def get_centers_of_markers(self, markers):
        valid = []
        centers = []
        for marker in markers:
            x, y, w, h = cv2.boundingRect(marker)
            aspect_ratio = float(w) / float(h)

            is_square = 0.8 < aspect_ratio < 1.2
            if is_square:
                try:
                    valid.append(marker)
                    cX = x + w // 2
                    cY = y + h // 2
                    centers.append((cX, cY))
                except:
                    self.debug_print(f"[ERROR] in appending centers: ({cX}, {cY})")
        self.debug_print(centers)
        img = self.img.copy()
        cv2.drawContours(img, valid, -1, (0, 255, 0), 3)
        self.save_debug("contour", img)
        return centers

    def get_contours(self, img, table=False):
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(self.gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        return contours

    def orientation_check(self):
        height, width, _ = self.img.shape
        if height > width:
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
            height, width = width, height
            self.debug_print("rotated1")
            self.save_debug("rotate1", self.img)
        self.debug_print(f"{height}, {width}")
        self.height, self.width = height, width
        centers = self.get_centers_of_markers(self.get_contours(self.img))

        middle_marker = min(centers, key=lambda c: abs(c[0] - width / 2))
        self.debug_print(f"Middle marker: {middle_marker}")
        if middle_marker[1] > self.height:
            self.debug_print("rotated2")
            self.img = cv2.rotate(self.img, cv2.ROTATE_180)
            self.save_debug("rotate2", self.img)
            centers = self.get_centers_of_markers(self.get_contours(self.img))
        self.save_debug("AFTER ROTATIONS", self.img)
        self.markers = sorted(centers, key=lambda x: x[1], reverse=True)
        self.debug_print(f"sorted: {self.markers}")

    def fix_angle(self):
        try:
            markers = [self.markers[1], self.markers[0]]
            markers = sorted(markers, key=lambda x: x[0])
            self.debug_print(f"sorted in angles func: {markers}")
            dy = markers[1][1] - markers[0][1]
            dx = markers[1][0] - markers[0][0]
            angle = math.degrees(math.atan2(dy, dx))
            self.debug_print(f"angle: {angle}")
            center = (self.width // 2, self.height // 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            self.img = cv2.warpAffine(
                self.img,
                M,
                (self.width, self.height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            self.save_debug("fix_angle", self.img)
        except:
            print("[ERROR] in fix_angle function")

    def isolate_table(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.adaptiveThreshold(
            self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.width // 15, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.height // 15))

        hor_lines = cv2.erode(self.thresh, hor_kernel, iterations=1)
        hor_lines = cv2.dilate(hor_lines, hor_kernel, iterations=1)

        ver_lines = cv2.erode(self.thresh, ver_kernel, iterations=1)
        ver_lines = cv2.dilate(ver_lines, ver_kernel, iterations=1)

        table_skeleton = cv2.add(hor_lines, ver_lines)
        kernel_thick = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.table_skeleton = cv2.dilate(table_skeleton, kernel_thick, iterations=1)

        self.save_debug("table_mask", table_skeleton)

    def extract_cells(self):
        contours, _ = cv2.findContours(
            self.table_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        debug_img = self.img.copy()
        i = 1
        cells = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5000 > area > 1000:
                cells.append(cv2.boundingRect(cnt))
        cells = sorted(cells, key=lambda b: b[1])

        rows = []
        if cells:
            current_row = [cells[0]]
            for i in range(1, len(cells)):
                if abs(cells[i][1] - current_row[-1][1]) < 20:
                    current_row.append(cells[i])
                else:
                    if len(current_row) == 10:
                        rows.append(current_row)
                    current_row = [cells[i]]
            rows.append(current_row)

        final_cells = []
        for row in rows:
            sorted_row = sorted(row, key=lambda b: b[0])
            if len(sorted_row) > 1:
                final_cells.extend(sorted_row)
            else:
                pass
        self.final_cells = final_cells
        for i, (x, y, w, h) in enumerate(final_cells):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(debug_img, str(i+1), (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        self.save_debug("contour", debug_img)

    def clean_cell(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray

    def prepare_data_for_cnn(self, cell_rescaled):
        micro_pad = 5
        cell_bin = cv2.adaptiveThreshold(
            src=cell_rescaled,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=31,  # blockSize%2==1
            C=7,
        )  # Noise filter const

        cell_contour, _ = cv2.findContours(
            image=cell_bin, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )
        if len(cell_contour) > 0:
            contour_list = []
            contour_area_list = []
            isNumber = True
            for cnt in cell_contour:
                cell_contour_area = cv2.contourArea(cnt)
                if cell_contour_area > 100:
                    contour_area_list.append(cell_contour_area)
                    contour_list.append(cnt)

            if len(contour_list) >= 1:
                contour_contected = np.vstack((contour_list))
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour_contected)

                cy1 = max(0, y_c - micro_pad)
                cy2 = min(cell_bin.shape[0], y_c + h_c + micro_pad)
                cx1 = max(0, x_c - micro_pad)
                cx2 = min(cell_bin.shape[1], x_c + w_c + micro_pad)

                cv2.rectangle(
                    img=cell_rescaled,
                    pt1=(x_c, y_c),
                    pt2=(x_c + w_c, y_c + h_c),
                    color=(0, 0, 255),
                )

                crop = cell_bin[cy1:cy2, cx1:cx2]

                h_crop, w_crop = crop.shape[:2]
                scale = 128 / max(h_crop, w_crop)
                new_w, new_h = int(w_crop * scale), int(h_crop * scale)

                resized = cv2.resize(crop, (new_w, new_h))
                canvas = np.zeros((128, 128), dtype=np.uint8)

                x_offset = (128 - new_w) // 2
                y_offset = (128 - new_h) // 2

                canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
                    resized
                )

                new_w, new_h = int(w_crop * scale), int(h_crop * scale)

                return canvas

    def get_cell_images(self):
        cell_images = {1: [], 2: [], 3: []}

        for i, (x, y, w, h) in enumerate(self.final_cells):
            padding = -1
            cell = self.img[
                y - padding : y + h + padding, x - padding : x + w + padding
            ]
            cell_rescaled = cv2.resize(
                cell, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC
            )
            cell_rescaled = self.clean_cell(cell_rescaled)

            canvas = self.prepare_data_for_cnn(cell_rescaled)
            if i == 20:
                self.save_debug("cell_rescaled", cell_rescaled)
                if canvas is not None:
                    self.save_debug("cell_canvas", canvas)

            if canvas is not None:
                # self.debug_print(f"{i}")
                cell_images[i // 60 + 1].append(canvas)
        return cell_images

    def get_cell_images_data_getter(self):
        path = self.path[self.path.rfind("/") + 1 : -4]
        data = []
        for i, (x, y, w, h) in enumerate(self.final_cells):
            padding = -1
            cell = self.img[
                y - padding : y + h + padding, x - padding : x + w + padding
            ]
            cell_rescaled = cv2.resize(
                cell, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC
            )
            cell_rescaled = self.clean_cell(cell_rescaled)

            canvas = self.prepare_data_for_cnn(cell_rescaled)
            if i == 93:
                self.save_debug("cell_rescaled", cell_rescaled)
                if canvas is not None:
                    self.save_debug("cell_canvas", canvas)

            if canvas is not None:
                # self.debug_print(f"{path}, {i//60+1}, {i}")
                data.append((canvas, path, i // 60 + 1, i))
        return data
