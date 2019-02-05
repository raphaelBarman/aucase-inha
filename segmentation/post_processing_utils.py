import json
import gzip
import os
import numpy as np
import shapely.geometry as geometry
import cv2

CLASSES = ['section_author', 'sale_description']


# Full pipeline
def get_bboxes_sel(file_name,
                   gray_img,
                   prediction_data,
                   bin_thresh_content=0.5,
                   bin_thresh_section=0.15,
                   ksize_w=0,
                   ksize_h=0,
                   line_padding=30,
                   crop_padding=1,
                   indent_thresh=0.8,
                   contour_area_thresh=100,
                   area_thresh=1000,
                   ocr_dir=None,
                   use_inha_ocr=True,
                   idx2cote=None):

    use_ocr = ocr_dir is not None and (use_inha_ocr or idx2cote is not None)

    probs, original_shape = prediction_data[file_name.replace('.jpg', '')]
    if use_ocr:
        if use_inha_ocr:
            ocr_bboxes = get_inha_ocr_bboxes(file_name, ocr_dir)
        else:
            height, width = gray_img.shape
            ocr_bboxes = get_google_ocr_bboxes(file_name, width, height,
                                               ocr_dir, idx2cote)
        ocr_bboxes_geom = bboxes2geom(ocr_bboxes)
        lines_geom = []
        if len(ocr_bboxes) > 0:
            lines_description, lines_bboxes = get_lines_and_bboxes(ocr_bboxes)
            lines_geom = lines2geom(lines_description)

    bboxes_sel = []

    probs = cv2.resize(
        probs, tuple(original_shape[::-1]), interpolation=cv2.INTER_NEAREST)
    probs[:, :, 0][(probs < bin_thresh_section)[:, :, 0]] = -1
    probs[:, :, 1][(probs < bin_thresh_content)[:, :, 1]] = -1
    classes = np.argmax(probs, axis=2)
    for class_idx in range(2):
        mask = (classes == class_idx).astype(np.uint8) * 255
        ksize_open = (ksize_w, ksize_h)
        mask = cv2.dilate(mask, np.ones(ksize_open))
        mask = filter_mask_by_area(mask, contour_area_thresh)
        if use_ocr:
            mask = complete_with_lines(mask, lines_geom, padding=line_padding)
        mask = cut_idents_with_context(
            mask,
            indent_threshold=indent_thresh,
            area_threshold=area_thresh,
            crop_padding=crop_padding)
        if use_ocr:
            mask = complete_with_bboxes(mask, ocr_bboxes_geom)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

        contours = [
            contour for contour in contours
            if cv2.contourArea(contour) > area_thresh
        ]

        bboxes_sel.extend(
            get_content_aware_bboxes(file_name, contours, gray_img,
                                     CLASSES[class_idx], area_thresh))

    return np.array(bboxes_sel)


def get_bboxes_sel_fastrcnn(file_name,
                            gray_img,
                            prediction_data,
                            proba_threshold_section=0.5,
                            line_padding_section=5,
                            complete_with_lines_section=1,
                            complete_with_bboxes_section=1,
                            proba_threshold_sale=0.5,
                            line_padding_sale=5,
                            complete_with_lines_sale=1,
                            complete_with_bboxes_sale=1,
                            ocr_dir=None,
                            use_inha_ocr=True,
                            idx2cote=None):

    use_ocr = ocr_dir is not None and (use_inha_ocr or idx2cote is not None)

    h, w = gray_img.shape
    pred_data = prediction_data[file_name.replace('.jpg', '')]

    classes = pred_data.get('detection_classes')
    section_idx = classes == 1
    sale_idx = classes == 2

    probas_section = pred_data.get('detection_scores')[section_idx]
    boxes_section = (pred_data.get('detection_boxes') * [h, w, h, w]).astype(
        int)[section_idx][probas_section > proba_threshold_section]
    bboxes_geom_section = predboxes2geom(boxes_section)

    probas_sale = pred_data.get('detection_scores')[sale_idx]
    boxes_sale = (pred_data.get('detection_boxes') * [h, w, h, w]
                  ).astype(int)[sale_idx][probas_sale > proba_threshold_sale]
    bboxes_geom_sale = predboxes2geom(boxes_sale)

    if (complete_with_lines_section or complete_with_bboxes_section or
            complete_with_lines_sale or complete_with_bboxes_sale) and use_ocr:
        if use_inha_ocr:
            ocr_bboxes = get_inha_ocr_bboxes(file_name, ocr_dir)
        else:
            height, width = gray_img.shape
            ocr_bboxes = get_google_ocr_bboxes(file_name, width, height,
                                               ocr_dir, idx2cote)
        ocr_bboxes_geom = bboxes2geom(ocr_bboxes)
    else:
        ocr_bboxes = []
    lines_geom = []
    if len(ocr_bboxes) > 0:
        lines_description, lines_bboxes = get_lines_and_bboxes(ocr_bboxes)
        lines_geom = lines2geom(lines_description)

    if use_ocr:
        if complete_with_lines_section and len(bboxes_geom_section) > 0:
            bboxes_geom_section = complete_geoms_with_geoms(
                bboxes_geom_section, lines_geom, line_padding_section, 0)
        if complete_with_bboxes_section and len(bboxes_geom_section) > 0:
            bboxes_geom_section = complete_geoms_with_geoms(
                bboxes_geom_section, ocr_bboxes_geom, 0, 0)

        if complete_with_lines_sale and len(bboxes_geom_sale) > 0:
            bboxes_geom_sale = complete_geoms_with_geoms(
                bboxes_geom_sale, lines_geom, line_padding_sale, 0)
        if complete_with_bboxes_sale and len(bboxes_geom_sale) > 0:
            bboxes_geom_sale = complete_geoms_with_geoms(
                bboxes_geom_sale, ocr_bboxes_geom, 0, 0)

    bboxes_sel = []
    for bbox in bboxes_geom_section:
        bboxes_sel.append(('section_author', bbox))
    for bbox in bboxes_geom_sale:
        bboxes_sel.append(('sale_description', bbox))
    return np.array(bboxes_sel)


# Mask operations
def filter_mask_by_area(mask, area):
    mask_filtered = np.zeros_like(mask)
    cnt, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
    [
        cv2.drawContours(mask_filtered, [c], -1, 255, -1) for c in contours
        if cv2.contourArea(c) > area
    ]
    return mask_filtered


def complete_with_lines(mask, lines_geom, padding=5, copy=False):
    if copy:
        mask = mask.copy()
    cnt, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour2geom(contour)
        for box in get_intersecting(contour, lines_geom):
            minx, miny, maxx, maxy = np.array(box.bounds).astype(int)
            miny -= padding
            maxy += padding
            cv2.rectangle(mask, (minx, miny), (maxx, maxy), 255, -1)
    return mask


def cut_idents_with_context(mask,
                            indent_threshold=0.1,
                            area_threshold=200,
                            crop_padding=10,
                            copy=False):
    if copy:
        mask = mask.copy()
    cnt, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
    contours = [
        contour for contour in contours
        if cv2.contourArea(contour) > area_threshold
    ]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= crop_padding * 2:
            continue
        minx, maxx = x + crop_padding, x + w - crop_padding
        miny, maxy = y, y+h
        crop = mask[miny:maxy, minx:maxx] // 255
        top_crop, bottom_crop = crop[:, 0].argmax(), crop[:, 0][::-1].argmax()
        crop = mask[miny + top_crop:maxy - bottom_crop, minx:maxx] // 255
        mask[miny + top_crop + get_indent_lines(
            crop, indent_threshold)] = np.zeros(mask.shape[1])
    return mask


def complete_with_bboxes(mask, bboxes_geom, copy=False):
    if copy:
        mask = mask.copy()
    cnt, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 2)
    for contour in contours:
        if len(contour) < 3:
            continue
        contour = contour2geom(contour)
        for box in get_intersecting(contour, bboxes_geom):
            minx, miny, maxx, maxy = np.array(box.exterior.bounds).astype(int)
            cv2.rectangle(mask, (minx, miny), (maxx, maxy), 255, -1)
    return mask


def cut_at_idents(mask, indent_lines, lines, padding=4, copy=False):
    if copy:
        mask = mask.copy()
    lines = lines[:, 0]
    to_remove = set()
    for line in indent_lines:
        if (line < lines.min() or line > lines.max()):
            continue
        signs = (np.sign(lines - line) + 1) // 2
        idx = np.abs(signs[:-1] - signs[1:]).argmax()
        to_remove.add((lines[idx] + lines[idx + 1]) // 2)
    to_remove = np.array(list(to_remove))
    to_remove = np.hstack([
        to_remove,
        np.array([(to_remove - i, to_remove + i)
                  for i in range(padding)]).reshape(-1)
    ])
    mask[to_remove] = np.zeros(mask.shape[1])
    return mask


def get_content_aware_bboxes(file_name,
                             contours,
                             gray_img,
                             class_,
                             area_threshold=200):
    bboxes = []
    image = gray_img
    for contour in contours:
        mask_content = np.zeros(image.shape, dtype=image.dtype)
        cv2.drawContours(mask_content, [contour], -1, 255, -1)
        img = cv2.bitwise_and(mask_content, image)
        ret, thresh = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > area_threshold:
            bboxes.append((class_, geometry.box(x, y, x + w, y + h)))
    return bboxes


def get_lines_and_bboxes(bboxes, line_threshold=5):
    """
    Gives for a set of bbox the corresponding lines
    can tweak the line threshold
    returns a sorted list
    """
    diffs = np.abs((bboxes[:, :, 1].mean(axis=1)[::-1][:-1] -
                    bboxes[:, :, 1].mean(axis=1)[::-1][1:]))
    diffs[diffs < line_threshold] = 0
    mask_diffs = diffs > 0
    lines = np.array([
        x.mean() for x in np.split(bboxes[:, :, 1].mean(axis=1)[::-1],
                                   np.where(mask_diffs)[0] + 1)
    ]).astype(int)
    xs_splitted = np.split(bboxes[:, :, 0][::-1], np.where(mask_diffs)[0] + 1)
    start_xs = np.array([x.min() for x in xs_splitted]).astype(int)
    end_xs = np.array([x.max() for x in xs_splitted]).astype(int)
    return np.array(
        sorted(
            np.stack([lines, start_xs, end_xs], axis=-1),
            key=lambda x: x[0])), np.split(bboxes[::-1],
                                           np.where(mask_diffs)[0] + 1)[::-1]


# Misc
def get_indent_lines(mask, indent_threshold=0.1):
    mean_column = (mask // 255).sum(axis=0)
    crop_left, crop_right = (
        mean_column >
        0).argmax(), len(mean_column) - (mean_column > 0)[::-1].argmax()
    indent_lines = np.where((mask[:, crop_left:crop_right].argmax(axis=1) /
                             (crop_right - crop_left)) > indent_threshold)[0]
    return np.sort(indent_lines)


def complete_geoms_with_geoms(geoms, other_geoms, padding_v=5, padding_h=0):
    if len(other_geoms) <= 0:
        return geoms
    geoms_res = []
    for geom in geoms:
        geom_minx, geom_miny, geom_maxx, geom_maxy = np.array(
            geom.bounds).astype(int)
        for box in get_intersecting(geom, other_geoms):
            minx, miny, maxx, maxy = np.array(box.bounds).astype(int)
            miny -= padding_v
            maxy += padding_v
            minx -= padding_h
            maxx += padding_h
            if minx < geom_minx:
                geom_minx = minx
            if miny < geom_miny:
                geom_miny = miny
            if maxx > geom_maxx:
                geom_maxx = maxx
            if maxy > geom_maxy:
                geom_maxy = maxy
        geoms_res.append(
            geometry.box(geom_minx, geom_miny, geom_maxx, geom_maxy))
    return geoms_res


# Loading
def get_lines(bboxes, line_threshold=5):
    """
    Gives for a set of bbox the corresponding lines
    can tweak the line threshold
    returns a sorted list
    """
    diffs = np.abs((bboxes[:, :, 1].mean(axis=1)[::-1][:-1] -
                    bboxes[:, :, 1].mean(axis=1)[::-1][1:]))
    diffs[diffs < line_threshold] = 0
    mask_diffs = diffs > 0
    lines = np.array([
        x.mean() for x in np.split(bboxes[:, :, 1].mean(axis=1)[::-1],
                                   np.where(mask_diffs)[0] + 1)
    ]).astype(int)
    xs_splitted = np.split(bboxes[:, :, 0][::-1], np.where(mask_diffs)[0] + 1)
    start_xs = np.array([x.min() for x in xs_splitted]).astype(int)
    end_xs = np.array([x.max() for x in xs_splitted]).astype(int)
    return np.array(
        sorted(
            np.stack([lines, start_xs, end_xs], axis=-1), key=lambda x: x[0]))


def get_google_ocr_bboxes(file_name, base_width, base_height, ocr_dir,
                          idx2cote):
    file_name = internal2inha(file_name, idx2cote)
    filepath = os.path.join(ocr_dir, file_name + '_ocr.json.gz')
    if not os.path.exists(filepath):
        return np.array()
    with gzip.GzipFile(filepath, 'r') as infile:
        json_bytes = infile.read()
    json_str = json_bytes.decode('utf-8')
    ocr = json.loads(json_str)

    bboxes = []

    page = ocr['fullTextAnnotation']['pages'][0]
    if len(ocr['fullTextAnnotation']['pages']) > 1:
        return bboxes

    ocr_width = page['width']
    ocr_height = page['height']
    scale_x = base_width / ocr_width
    scale_y = base_height / ocr_height

    for page in ocr['fullTextAnnotation']['pages']:
        for block in page['blocks']:
            for paragraph in block['paragraphs']:
                for word in paragraph['words']:
                    bbox = gVert2opencv(word['boundingBox']['vertices'],
                                        scale_x, scale_y)
                    if bbox.shape == (4, 2):
                        bboxes.append(bbox)
    return np.array(bboxes)


def get_inha_ocr_bboxes(file_name,
                        ocr_dir='/scratch/raphael/data/ocr/ocr_inha/'):
    filepath = os.path.join(ocr_dir,
                            file_name.replace('.jpg', '') + '_ocr.json.gz')
    if not os.path.exists(filepath):
        print(filepath, 'not found')
        return np.array([])
    with gzip.GzipFile(filepath, 'r') as infile:
        json_bytes = infile.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    bboxes = []
    for block in data['textblocks']:
        for line in block['CONTENT']:
            for string in line['CONTENT']:
                if string['TYPE'] == 'String':
                    bboxes.append(alto2coords(string))
    return np.array(bboxes)


def get_not_intersecting(bboxes):
    bboxes = sorted(bboxes, key=lambda x: -x.area)

    bboxes_sel = []
    for box in bboxes:
        intersecting = get_intersecting(box, bboxes)
        max_area = max([x.area for x in intersecting])
        if box.area >= max_area:
            bboxes_sel.append(box)
    return np.array(bboxes_sel)


def get_not_intersecting_with_class(bboxes):
    bboxes = sorted(bboxes, key=lambda x: -x[1].area)
    sector_bboxes = [box[1] for box in bboxes if box[0] == 'section_author']
    description_bboxes = [
        box[1] for box in bboxes if box[0] == 'sale_description'
    ]

    sector_bboxes_sel = list(get_not_intersecting(sector_bboxes))
    if len(sector_bboxes_sel) > 0:
        sector_bboxes_sel = [('section_author', x) for x in sector_bboxes_sel]
    description_bboxes_sel = list(get_not_intersecting(description_bboxes))
    if len(description_bboxes_sel) > 0:
        description_bboxes_sel = [('sale_description', x)
                                  for x in description_bboxes_sel]
    return np.array(sector_bboxes_sel + description_bboxes_sel)


# Conversion
def alto2coords(alto):
    x = int(alto['HPOS'])
    y = int(alto['VPOS'])
    w = int(alto['WIDTH'])
    h = int(alto['HEIGHT'])
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x,
                                                          y + h]]).astype(int)


def bbox2geom(bbox):
    minx, miny = bbox.min(axis=0)
    maxx, maxy = bbox.max(axis=0)
    return geometry.box(minx, miny, maxx, maxy)


def bboxes2geom(bboxes):
    return [bbox2geom(bbox) for bbox in bboxes]


def predbox2geom(bbox):
    miny, minx, maxy, maxx = bbox
    return geometry.box(minx, miny, maxx, maxy)


def predboxes2geom(bboxes):
    return [predbox2geom(bbox) for bbox in bboxes]


def line2geom(line):
    y, start_x, end_x = line
    return geometry.LineString([(start_x, y), (end_x, y)])


def lines2geom(lines):
    return [line2geom(line) for line in lines]


def contour2geom(contour):
    return geometry.Polygon(np.squeeze(contour))


def get_intersecting(poly, polys):
    intersects = []
    poly = poly.buffer(0)
    for p in polys:
        # p = p.buffer(0)

        if p.intersects(poly):
            intersects.append(p)
    return intersects


def gVert2opencv(vertices, scale_x=1, scale_y=1):
    return np.array([(x['x'] * scale_x, x['y'] * scale_y) for x in vertices
                     if 'x' in x and 'y' in x]).astype(int)


def internal2inha(internal, idx2cote):
    internal = internal.split('.')[0]
    doc_id = int(internal.split('_')[0])
    page_num = int(internal.split('_')[1].rstrip('lr')) + 1
    suffix = ""
    if 'r' in internal.split('_')[1]:
        suffix = "r"
    elif 'l' in internal.split('_')[1]:
        suffix = 'l'
    if doc_id not in idx2cote:
        print("Did not find", doc_id)
    else:
        return 'B751025206%s_%03d%s' % (idx2cote[doc_id], page_num, suffix)
