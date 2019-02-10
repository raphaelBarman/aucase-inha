import shapely.geometry as geometry
import numpy as np
import gzip
import cv2
import os
import json

bib_num_url = "https://bibliotheque-numerique.inha.fr"


def bbox2geom(bbox):
    minx, miny = bbox.min(axis=0)
    maxx, maxy = bbox.max(axis=0)
    return geometry.box(minx, miny, maxx, maxy)


def gVert2opencv(vertices, scale_x=1, scale_y=1):
    return np.array([(x['x'] * scale_x, x['y'] * scale_y) for x in vertices
                     if 'x' in x and 'y' in x]).astype(int)


def gVert2geom(vertices, scale_x=1, scale_y=1):
    return bbox2geom(gVert2opencv(vertices, scale_x, scale_y))


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


def internal2url(internal, idx2inha):
    internal = internal.split('.')[0]
    doc_id = idx2inha[int(internal.split('_')[0])]
    page_num = int(internal.split('_')[1].rstrip('lr')) + 1
    query_url = "viewer/%d/?offset=#page=%d&viewer=picture&o=bookmarks&n=0&q="
    url = bib_num_url + query_url % (doc_id, page_num)
    return url


def get_google_ocr_bboxes(file_name, base_width, base_height, ocr_dir,
                          idx2cote):
    file_name = internal2inha(file_name, idx2cote)
    filepath = os.path.join(ocr_dir, file_name + '_ocr.json.gz')
    if not os.path.exists(filepath):
        print("Could not find", filepath)
        return np.array([]), 1, 1
    with gzip.GzipFile(filepath, 'r') as infile:
        json_bytes = infile.read()
    json_str = json_bytes.decode('utf-8')
    ocr = json.loads(json_str)

    bboxes = []
    if 'fullTextAnnotation' not in ocr:
        print("No text in", filepath)
        return bboxes, 1, 1
    page = ocr['fullTextAnnotation']['pages'][0]
    if len(ocr['fullTextAnnotation']['pages']) > 1:
        print("More than one page in", filepath)
        return bboxes, 1, 1

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
                        bboxes.append((bbox2geom(bbox), word))
    return np.array(bboxes), scale_x, scale_y


def words2string(words):
    res = ""
    for word in words:
        res += word2string(word)
    return res


def word2string(word):
    res = ""
    if 'symbols' not in word or len(word['symbols']) == 0:
        return res
    for symbol in word['symbols']:
        res += symbol['text']
        if 'property' in symbol and 'detectedBreak' in symbol['property']:
            break_type = symbol['property']['detectedBreak']['type']
            if break_type == 'SPACE' or break_type == 'EOL_SURE_SPACE':
                res += ' '
            elif break_type == 'LINE_BREAK':
                res += "\n"
            elif break_type == 'HYPHEN':
                pass


#                 res += '-\n'
            else:
                print("weird break", break_type)
                res += ' '
    return res


def overlap(bbox1, bbox2):
    biggest = bbox1
    smallest = bbox2
    biggest_area = bbox1.area
    smallest_area = bbox2.area
    if biggest_area < smallest_area:
        tmp = biggest
        biggest = smallest
        smallest = tmp
        tmp = biggest_area
        biggest_area = smallest_area
        smallest_area = tmp
    if smallest_area == 0:
        return 0
    return smallest.intersection(biggest).area / smallest_area


def get_overlap_values(poly, polys):
    #     poly = poly.buffer(0)
    return np.array([overlap(poly, p) for p in polys])


def get_best_matching_box(ocr_bbox, bboxes_sel, threshold=0.1):
    overlaps = get_overlap_values(ocr_bbox, bboxes_sel[:, 1])
    if max(overlaps) < threshold:
        return None
    return tuple(bboxes_sel[np.argmax(overlaps)].tolist())


def bbox_geom2key(bbox_geom):
    return (bbox_geom[0], str(bbox_geom[1]))


def page_content_from_boxes(bboxes_sel,
                            word_bboxes,
                            filtered=False,
                            image=None,
                            matching_threshold=0.5,
                            filling_threshold=5e-3):
    if len(bboxes_sel) == 0 or len(word_bboxes) == 0:
        return {}
    key2bbox_sel = {
        bbox_geom2key(bbox_sel): bbox_sel
        for bbox_sel in bboxes_sel
    }
    ocr_bboxes = np.array(word_bboxes)[:, 0]
    geom2content = {str(x[0]): x[1] for x in np.array(word_bboxes)}

    if filtered:
        gray_img = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ocr_bboxes_filtered = []
        _, thresh = cv2.threshold(gray_img, 130, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        for ocr_bbox in ocr_bboxes:
            word = word2string(geom2content[str(ocr_bbox)])
            minx, miny, maxx, maxy = np.array(ocr_bbox.bounds).astype(int)
            ratio = (thresh[miny:maxy, minx:maxx] / 255).mean() / len(word)
            if ratio > filling_threshold:
                ocr_bboxes_filtered.append(ocr_bbox)
        ocr_bboxes = ocr_bboxes_filtered

    page_content = {}
    for ocr_bbox in ocr_bboxes:
        best_match = get_best_matching_box(
            ocr_bbox, bboxes_sel, threshold=matching_threshold)
        if best_match is None:
            continue
        key = bbox_geom2key(best_match)
        if key not in page_content:
            page_content[key] = []
        page_content[key].append(geom2content[str(ocr_bbox)])
    page_content = sorted(
        [(key2bbox_sel[k], v) for k, v in page_content.items()],
        key=lambda x: x[0][1].bounds[-1])
    return page_content


def get_words_bbox(words, scale_x=1, scale_y=1):
    coords_bboxes = []
    for word in words:
        coords_bboxes.append(gVert2opencv(word['boundingBox']['vertices']))
    coords_bboxes = np.array(coords_bboxes).reshape(-1, 2) * [scale_x, scale_y]
    xmin, ymin = coords_bboxes.min(axis=0)
    xmax, ymax = coords_bboxes.max(axis=0)
    return xmin, ymin, xmax, ymax


def word2bbox(word, scale_x=1, scale_y=1):
    coords = gVert2opencv(word['boundingBox']['vertices'])
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    return xmin, ymin, xmax, ymax


def words2words_ref(words):
    words_ref = []
    for word in words:
        words_ref.append((word2bbox(word), word2string(word)))
    return np.array(words_ref)
