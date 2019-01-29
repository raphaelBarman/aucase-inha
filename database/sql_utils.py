import json


class Sql_utils:
    def __init__(self, idx2cote_path, idx2inha_path, inha_bib_num_metadata_path, iiif_manifests_path):
        with open(idx2cote_path, 'r') as infile:
            self.idx2cote = {int(k) : v for k,v in json.load(infile).items()}

        with open(idx2inha_path, 'r') as infile:
            self.idx2inha = {int(k): int(v) for k,v in json.load(infile).items()}

        with open(iiif_manifests_path, 'r') as infile:
            self.iiif_manifests = {int(k): v for k,v in json.load(infile).items()}

        with open(inha_bib_num_metadata_path, 'r') as infile:
            self.bib_num_metatada =  json.load(infile)

        self.base_doc_url = 'http://bibliotheque-numerique.inha.fr/idurl/1/'
        self.idx2base_url = lambda idx: self.base_doc_url + str(self.idx2inha[idx])

        
        self.inha2metadata = {int(x['identifier'][0].split('/')[-1]) : x for x in self.bib_num_metatada}
        self.idx2metadata = lambda idx: self.inha2metadata[self.idx2inha[idx]]
        self.idx2actors = lambda idx: self.idx2metadata(idx)['creator']

    def cote2date(self, cote):
        return cote.split('_')[-1]

    def idx2date_formatted(self, idx):
        date = self.cote2date(self.idx2cote[idx])
        return "-".join([date[:4], date[4:6], date[6:]])

    def basename2iiifbase(self, basename):
        doc_inha = self.idx2inha[int(basename.split('_')[0])]
        page = int(basename.split('_')[1].strip('lr'))
        if doc_inha not in self.iiif_manifests:
            return None
        manifest = self.iiif_manifests[doc_inha]
        if not 'sequences' in manifest and len(manifest['sequences'][0]['canvases']) <= page:
            return None
        return manifest['sequences'][0]['canvases'][page]['images'][0]['resource']['@id'].replace('/full/full/0/default.jpg', '')

    def get_section_infos(self, section, df_sections, parent=None):
        section_id = int(section.id)
        sale_id = section.doc
        page = section.page
        num_entity = section.entity
        class_ = section.class_
        (basename, text,
        xmin, ymin,
        xmax, ymax,
        inha_url) = df_sections.loc[section.doc, section.page, section.entity][[
            'basename', 'text', 'xmin', 'ymin', 'xmax', 'ymax', 'image_url'
        ]]
        iiif_url = self.basename2iiifbase(basename)
        parent_section_sale, parent_section_page, parent_section_entity = None, None, None
        if parent is not None:
            parent_section_sale, parent_section_page, parent_section_entity = parent.doc, parent.page, parent.entity
        return (sale_id, page, num_entity, parent_section_sale, parent_section_page, parent_section_entity , class_, text, (int(xmin), int(ymin), int(xmax), int(ymax)), inha_url, iiif_url)
