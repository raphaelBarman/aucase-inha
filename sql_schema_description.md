This is a description of the sql schema used for the database.

## Actor table
This table contains all the different actors that appear in the metada of the digital library of the INHA.

It has the following fields:
- actor_id: id of the actor
- first_name: first name of the actor
- last_name: last name of the actor
- role: role of the actor, can be one of "Commisaire-priseur", "Expert", "Commisaire-priseur, Expert", "Éditeur"
## Sale table
This table contains all the metadata one the auction catalogues which are sales.

It has the following fields:
- sale_id: id of the sale
- date: date of the sale
- cote_inha: classification number at INHA
- url_inha: url of the numerization of the catalogue
## Actor sale table
This table makes the link between the actors and the different sales they participated in.

It has the following fields:
- actor_id: id of the actor
- sale_id: id of the sale
## Section table
This table contains all the sections that were detected by the segmentation.

Sections characterize the object following them, for example by specifying
the type of the object ("painting") or the author of the object ("Picasso").

A section can be placed under another section, for example author sections are
most of the time placed under a category section (e.g. "Picasso" is placed under the "Painting" category).
This is represented in the table by having an internal (to the table) reference to the
other section that is the parent of the current section.

A section is not identified directly by an id, but rather by a tripled, which is composed of the sale it appears in, the page and the entity number in the page. For example, the section with the identifier (1787, 5, 4) is the fifth entity of the page six (there is an offset because we start to count at zero) of the sale number 1787.

It has the following fields:
- sale_id: id of the sale the section appears in
- page: page in which the section appears
- entity: entity number of the section in the page
- parent_section_sale: id of the sale of the parent section
- parent_section_page: page of the parent section
- parent_section_entity: entity number of the parent section
- class: class of the section, can be one of "author", "category", "ecole"
- text: text of the section
- bbox: (xmin, ymin, xmax, ymax) of the bounding box of the section
- inha_url: url of the page of the numerization
- iiif_url: iiif base url of the page, can be used with the bbox field to obtain a crop of the actual section
## Object table
This table contains all the objects that were detected by the segmentation.

It uses the same principle as in the section table for the identifier and the parent section.

It has the following fields:
- sale_id: id of the sale the object appears in
- page: page in which the object appears
- entity: entity number of the object in the page
- parent_section_sale: id of the sale of the parent section
- parent_section_page: page of the parent section
- parent_section_entity: entity number of the parent section
- num_ref: corrected number of reference of the object, not 100% accurate
- text: text of the object
- bbox: (xmin, ymin, xmax, ymax) of the bounding box of the object
- inha_url: url of the page of the numerization
- iiif_url: iiif base url of the page, can be used with the bbox field to obtain a crop of the actual object
