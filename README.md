[**Version française**](https://github.com/sriak/aucase-inha/blob/master/LISEZMOI.md)

# General description
Auction Catalog Segmentation (AuCaSe) is a project realised by Raphaël Barman at
the [Institut national d'histoire de l'art (INHA)](https://www.inha.fr) during an
internship as part of his Msc degree of Digital Humanities at the
[École polytechnique fédérale de Lausanne (EPFL)](https://www.epfl.ch).

The goal of the project was to use the numerization of auction catalogues owned by
the INHA in order to create a database of object sales with their metadata, through
a process of image segmenation using deep learning.

The project managed to extract from 1911 auction catalogues published by the auction
house Drouot between 1st January 1939 and 31st Decembre 1945 more than 300'000 objects
made by more than 5'000 artists for sale at auctions supervised by more than 200 auctionner and experts.

This data will be free to explore and download in the future.

# Repository structure

The main entry point to this repository is the "Notebook annoté" file (in french) which
contains all the steps of the pipeline from downloading the numerizations to importing
the data into a sql database.

Technically, by modifying the config.toml file and running the main.py file, the whole pipeline should run.
However, it still needs some steps to be achieved before running it, such as training or getting different models.
In general, it would be a good idea to comment out most of the step of the main.py and run them one by one.
The whole pipeline can take days or even a week or two (without GPU) to complete.
