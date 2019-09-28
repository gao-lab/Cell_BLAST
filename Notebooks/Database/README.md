# README

The pipeline trains models for ACA datasets

* Datasets of the same organ are grouped in a single jupyter notebook,
  which is responsible for model training, data visualizations, etc.
* Generated results are compressed to tarballs and ready to be transferred to the website server.
* Meta tables for datasets and cells in each dataset are turned into a MySQL database dump,
  which can then be loaded on the website server.
