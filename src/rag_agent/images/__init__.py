"""Image pipeline (ТЗ S4, Э5): images of the corpus become searchable nodes.

1. Every image is found at parse time: a .png / .jpg file, an image output of a notebook cell, a picture
   in a DOCX or a PDF (``sources``); its node carries where to read the pixels from.
2. After indexing, each image is classified by type — chart, diagram, screenshot, scan, photo — by a small
   fine-tuned CNN (``classifier``, scripts/train_image_classifier.py).
3. By type: a chart is translated into a table plus a description of its axes and trend (the DePlot idea,
   done by the VLM), a diagram is described block by block, a screenshot or scan is read by OCR and
   summarised; the text becomes the node's text and is embedded like any other node (``describe``).
4. Optionally a parallel visual index: ColQwen2 multi-vector embeddings of the images and PDF pages,
   searched by late interaction and fused with the text index by RRF (``visual``, H4).

Descriptions are cached by the image's content hash and the models used, so re-indexing is cheap.
"""
