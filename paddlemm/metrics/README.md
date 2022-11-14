# Dependent file download

Step 1. Install JAVA by ```apt install openjdk-8-jdk```

Step 2. The calculation of image caption requires downloading related dependent files:
- Download paraphrase-en.gz [here](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092/pycocoevalcap/meteor/data) , put the file in the metrics/caption/meteor/data directory
- stanford-corenlp jar package：
  - Download stanford-corenlp-full-2015-12-09.zip [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)
  - Unzip, put the stanford-corenlp-3.6.0-models.jar in the folder under the metrics/caption/spice/lib directory