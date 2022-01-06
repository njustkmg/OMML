# 依赖文件下载

图文生成指标计算需要依赖下载相关依赖文件：
- 下载 paraphrase-en.gz [地址](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092/pycocoevalcap/meteor/data) ，将文件放在 metrics/caption/meteor/data 目录下
- stanford-corenlp jar包：
  - 下载 stanford-corenlp-full-2015-12-09.zip [地址](http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)
  - 解压，将文件夹中的 stanford-corenlp-3.6.0-models.jar 放在 metrics/caption/spice/lib 目录下