# Tamizhi-Net OCR: Creating A Quality Large Scale Tamil-Sinhala-English Parallel Corpus Using Deep Learning Based Printed Character Recognition (PCR)

![project] ![research]



- <b>Project Mentor</b>
    1. Dr.Uthayasanker Thayasivam
- <b>Contributor</b>
    1. Charangan Vasantharajan

---

## Summary

This research is about developing a simple, and automatic OCR engine that can extract text from  documents (with legacy fonts usage and printer-friendly encoding which are not optimized for text extraction) to create a parallel corpus. 

For this purpose, we enhanced the performance of Tesseract 4.1.1 by employing LSTM-based training on many legacy fonts to recognize printed characters in the above languages. Especially, our model detects code-mix text, numbers, and special characters from the printed document.

## Description

This project consists of the following.

- Dataset
- Model Training
- Model
- Improvements
- Corpus Creation

###Dataset
We created box files with coordinates specification, and then, we rectified misidentified characters, adjusted letter tracking, or spacing between characters to eliminate bounding box
overlapping issues using jTessBoxEditor.

<p align="center">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/jTessBoxEditor.png" width="600">
</p>

The following instructions will guide to generate TIFF/Box files. 

```
tesstrain.sh --fonts_dir data/fonts \
	     --fontlist \
	     --lang tam \    
	     --linedata_only \
		 --noextract_font_properties \
		 --training_text data/langdata/tam/tam.training_text \
	     --langdata_dir data/langdata \
	     --tessdata_dir data/tessdata \
	     --save_box_tiff \
	     --maxpages 100 \
	     --output_dir data/output
```


###Model Training
The table illustrates the command line flags used during the training. We have finalized the below
numbers after conducting several experiments with different values.

| Flag  | Value |
| ------------- | ------------- |
| traineddata  | path of traineddata file that contains the unicharset, word dawg, punctuation pattern dawg, number dawg  |
| model_output   | path of output model files /
checkpoints  |
| learning_rate  | 1e-05  |
| max_iterations  | 5000  |
| target_error_rate | 0.001 |
| continue_from  | path to previous checkpoint from which to continue training.  |
| stop_training  | convert the training checkpoint to full traineddata.  |
| train_listfile  | filename of a file listing training data files.  |
| eval_listfile  | filename of a file listing evaluating data files.  |


The following instructions will guide to start training.

```
OMP_THREAD_LIMIT=8 lstmtraining \
	--continue_from data/model/tam.lstm \
	--model_output data/finetuned_model/ \
	--traineddata data/tessdata/tam.traineddata \
	--train_listfile data/output/tam.training_files.txt \
	--eval_listfile data/output/tam.training_files.txt \
	--max_iterations 5000
```


###Model
The architecture of PCR is shown below. As the first step, we detect the file type and convert it to images if the input file is PDF. Then images are binarized and then image character boundary detection techniques are applied to find character boxes. Finally, deep learning modules identify word and line boundaries first then the characters are recognized. Finally using a language model, post-processing the file. 


<p align="center">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/model.png" width="600">
</p>

###Improvements
We compared the extracted text using our Tamizhi-Net Model with existing Tesseract below.

Tamil
<p align="center">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/subjective_tamil-original.png" width="600">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/subjective_tamil-tesseract.png" width="600">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/subjective_tamil-tamizhinet.png" width="600">
</p>

Sinhala
<p align="center">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/subjective_sinhala-original.png" width="600">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/subjective_sinhala-tesseract.png" width="600">
<img src="https://github.com/aaivu/aaivu-tamizhi-net-OCR/blob/master/docs/subjective_sinhala-tamizhinet.png" width="600">
</p>

###Corpus Creation
 To create a parallel corpus, we used [www.parliament.lk](www.parliament.lk) website to download the required PDFs of all three languages and feed them into our model to get extracted texts. 

###Publication
[Tamizhi-Net OCR: Creating A Quality Large Scale Tamil-Sinhala-English Parallel Corpus Using
Deep Learning Based Printed Character Recognition (PCR)](https://arxiv.org/pdf/2109.05952.pdf)

### Cite

```
@misc{vasantharajan2021tamizhinet,
      title={Tamizhi-Net OCR: Creating A Quality Large Scale Tamil-Sinhala-English Parallel Corpus Using Deep Learning Based Printed Character Recognition (PCR)}, 
      author={Charangan Vasantharajan and Uthayasanker Thayasivam},
      year={2021},
      eprint={2109.05952},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
