# CS 254 Machine Learning
## Project Short-Proposal
### Thompson, Gamble, Cafiero
 
### Introduction

*Motivate and abstractly describe the problem you are addressing and how you are addressing it. What is the problem? Why is it important? What is your basic approach (few words are enough here)? A short discussion of how it fits into related work in the area is also desirable.*
 
A challenge for any classification is to be able to separate signal from noise. A canonical example is speech recognition in a noisy environment, or *speech enhancement*. Without such separation, signal can be lost or undetected. Here we address a related problem: detecting the presence or absence of a particular musical instrument in ensemble context.

There is considerable literature on the topic of *source separation* in music, particularly with respect to distinguishing *lead* and *accompaniment.* This is well-known as an interesting and difficult problem. Here we have (what we hope is) a somewhat more modest goal -- to classify samples with regard to whether or not a particular instrument is playing among others. We will not try to distinguish lead from accompaniment. 
 
### Problem Definition and Algorithm

*Precisely define the problem you are addressing (i.e. formally specify the inputs and outputs). Elaborate on why this is an interesting and important problem.*

The inputs to this problem will be features extracted from short samples (i.e., 5 - 15 seconds) of music performance by a small ensemble. The feature set may include (but is not necessarily limited to) zero crossing rate, spectral centroid, mel frequency cepstral coefficients (MFCC), spectral rolloff, and spectral bandwidth. Samples will be labeled indicating the presence or absence of the target instrument.

The output to this problem will be a binary classification of samples --- is the instrument present or is it not?
 
Many related problems, especially with regard to speech, can assume a stationary background of noise. Not so with music. "[M]usical sources are characterized by a very rich, non-stationary spectro-temporal structure." [Rafii 2]

Another related problem is identification of bird or amphibian songs in the wild [Acevedo 1] or fish in the ocean [Sattar 1]. In both cases, sounds produced by certain animals are identified from a noisy background. These problems do not have the additional challenge of non-stationary noise as with music. Still, their approaches may be instructive.

### Dataset

*Answer all the questions related to your dataset, such as: Describe your dataset? Is your dataset available? Is it labeled, or do you need an effort to label it (how? when? how many samples do you need to label?)? Do you need special hardware to process your data? Etc …*

Our dataset will consist of several thousand samples -- the precise number is yet to be determined. Audio tracks can be collected from sources like freemusicarchive.org, the MUSDB18 corpus, and personal collections. Samples will be extracted from these tracks (we already have working code to extract snips from FLAC audio tracks and optionally mix them down from stereo to mono).

Multiple, non-overlapping samples can be extracted from the same track.

At present, apart from the small MUSDB18 corpus we are unaware of any publicly-available source for the labeled data that we'd need, so we will have to label most samples ourselves. However, since samples will be short, and labeling will be a simple binary value, we do not believe that this will be too onerous.

It is unclear to us at this point what will constitute a suitable set of samples. If we focus on a single instrument (as currently planned) should we only consider samples where the instrument is playing in a prominent role, *e.g.*, playing the melody or "soloing" and exclude samples where the instrument may be part of the accompaniment? Or should we attempt what is likely a more difficult problem: detecting whether or not that instrument is present in a sample regardless of its role. In this latter case we might work with a broader class of input samples.

Another possible subset of data to use for training is monophonic instrument samples, e.g. samples of solo tenor sax. Though the goal of the project is to use polyphonic test data (be able to identify a target instrument soloing over accompanying instrumentation), research has shown that training on monophonic data is beneficial for polyphonic identification [Brown 1, Bhalke 1, Toghiani-Rizi 1].

We will experiment with both single-channel and multi-channel approaches. We expect that the  single-channel approach will simplify models, perhaps at the cost of some accuracy, while in general the multi-channel approach will perform better (due to having an additional dimension).

Based on our current state of investigations into fair use, we believe the sampling that we propose (taking short samples and then extracting features from these samples) falls under "fair use" under US Copyright law. This is not a legal opinion but reflects our best understanding of the matter. This has been corroborated by Prof. Paul P. Philbin, Director of Access, Technology and Media Services, at the UVM David W. Howe Memorial Library. 

If we determine that using samples from private collections might infringe on copyright then we will stick with sources like MUSDB18 and freemusicarchive.org, but this will certainly limit the number of samples we'd have at our disposal.

At present, we do not anticipate needing any special hardware to process data. We have tested feature extraction (spectral centroid, MFCC, spectrogram, etc.) from suitably sized samples and find that this runs in a reasonable amount of time. Once feature extraction is perfomed, features can be stored to a modestly-sized Python pickle.

### Related Work

*Answer the following questions for each piece of related work that addresses the same or a similar problem. What is their problem and method? How is your problem approach could be different or better?*

Much of the literature we've identified so far focuses on source separation. This gives useful insight into our problem as well. It may make sense to attempt to separate lead from accompaniment before attempting to identify the instrument perfoming the lead. Even a partial separation may be useful in this regard. A great many techniques have been put to use including non-negative matrix factorization (NMF) [Vembu 1], robust principal component analysis (RPCA)[Jeong 1], and REpeating Pattern Extraction Technique (REPET) [Rafii 2], among others. It is unclear if any of these methods might be feasible for this project.

Alternatively, it may be the case that if we choose a single instrument that we might be able to produce a partial separation or enhanced signal through preprocessing, *e.g.* attempting to filter out frequencies not typically produced by the target instrument. This might involve high-pass or band-pass filtering or other techniques. 

We do not expect to use music structure analysis, factorization of known melody or similar techniques, or anything that requires prior knowledge of the music (*e.g.*, musical score) or performer.

Regarding machine learning techniques and algorithms, it appears from our preliminary review of the literature that deep learning may produce the best result for this type of problem. However, since we will begin learning other classifiers first we will likely experiment with them and perhaps change to a deep learning model later. This is to be determined in the future.
  
A number of methods have proven successful with audio recognition in other domains (*e.g.*, vocal music, speech, fish sounds) such as robust principal component analysis (RPCA), multiresolution acoustic features (MRAF), a variation of Mel Frequency Cepstral Coefficients (MFCC) with a Wiener filter, and Spectral Subband Centroid (SSC).[Jeong 1, Priyanka 1, 12, Chauhan 1, Deng 1] Based on our current understanding of these methods, it seems that we may wish to experiment with RPCA, but its unlikely that MRAF will be suitable for our application. This is due to the fact that unlike individual words in human speech, musical instruments can have a wide range of durations from milliseconds to seconds, and in the case of bowing, pumping, and circular breathing, tens of seconds! For this same reason, however, the modified MFCC-Wiener method could provide a solution due to its nature of being more effective in the frequency domain vs. the time domain. [Chauhan 1] SSC was found to be successful for audio fingerprinting when used with harmonic enhancement, and though harmonic enhancement likely won't be useful to us because it searches for general predominant pitches, SSC could produce good audio feature extraction results. [Deng 1]

### Bibliography

*Be sure to include a standard, well-formatted, comprehensive bibliography with citations from the text referring to previously published papers in the scientific literature that you utilized or are related to your work. Include references to the public code that you might use for your project.*

##### Tools

[McFee 1] McFee, Brian, et al. "librosa: A python package for music and audio analysis." https://zenodo.org/record/3955228#.X1tpc2gYBDU (also https://github.com/librosa/librosa)

[McFee 2] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

[Giannak 1] Giannakopoulos, Theodoros. "PyAudioAnalysis: A Python library for audio feature extraction, classification, segmentation and applications." https://github.com/tyiannak/pyAudioAnalysis (we have not used this so far but may do so in the future)

##### Topical

[Alias 1] Alías, Francesc, et al. A Review of Physical and Perceptual Feature Extraction Techniques for Speech, Music and Environmental Sounds

[Acevedo 1] Acevedo, Miguel A., et al. "Automated classification of bird and amphibian calls using machine learning: A comparison of methods" Ecological Informatics 4, 2009.

* [Bhalke 1] SPECTROGRAM BASED MUSICAL

[Brown 1] Brown, Judith. Computer identification of musical instruments using pattern recognition with cepstral coefficients as features. J. Acoust. Soc. Am. 105, 1999.

[Chauhan 1] Chauhan, Paresh M.; Desai, Nikita P., "Mel Frequency Cepstral Coefficients (MFCC) Based Speaker Identification in Noisy Environment Using Wiener Filter", 2014 International Conference on Green Computing Communication and Electrical Engineering (ICGCCEE), 2014.

[Chesmore 1] Chesmore, E.D.; Ohya, E. "Automated identification of field-recorded songs of four British grasshoppers using bioacoustic signal recognition", Bulletin of Entomological Research 94, 2004.

[Crammer 1] Crammer, Koby; Singer, Yoram. "On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines", Journal of Machine Learning Research 2, 2001

[Deng 1] Deng, Jijun; Wan, Wanggen; Yu, XiaoQing; Pan, Xueqian; Yang, Wei. "Audio Fingerprinting Based On Harmonic Enhancement And Spectral Subband Centroid", IET International Communication Conference on Wireless Mobile and Computing (CCWMC 2011), 2011.

[Eggink 1] Eggink, Jana; Brown, Guy J. "Instrument recognition in accompanied sonatas and concertos", 2004 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2004.

[Eronen 1] Eronen, A.; Klapuri, A., "Musical instrument recognition using cepstral coefficients and temporal features", 2000 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2000.

[Eronen 2] Eronen, A. "Comparison of features for musical instrument recognition", Proceedings of the 2001 IEEE Workshop on the Applications of Signal Processing to Audio and Acoustics, 2001.

[Essid 1] Essid, S.; et al., "Instrument recognition in polyphonic music", Proceedings. (ICASSP '05). IEEE International Conference on Acoustics, Speech, and Signal Processing, 2005.

[Halperin 1] Halperin, Tavi; Ephrat, Ariel; Yedid Hoshen. "Neural Separation of Observed and Unobserved Distributions". Facebook Research, 2018. https://research.fb.com/publications/neural-separation-of-observed-and-unobserved-distributions/

[Jeong 1] Jeong, I.-Y.; Lee, K. "Vocal separation using extended robust principal component analysis with Schatten P/Lp-norm and scale compression" International Workshop on Machine Learning for Signal Processing, Reims, France, Nov. 2014.

[Li 1]: Li, Peter; et al. Automatic Instrument Recognition in Polyphonic Music Using Convolutional Neural Networks (https://arxiv.org/abs/1511.05520)

[**Martins 1] Martins, L.; Burred, J.; Tzanetakis, G. Polyphonic Instrument Recognition Using Spectral Clustering, ISMIR, 2007.

[Priyanka 1] Priyanka, M. Anbu Swarna; et al. "Multiresolution Feature Extraction (MRFE) based speech recognition system", 2013 International Conference on Recent Trends in Information Technology (ICRTIT), July 2013.

[Rafii 1] Rafii, Zafar; Liutkus, Antoine; Fabian-Robert Stöter; Mimilakis, Stylianos Ioannis; Bittner, Rachel. "MUSDB18 - a corpus for music separation". 2017. https://doi.org/10.5281/zenodo.1117372

[Rafii 2] Rafii, Zafar; Liutkus, Antoine; Fabian-Robert Stöter, Stylianos Ioannis Mimilakis; Derry FitzGerald; Bryan Pardo. "An Overview of Lead and Accompaniment Separation in Music". IEEE/ACM Transactions on Audio, Speech and Language Processing. 2018. https://doi.org/10.1109/TASLP.2018.2825440 (also: https://arxiv.org/pdf/1804.08300.pdf)

[Sattar 1] Sattar, Farook; Cullis-Suzuki, Sarika; Jin, Feng. "Acoustic analysis of big ocean data to monitor fish sounds" Ecological Informatics, 34. 2016.

* [Toghiani-Rizi 1] Musical INtrusment Recog

[Vembu 1] Vembu, S.; Baumann, S. "Separation of vocals from polyphonic audio recordings," 6th International Conference on Music Information Retrieval, London, UK, Sep 2005.

##### See also

* https://freemusicarchive.org

* https://sigsep.github.io/datasets/musdb.html

* Python software for convex optimization (used in RPCA) https://cvxopt.org/

* Zafar, Rafii: https://scholar.google.com/citations?user=8wbS2EsAAAAJ&hl=en