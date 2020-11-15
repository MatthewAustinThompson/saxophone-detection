# CS 254 Machine Learning
## Project Proposal
### Thompson, Gamble, Cafiero

### 1. Introduction
*Motivate and abstractly describe the problem you are addressing and how you are addressing it. What is the problem? Why is it important? What is your basic approach? A short discussion of how it fits into related work in the area is also desirable. Summarize the basic results and conclusions that you will achieve.*

A challenge for any classification is to be able to separate signal from noise. A canonical example is speech recognition in a noisy environment, or *speech enhancement*. Without such separation, signal can be lost or undetected. Here we address a related problem: detecting the presence or absence of a particular musical instrument in ensemble context.

There is considerable literature on the topic of *source separation* in music, particularly with respect to distinguishing *lead* and *accompaniment.* This is well-known as an interesting and difficult problem. Here we have (what we hope is) a somewhat more modest goal -- to classify samples with regard to whether or not a particular instrument is playing among others. We will not try specifically to distinguish lead from accompaniment, only presence or lack thereof.


### 2. Problem Definition and Algorithm

#### 2.1 Task Definition
*Precisely define the problem you are addressing (i.e. formally specify the inputs and outputs). Elaborate on why this is an interesting and important problem.*

The inputs to this problem will be features extracted from short samples (*i.e.*, 5 seconds) of music performance by a small ensemble. The feature set may include (but is not necessarily limited to) zero crossing rate, spectral centroid, mel frequency cepstral coefficients (MFCC), spectral rolloff, and spectral bandwidth. Samples will be labeled indicating the presence or absence of the target instrument.

The output to this problem will be a binary classification of samples --- is the instrument present or is it not?

Many related problems, especially with regard to speech, can assume a stationary background of noise. Not so with music. "[M]usical sources are characterized by a very rich, non-stationary spectro-temporal structure." [Rafii 2]

Another related problem is identification of bird or amphibian songs in the wild [Acevedo 1] or fish in the ocean [Sattar 1]. In both cases, sounds produced by certain animals are identified from a noisy background. These problems do not have the additional challenge of non-stationary noise as with music. Still, their approaches may be instructive.


#### 2.2 Dataset
*Answer all the questions related to your dataset, such as: Describe your dataset? is your dataset available? is it labeled, or do you need an effort to label it (how? when ? how many samples do you need to label?)? Do you need special hardware to process your data?*

Our dataset will consist of a few thousand samples --- the precise number is yet to be determined. Audio tracks have been be collected from personal collections. Samples have been be extracted from these tracks (we have working code to extract snips from FLAC audio tracks and optionally mix them down from stereo to mono).

Multiple, non-overlapping samples can be extracted from the same track.

At present, apart from the small MUSDB18 corpus we are unaware of any publicly-available source for the labeled data that we'd need, so we are labeling samples ourselves. However, since samples will be short, and labeling will be simple binary values, this is not too onerous.

It is unclear to us at this point what will constitute a suitable set of samples. If we focus on a single instrument (as currently planned) should we only consider samples where the instrument is playing in a prominent role, *e.g.*, playing the melody or "soloing" and exclude samples where the instrument may be part of the accompaniment? Or should we attempt what is likely a more difficult problem: detecting whether or not that instrument is present in a sample regardless of its role. In this latter case we might work with a broader class of input samples. At present our experiments have focused on the target instrument being in a prominent role.

Another possible subset of data to use for training is monophonic instrument samples, e.g. samples of solo tenor sax. Though the goal of the project is to use polyphonic test data (be able to identify a target instrument soloing over accompanying instrumentation), research has shown that training on monophonic data is beneficial for polyphonic identification [Brown 1, Bhalke 1, Toghiani-Rizi 1].

We may experiment with both single-channel and multi-channel approaches. We expect that the single-channel approach will simplify models, perhaps at the cost of some accuracy, while in general the multi-channel approach will perform better (due to having an additional dimension).

Based on our current state of investigations into fair use, we believe the sampling that we propose (taking short samples and then extracting features from these samples) falls under "fair use" under US Copyright law. This is not a legal opinion but reflects our best understanding of the matter. This has been corroborated by Prof. Paul P. Philbin, Director of Access, Technology and Media Services, at the UVM David W. Howe Memorial Library.

At present, we do not anticipate needing any special hardware to process data. We have tested feature extraction (spectral centroid, MFCC, spectrogram, spectral rolloff etc.) from suitably sized samples and find that this runs in a reasonable amount of time. Once feature extraction is performed, features can be stored to a modestly-sized Python pickle.

We have developed a pipeline for
* Randomly selecting audio files for sampling from a suitable pool
* Extracting samples (up to eight 5 second snips), mixing down to mono and saving as a WAV file
* Automatically generating template file(s) for labeling
* Extracting features from WAV audio samples and saving to pickles
* Extracting data from pickles and labels from a matching CSV file and converting all of these data to a Pandas dataframe


#### 2.3 Algorithm Definition
*Describe in reasonable detail the algorithm you are using to address this problem. Explain your baseline system and justify the choice of the baseline system? Do you have the plan to use other techniques or do you want to extend or enhance the current baseline system? Elaborate here and discuss different scenarios that you might approach. A psuedocode description of the algorithm you are using is frequently useful.*

We have tested our dataset on multiple classification models (logistic regression, random forest, and support vector machine). So far, we have achieved the best results with a support vector machine. For example, achieving a 0.82 recall for tenor saxophone, in the best case. One paper [Eronen 2] indicates "In instrument family classification, the recognition accuracy for the tenor saxophone was the worst [55%]" so we understand this instrument is a difficult problem.

We believe we can enhance the current support vector machine system with a refined feature set and more samples, but a neural network may end up being the most successful. We also plan to test ensemble methods.


### 3. Experimental Evaluation

#### 3.1 Methodology
*What are the criteria you are using to evaluate your method? What specific hypotheses do your experiment test? Describe the experimental methodology that you used, and why do you think it is the right way to measure the performance. What is the training/test data that was used, and why is it realistic or interesting? Exactly what performance data did you collect and how are you presenting and analyzing it? Comparisons to competing methods that address the same problem are useful.*

Our primary criterion for evaluating our method is recall. This is the most appropriate measure.

The idea will be to see if our model can distinguish between the presence and absence of the target instrument (tenor saxophone). Since our data set is skewed --- more samples without tenor saxophone than with the instrument --- accuracy is not a useful measure. Hence our focus on recall.

Our initial tests so far have been on a data set of 2,263 five-second samples (labeling is in progress but not complete). We exclude some of these samples due to a number of factors: false start in alternate take, music not playing, music fading out, poor sound quality, confusing sample (labeling), big band arrangement / sectional, cluttered chart, announcer speaking, etc. Our yield after filtering for training and testing tenor saxophone is currently 1,547 samples.

We produce multiple labels for each sample, but only use one label for testing and training. Labels are

* Soprano (or sopranino) saxophone
* Alto saxophone (or stritch)
* Tenor saxophone (or saxello/manzello)
* Alto or tenor saxophone (uncertain)
* Baritone saxophone
* Clarinet (or tarogato)
* Other reed (*e.g.*, oboe, bassoon, C melody)
* Trumpet or cornet
* Trombone (including valve trombone)
* Other brass instrument (*e.g.*, euphonium, flugelhorn, tuba)
* Extended technique
* Exclude

Samples may have multiple labels and we can filter accordingly when training and testing.

With current extraction process, each sample consists of 9,483 features.

| Feature class      | # of features |
|--------------------|--------------:|
| Zero crossings     |             1 |
| MFCC x 20          |         8,620 |
| Spectral centroids |           431 |
| Spectral rolloff   |           431 |
| All features       |         9,483 |

We are currently adding spectral bandwidth, spectral contrast, and spectral flatness.


#### 3.2 Results and Discussion
*Present the quantitative results of your baseline system experiments. Graphical data presentation such as graphs and histograms are frequently better than tables.*

So far our results have been moderately successful. Our best recall for tenor saxophone is 0.74 with the full data set. However, on a somewhat simpler problem, where we do not attempt to identify tenor saxophone when other brass or wind instruments are present our recall improves to 0.82. For example, these are cases where the accompaniment to the tenor might be piano, bass, and drums, but not trombone and trumpet. The negative case would be when accompaniment is playing, but not the tenor.

We have used naive search on hyperparameter C (testing an array of hand-picked values) and found that best performance was likely in a range between 0.1 and 3.0.

![](assets/svm_score_vs_c.png)

We used random search (uniform distribution in indicated range, 5-fold cross validation, 100 iterations) to narrow this down and found a best value for C of 1.716. Using this value of C, our recall on tenor saxophone improves to 0.82. This is the best we have achieved so far. We expect this value to change if we add more data (which we expect to).

We have experimented with an ensemble approach, building ensembles of 100 and 400 SVM classifiers, using bagging, and setting `max_features` to 2,000, 4,000, or 6,000 (recall we have 9,483 total features). The ensemble of 100 classifiers gave us a recall of 0.80 (not an improvement) and the ensemble of 400 classifiers gave us a recall of 0.80 (not an improvement). We will continue to experiment with this approach. We believe we need more data for this to be effective.

We have not tried excluding classes of features (e.g., not using spectral centroids, but using MFCCs). This is something else with which we might experiment.


##### Preliminary results
![](assets/prelim_results.png)


###### Other instruments

The similar result for alto saxophone is recall of 0.51.

We believe that alto and tenor saxophone have similar acoustic characteristics. In fact, in some registers, over a five-second sample, we encountered cases where it was difficult for us to distinguish while labeling (and one of us plays the saxophone!). By excluding other brass and reeds we avoid providing our algorithm with a significant source of confusion. Note in the diagram below the considerable overlap in ranges between alto, tenor, and baritone saxophones.

![](assets/sax_ranges.png)

In future tests we will try to combine the two and see if we can build a model to detect the presence of alto or tenor sax (either, doesn't matter). Preliminary tests in this regard have not yielded promising results. (We find this a bit puzzling.)

Other instruments tested (e.g., trumpet) have lower recall.

We have tried decision tree, random forest, and SVM. We have tested SVM with linear, polynomial (d=2), and gaussian kernel. Gaussian kernel (RBF) performs best. All attempts so far to manually adjust gamma have resulted in poorer performance rather than just accepting the SciKit default.

We are using StandardScaler to scale data before running our algorithm.

We have not yet attempted PCA.


#### 3.3 Discussion
*Is your hypothesis supported? What conclusions do the results support the strengths and weaknesses of your method compared to other methods? How can the results be explained in terms of the underlying properties of the algorithm and/or the data? What is your target/goals to achieve by the end of the semester?*

We believe that we can improve on this result with an SVM or ensemble of SVMs, but it may be that a neural network is a better approach.

We see continued marginal improvement with increasing size of our data set, so we will continue to label more samples as time permits.

Our goal for the semester would be to continue experimenting with different approaches to see if we can push recall for some target instrument over 0.90. We understand this is a difficult problem and would consider this quite an accomplishment.


### 4. Related Work
*Answer the following questions for each piece of related work that addresses the same or a similar problem. What are their problems and methods? How is your problem approach could be different or better?*

Much of the literature we've identified so far focuses on source separation. This gives useful insight into our problem as well. It may make sense to attempt to separate lead from accompaniment before attempting to identify the instrument performing the lead. Even a partial separation may be useful in this regard. A great many techniques have been put to use including non-negative matrix factorization (NMF) [Vembu 1], robust principal component analysis (RPCA)[Jeong 1], and REpeating Pattern Extraction Technique (REPET) [Rafii 2], among others. It is unclear if any of these methods might be feasible for this project.

Alternatively, it may be the case that if we choose a single instrument that we might be able to produce a partial separation or enhanced signal through preprocessing, *e.g.* attempting to filter out frequencies not typically produced by the target instrument. This might involve high-pass or band-pass filtering or other techniques.

We do not expect to use music structure analysis, factorization of known melody or similar techniques, or anything that requires prior knowledge of the music (*e.g.*, musical score) or performer.

Regarding machine learning techniques and algorithms, it appears from our preliminary review of the literature that deep learning may produce the best result for this type of problem. However, since we will begin learning other classifiers first we will likely experiment with them and perhaps change to a deep learning model later. This is to be determined in the future.

A number of methods have proven successful with audio recognition in other domains (*e.g.*, vocal music, speech, fish sounds) such as robust principal component analysis (RPCA), multiresolution acoustic features (MRAF), a variation of Mel Frequency Cepstral Coefficients (MFCC) with a Wiener filter, and Spectral Subband Centroid (SSC). [Jeong 1, Priyanka 1, 12, Chauhan 1, Deng 1] Based on our current understanding of these methods, it seems that we may wish to experiment with RPCA, but its unlikely that MRAF will be suitable for our application. This is due to the fact that unlike individual words in human speech, musical instruments can have a wide range of durations from milliseconds to seconds, and in the case of bowing, pumping, and circular breathing, tens of seconds! For this same reason, however, the modified MFCC-Wiener method could provide a solution due to its nature of being more effective in the frequency domain vs. the time domain. [Chauhan 1] SSC was found to be successful for audio fingerprinting when used with harmonic enhancement, and though harmonic enhancement likely won't be useful to us because it searches for general predominant pitches, SSC could produce reasonable audio feature extraction results. [Deng 1]


### 5. Next Steps
*Precisely describe your next steps toward the final project objectives and how you will approach them. Elaborate enough to recognize your next steps including methods, timeline, etc*
*Explicitly define the role of each student on the project, make sure that each student will develop or participate in developing a machine learning algorithm.*

Our next steps will be to improve on our current result. Specific actions will include:

* We will label more samples. This is an ongoing process and we expect to at least double our number of samples. We will continue to label more samples as time permits. We have more audio files than we can label in a semester so we will not run out.

* We will experiment with additional features: frame RMS, spectral bandwidth, etc. We may also experiment with using more bands for finer-grained MFCCs (*e.g.*, 30 bands instead of the default 20, *etc.*) Some of this work is already underway.

* We will experiment with polynomial features.

* We will experiment with PCA. The current sub-features (e.g. MFCC, spectral rolloff) are large vectors up to size 1,292. Once each of these sub-feature vectors is flattened into a single feature vector for a datum, there is a tremendous number of features, nearly 30,000 per datum.

    * PCA will be used in conjunction with polynomial features to avoid having hundreds of thousands of features following poly expansion.

    * PCA will be used to try to plot data and see if any interesting clustering can be determined.

* We may experiment with multi-channel (stereo) data. This may take a few weeks and we will need to modify our data collection pipeline and reserve space for stereo samples.

* We will perform systematic experimentation on model hyperparameters (using grid and random search). This will include C and gamma, and perhaps other hyperparameters.

* We may build an ensemble classifier consisting of a number of SVMs.

* We may experiment with the following features:
    * Line spectral frequencies
    * Stereo panning spectrum feature
    * Spectral flux
    * Harmonicity
    * Spectral crest factor
    * Complex cepstrum
    * Linear prediction cepstrum coefficients
    * Modulation spectrogram
    * Wavelet-based direct approach (noise reduction)
    * Hurst parameter features (noise reduction)
    
* Sometimes, feeding more and more features into a ML algorithm will not yield better results. Feature selection is extremely important.
    * Experiment with SVM ensembles using different features and compare results resulting from using different feature sets.
    * Further research features being used by researchers in the field for similar tasks
<!-- bg - feature selection exploration - try training with different combos of features
cc to bg: we're doing this already with ensembles of SVMs see: 5s_tenor_svm.ipynb -->


### Student contributions

* Each student will contribute to the machine learning algorithm(s) used.
* Each student will label samples.
* Each student will contribute to the literature review.
* Each student will contribute to preparing the report.
* The pipeline and infrastructure for extracting samples and features was developed by Clayton, and expanded by Matthew.


### Bibliography

*Be sure to include a standard, well-formatted, comprehensive bibliography with citations from the text referring to previously published papers in the scientific literature that you utilized or are related to your work. Include references to the public code that you might use for your project.*

[Alias 1] Alías, Francesc, et al. A Review of Physical and Perceptual Feature Extraction Techniques for Speech, Music and Environmental Sounds. Applied Sciences, May 2016.

[Acevedo 1] Acevedo, Miguel A., et al. "Automated classification of bird and amphibian calls using machine learning: A comparison of methods" Ecological Informatics 4, 2009.

[Bhalke 1] D. G. Bhalke, C. B. R. Rao, D. S. Bormane, and M. Vibhute, “Spectrogram Based Musical Instrument Identification Using Hidden Markov Model (hmm) for Monophonic and Polyphonic Music Signals,” Acta Technica Napocensis: Electronica - Telecomunicatii; Cluj-Napoca, vol. 52, no. 2, pp. 1–9, 2011.

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

[Rida 1] Rida, Imad. Feature Extraction for Temporal Signal Recognition: An Overview. Preprint 2018, arXiv:1812.01780

[Sattar 1] Sattar, Farook; Cullis-Suzuki, Sarika; Jin, Feng. "Acoustic analysis of big ocean data to monitor fish sounds" Ecological Informatics, 34. 2016.

[Toghiani-Rizi 1] B. Toghiani-Rizi and M. Windmark, “Musical Instrument Recognition Using Their Distinctive Characteristics in Artificial Neural Networks,” arXiv:1705.04971 [cs, stat], May 2017, Accessed: Oct. 14, 2020. [Online]. Available: http://arxiv.org/abs/1705.04971.

[Vembu 1] Vembu, S.; Baumann, S. "Separation of vocals from polyphonic audio recordings," 6th International Conference on Music Information Retrieval, London, UK, Sep 2005.


##### Tools

[McFee 1] McFee, Brian, et al. "librosa: A python package for music and audio analysis." https://zenodo.org/record/3955228#.X1tpc2gYBDU (also https://github.com/librosa/librosa)

[McFee 2] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

[Giannak 1] Giannakopoulos, Theodoros. "PyAudioAnalysis: A Python library for audio feature extraction, classification, segmentation and applications." https://github.com/tyiannak/pyAudioAnalysis (we have not used this so far but may do so in the future)


##### Topical

[Acevedo 1] Acevedo, Miguel A., et al. "Automated classification of bird and amphibian calls using machine learning: A comparison of methods" Ecological Informatics 4, 2009.

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

[Vembu 1] Vembu, S.; Baumann, S. "Separation of vocals from polyphonic audio recordings," 6th International Conference on Music Information Retrieval, London, UK, Sep 2005.


##### See also

* Python software for convex optimization (used in RPCA) https://cvxopt.org/

* Zafar, Rafii: https://scholar.google.com/citations?user=8wbS2EsAAAAJ&hl=en
