# Instrument Detection

Matthew Thompson, Brandon Gamble, Clayton Cafiero

## Creating samples

We create samples by:

* Collecting a list of filenames from audio library,
* shuffling that list, and
* partitioning the list into batches.

See: `select_sample_files.py` and `sample_files_to_batches.py`

Then for each batch, we 
* sample target FLAC and MP3 audio files,
* mix them down from stereo to mono, and then 
* take 5s snippets from audio files (44.1k WAV).

See: `create_samples_5s.py`.

Some files were handled a little differently, e.g., the Sonny Rollins "soloscope" concert, and the University of Iowa alto sax samples. For these, see `create_samples_iowa.py` and `create_samples_soloscope.py`. Other files required downsampling; see: `downsample.sh`.

This is all hard-wired for paths on a team member's local network, *e.g.,* source audio files on NAS, and requires that the original source files are present in the appropriate locations. Samples are stored to `data/5s/wav_samples`.

* Select all files from my jazz/mainstream folder
    * ~3,125 entries after excluding Dizzy Gillespie's "Birks Works" (big band) and similar titles
* Shuffle file list
* Take files in batches of 100 until we have enough
    * Run `create_samples_5s.py`
    * This will generate up to eight samples per track
    * Non-conforming tracks are excluded (*e.g.*, non-conforming sample rate)
* 100 source files will yield about 500 - 700 samples
    * 600 samples is about 1.5 to 2.0 hours worth of labeling (it gets faster with a little practice)

Right now (12 November) we have 9,080 usable, labeled samples.

## Extracting features

Once samples are extracted to WAV files, we extract features.

* Extracting features
    - Zero crossings
    - Mel frequency cepstral coefficients
    - Log mel spectrogram
    - Spectral rolloff
    - Spectral centroid
    - Spectral flatness
    - Spectral bandwidth
    - Spectral contrast
* Saving features to pickle
    - All features except log mel spectrogram go in their own pickle; one pickle per audio sample, with `.dat` extension.
    - Log mel spectrogram data goes in its own pickle; one pickle per audio sample, with `.lms` extension.
* Plotting audio data (optional) 
    - Waveform
    - Mel frequency cepstral coefficients

## Workflow to get to labeled samples with extracted features

1. Run `select_sample_files.py` and direct output to file.
2. Manually split file into reasonable chunks and curate (remove stuff that's really far out or almost entirely extended technique --- for example, I know trombonists that couldn't tell you that Radu Malfatti plays a trombone).
3. Run `create_samples.py` using curated, chunked files as input. Do this as needed so we don't create more samples than we need.
4. Run `make_labeling_template.py` specifying the appropriate directory.
5. Put on the earbuds and slog through hours of labeling samples.
6. Once samples are labeled, move them to `data/labeled` and then run feature extraction with `feature_extract_r2.py`

#### Labels

Samples are labeled according to the following, where everything but sample_name and batch is numeric column taking values, 0 and 1 only:

* sample_name = filename of sample
* sop = soprano or sopranino sax (either, latter uncommon)
* alto = alto sax (or stritch, uncommon)
* tenr = tenor sax (or manzello, uncommon)
* tora = alto or tenor but not sure which
* bari = baritone or bass saxophone (latter uncommon)
* clrt = clarinet or bass clarinet (or tarogato, uncommon)
* othr = other reed instrument (oboe, bassoon, etc.)
* trmp = trumpet or cornet
* trmb = trombone
* otrb = other brass instrument (*e.g.*, flugelhorn, baritone, tuba, *etc.*)
* ext = instrument(s) is/are playing with "extended technique" (weird stuff)
* excl = exclude this sample from consideration
* batch = a label for the batch (*e.g.*, "1", "tbp03", "iowa_alto"; we label samples in batches)

From this we can "calculate" other fields, *e.g.*, if there is a sax (sop or alto or...), *etc.*

At this point you should have everything you need to run a MC algo. See `pickles_to_pandas.py` and `pickles_to_pandas_lms.py` for information on how to get all the pickled features into a nice Pandas dataframe.

## Hosting data on remote server

Data including WAV samples and extracted features are available at 172.105.27.120. Username: instrument. Contact a team member for password. We have found use of `rsync` most convenient, but `sftp` works too.

### Rsync

To push files to remote

     rsync -rptgovz ./data/5s/wav_sample/ instrument@172.105.27.120:~/data/5s/wav_sample/

To fetch files from remote

     rsync -rptgovz instrument@172.105.27.120:~/data/5s/wav_sample/ ./data/5s/wav_sample/     

If you're not sure what `rsync` is going to do use the `--dry-run` switch. This will show you what files will be transfered without actually transferring them.

## Jupyter Notebooks

[put information about notebooks here]

## Requirements

You must have a working binary of `ffmpeg`. See: https://ffmpeg.org. For OS/X this is `brew` installable with Homebrew. Other stuff is `pip` (or `conda`) installable. See `requirements.txt` in this repo.

## To clone this repo to your local machine

You must have a working installation of `git`. Then...

    git clone git@gitlab.uvm.edu:Clayton.Cafiero/instrumentdetection.git

To get the status of the repo

    git status
        
To stage your changes

    git add .
    
To commit your changes (save them to the local repo)

    git commit -m "Some message here describing your changes"
    
To push your changes to the remote host

    git push origin
    
To fetch current information about the state of the repository (including other work that's been committed)

    git fetch
    
To get the latest copy from the remote

    git pull
    
To create your own branch (a named working copy separate from the main branch)

    git checkout -b my_new_branch_name
    
We'll talk about `git merge` later.                            


## Some conversions and manipulations

### Converting aif files (Iowa samples) to wav (not nec. see below)

    for f in *.aif; do ffmpeg -i "$f" "${f%.aif}.wav"; done
    
### Padding Iowa samples to 5s

First, generate 1.05s of silence.

    sox -n -r 44100 -c 2 silence.wav trim 0.0 1.05

Then
    
    for f in *.aif; do sox silence.wav "$f" silence.wav "${f%.aif}.wav"; done
    
### Mixdown to mono

    for f in *.wav; do ffmpeg -i "$f" -ac 1 "${f%}.mono.wav"; done
    