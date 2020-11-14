# Downsample to 44100
# C. Cafiero

for f in *.flac; do

  RATE="$(soxi -r $f)"

  if [ $RATE -gt 44100 ]
  then
    ffmpeg -i "$f" -sample_fmt s16 -ar 44100 "${f%}_ds.flac";
  fi

done
