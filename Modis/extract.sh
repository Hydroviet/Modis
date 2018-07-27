#!/bin/bash

outDir=$1
filename=$1.tar.gz

if [[ ! -d $outDir ]]; then
    # Make output directory for data
    mkdir $outDir
    cd $outDir
    for (( i = 2000; i < 2019; i++ )); do
        mkdir $i
    done
    cd ..
    
    # Uncompress .tar.gz file
    if [[ -d temp ]]; then
        rm -rf temp
    fi
    mkdir temp
    tar -xzvf $filename -C temp
    echo "Untar $filename successfully"

    # Move data to appropriate folder
    cd temp
    for nirFile in *NIR*; do
        year=$(echo $nirFile | cut -b 10-13)
        mv $nirFile ../$outDir/$year
    done
    for redFile in *red*; do
        year=$(echo $redFile | cut -b 10-13)
        mv $redFile ../$outDir/$year
    done
    cd ..
    rm -rf temp
    echo "Archive NIR and red band successfully"
    echo ""
fi
