#!/bin/bash

outDir=$1
filename=$1.tar.gz

echo $outDir

if [[ ! -d $outDir ]]; then    
    # Uncompress .tar.gz file
    mkdir $outDir
    if [[ -d temp ]]; then
        rm -rf temp
    fi
    mkdir temp
    tar -xzvf $filename -C temp
    echo "Untar $filename successfully"

    # Move data to appropriate folder
    cd temp
    for file in *; do
      day=$(echo $file | cut -b 10-16)
      if [[ ! -d ../$outDir/$day ]]; then
        mkdir ../$outDir/$day
      fi
      mv $file ../$outDir/$day
    done
    cd ..
    rm -rf temp
    echo "Archive all bands successfully"
    echo ""
fi
