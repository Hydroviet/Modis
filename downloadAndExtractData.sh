#!/bin/bash

filename=$1 # Example: MOD13Q1
start=$2
end=$3

if [[ ! -d $filename ]]; then
    mkdir $filename
fi
cd $filename
 
{
    read
    while IFS=, read -r id site_id product latitude longitude email start_date end_date kmAboveBelow kmLeftRight order_uid
    do 
        if (( $id < $start )); then
            continue
        fi

        if [[ ! -d /media/lamductan/Transcend/MODIS/MOD13Q1/$id ]]; then
            if [[ ! -d ../$id ]]; then
                if [[ ! -f $id.tar.gz ]]; then
                    wget $order_uid/tif/GTiff.tar.gz -O $id.tar.gz
                fi
            fi
            ./../extract.sh $id
            rm -rf $id.tar.gz
            #mv $id /media/lamductan/Transcend/MODIS/MOD13Q1
        fi

        if (( $id == $end )); then
            break
        fi 
    done
} < ../$filename.csv

cd ..
