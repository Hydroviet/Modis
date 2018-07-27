#!/bin/bash

filename=$1 # Example: ../MOD13Q1

{
    read
    while IFS=, read -r id site_id product latitude longitude email start_date end_date kmAboveBelow kmLeftRight order_uid
    do 
        if [[ ! -d $id ]]; then
            if [[ ! -f $id.tar.gz ]]; then
                wget $order_uid/tif/GTiff.tar.gz -O $id.tar.gz
            fi
            ./extract.sh $id
            rm -rf $id.tar.gz
        fi
    done
} < $filename