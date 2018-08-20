#! /bin/bash
dataFolder=$1
start=$2
end=$3

cd $dataFolder

for (( i = $start; i <= $end; i++ )); do
    cd $i
    for (( year = 2000; year <= 2018; year++ )); do
        if [[ ! -d $year ]]; then
            mkdir $year
        fi
    done

    for folder in *; do
        year=$(echo $folder | cut -b 1-4)
        if [ "$folder" == "$year" ]; then
            continue
        fi
        mv $folder $year     
    done
    cd ..
done

cd ..
