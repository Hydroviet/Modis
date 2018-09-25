#!/bin/bash

for (( timeSteps = 11; timeSteps <= 11; timeSteps+=1 )); do
    for (( filters = 10; filters <= 10; filters+=10 )); do
        for (( kernelSize = 3; kernelSize <= 3; kernelSize+=2 )); do
            for (( nHiddenLayers = 3; nHiddenLayers <= 3; nHiddenLayers+=1 )); do
                python tuning.py -t $timeSteps -f $filters -k $kernelSize -n $nHiddenLayers -e 100
            done
        done
    done
done

#!/bin/bash

# for (( timeSteps = 8; timeSteps <= 8; timeSteps+=1 )); do
#     for (( filters = 30; filters <= 40; filters+=10 )); do
#         for (( kernelSize = 3; kernelSize <= 7; kernelSize+=2 )); do
#             for (( nHiddenLayers = 3; nHiddenLayers <= 4; nHiddenLayers+=1 )); do
#                 python tuning.py -t $timeSteps -f $filters -k $kernelSize -n $nHiddenLayers -e 100
#             done
#         done
#     done
# done