#!/bin/bash
# Basic while loop
counter=0
while [ $counter -lt 32 ]
do
    rsvg-convert -f pdf -o wl$counter.pdf wl$counter.svg
    echo $counter
    ((counter++))
done
echo All done
