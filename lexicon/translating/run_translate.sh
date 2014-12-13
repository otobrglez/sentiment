#!/usr/bin/env bash

# run with ./run_translate.sh [slownet] [positive-words] [negative-words]
# eg ./run_translate.sh slownet-2014-07-08.xml positive-words.txt negative-words.txt

# remove comments and redundadnd whitespace from opinion lexicon
(grep -iv '^;' positive-words.txt | grep -iv '^$') > positive.txt
(grep -iv '^;' negative-words.txt | grep -iv '^$') > negative.txt 
# translate lexicon with translate.rb
ruby translate.rb slownet-2014-07-08.xml positive.txt negative.txt

# convert to a proper list of words
(cat tr-positive-tmp.txt| tr ' ' '\n' | grep -v '^$' | sort | uniq) > tr-positive-words.txt
(cat tr-negative-tmp.txt | tr ' ' '\n' | grep -v '^$' | sort | uniq) > tr-negative-words.txt

# remove tmp files
# rm positive.txt negative.txt tr-positive-tmp.txt tr-negative-tmp.txt
