#!/usr/bin/env bash

# run with ./run_translate.sh [slownet] [positive-words] [negative-words]
# eg ./run_translate.sh slownet-2014-07-08.xml positive-words.txt negative-words.txt

# remove comments and redundadnd whitespace from opinion lexicon
(grep -ie '^[a-zA-Z0-9]' positive-words.txt) > positive.txt
(grep -ie '^[a-zA-Z0-9]' negative-words.txt) > negative.txt 
# translate lexicon with translate.rb
ruby translate.rb slownet-2014-07-08.xml positive.txt tr-positive-tmp.txt
ruby translate.rb slownet-2014-07-08.xml negative.txt tr-negative-tmp.txt

# convert to a proper list of words
(cat tr-positive-tmp.txt | tr '|' '\n' | grep -v '^$' | sort | uniq) > tr-positive-words.txt
(cat tr-negative-tmp.txt | tr '|' '\n' | grep -v '^$' | sort | uniq) > tr-negative-words.txt

# move words that appear in both lists into separate file
sort tr-positive-words.txt tr-negative-words.txt | uniq -d > fin-duplicates.txt
comm -23 tr-positive-words.txt fin-duplicates.txt > fin-positive-words.txt
comm -23 tr-negative-words.txt fin-duplicates.txt > fin-negative-words.txt

# remove tmp files
# rm positive.txt negative.txt tr-positive-tmp.txt tr-negative-tmp.txt tr-positive-words.txt tr-negative-words.txt
