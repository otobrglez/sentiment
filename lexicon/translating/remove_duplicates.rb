
if ARGV.size != 0
  puts <<-END
Usage:
  tr-positive-tmp.txt, tr-negative-tmp.txt and fin-duplicates.txt need to exist
  in CWD. This program takes the files and removes clusters of words that which
  contain words that appear in both lists.
  END
  exit
end

$dup_words = []
File.open("fin-duplicates.txt", "r") do|fin|
  fin.each_line do |word|
    $dup_words << word.chomp!
  end
end


def clean_duplicates(infile, outfile)
  File.open(infile, "r") do|fin|
    File.open(outfile, "w") do|fout|
      clusters = fin.each_line.map {|cluster| cluster.split("|").map(&:chomp)}.select {|c| c[0] != ""}
      unique_clusters = clusters.select {|cluster| cluster.all? {|word| not $dup_words.include? word }}
      unique_clusters.flatten.each {|word| fout.puts word}
    end
  end
end

clean_duplicates("tr-positive-tmp.txt", "tr-positive-words.txt")
clean_duplicates("tr-negative-tmp.txt", "tr-negative-words.txt")
