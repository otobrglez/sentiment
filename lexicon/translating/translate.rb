require 'nokogiri'

if ARGV.size != 3
  puts <<-END
Usage:
  #$0 [path to slownet] [path to positive words] [path to negative words]

  NOTE: files with positive and negative words must only contain one word per line
  END
  exit
end

@slownet = Nokogiri::XML(File.open(ARGV[0]))

def get_synonyms(eng)
  synonyms = []
  @slownet.xpath("//SYNONYM[@xml:lang='eng']/LITERAL[text()='#{eng}']/../../ID").each do|hit|
    hit.parent.xpath("SYNONYM[@xml:lang='slv']/LITERAL/text()").each {|synonym| synonyms << synonym }
  end
  synonyms.map(&:to_s).uniq.join(" ")
end

def construct_synonyms(infile, outfile)
  count = 0
  File.open(outfile, "w") do|fout|
    File.open(infile, "r") do|fin|
      fin.each_line do |line|
        fout.puts(get_synonyms line.chomp)
        count += 1
        puts "processed #{count} words..." if (count += 1) % 10 == 0
      end
    end
  end
end

puts "checking positive words..."
construct_synonyms(ARGV[1], "tr-positive-tmp.txt")
puts "checking negative words..."
construct_synonyms(ARGV[2], "tr-negative-tmp.txt")
