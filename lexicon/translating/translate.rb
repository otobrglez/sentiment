require 'nokogiri'

if ARGV.size != 3
  puts <<-END
Usage:
  #$0 [path to slownet] [path to list of words] [out file]

  NOTE: files with words must only contain one word per line
  END
  exit
end

@slownet = Nokogiri::XML(File.open(ARGV[0]))

def get_synonyms(eng)
  synonyms = []
  @slownet.xpath("//SYNONYM[@xml:lang='eng']/LITERAL[text()='#{eng}']/../../ID").each do|hit|
    hit.parent.xpath("SYNONYM[@xml:lang='slv']/LITERAL/text()").each {|synonym| synonyms << synonym }
  end
  synonyms.map(&:to_s).uniq.join("|")
end

def construct_synonyms_file(infile, outfile)
  count = 0
  File.open(outfile, "w") do|fout|
    File.open(infile, "r") do|fin|
      fin.each_line do |word|
        fout.puts(get_synonyms word.chomp)
        puts "processed #{count} words..." if (count += 1) % 10 == 0
      end
    end
  end
end

puts "checking #{ARGV[1]}..."
construct_synonyms_file(ARGV[1], ARGV[2])
