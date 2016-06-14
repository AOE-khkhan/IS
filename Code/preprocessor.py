from optparse import OptionParser
import sys,re
def preProcess(line):
	line = line.lower()
        line = line.replace("."," ")
        line = line.replace(","," ")
        line=line.replace("\""," ")
        line=line.replace(" for "," ")
        line=line.replace(" a "," ")
        line=line.replace("  an "," ")
        line=line.replace(" the "," ")
        line=line.replace(" to "," ")
        line=line.replace(" his "," ")
        line=line.replace(" it "," ")
        line=line.replace(" and "," ")
        line=line.replace(" of "," ")
        line=line.replace(" but "," ")
        line=line.replace(" he "," ")
        line=line.replace(" him "," ")
        line=line.replace(" with "," ")
        line=line.replace(" has "," ")
        line=line.replace(" is "," ")
        line=line.replace(" was "," ")
        line=line.replace(" so "," ")
        return line
def extractor(fname):
	f = open(fname).readlines()
	out = open("intermediate.txt","w")
	for line in f:
		line=line.strip("\n")
		wl = line.split(",")
		textcomm =""
		for i in range(2,len(wl)):
			textcomm+=wl[i]
			textcomm+=" "
		#textcomm = preProcess(textcomm)
		out.write(textcomm+"\n")
	out.close()
def main():
	parser = OptionParser()
	parser.add_option("-f", dest="filename", help="corpus filename",default="../Data/IndiaBatting.txt")
	(options, args) = parser.parse_args()
	extractor(options.filename)
if __name__ == "__main__":
	main()
