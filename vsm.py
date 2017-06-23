import numpy as np
import functools
import xml.etree.ElementTree as ET
import sys

def isChinese(c):
	if c >= '\u4e00' and c <= '\u9fff': return True
	else: return False

def mycmp(a, b):
	return points[b] - points[a]

def bigram(q):
	ret = []
	for i in range(len(q)-1):
		ret.append(q[i]+q[i+1])
	return ret

#Parse arguments
for i in range(1, len(sys.argv)):
	if   sys.argv[i] == '-i':
		query_file = sys.argv[i+1]
	elif sys.argv[i] == '-o':
		ranked_list = sys.argv[i+1]
	elif sys.argv[i] == '-m':
		model_dir = sys.argv[i+1]
	elif sys.argv[i] == '-d':
		NTCIR_dir = sys.argv[i+1]

vocabulary = {}
file_list = {}
inverted = {}
sw = '的是一在有個我不了這他也就人都說而們你要之會對及和與以很種中'
stop_words = set( c for c in sw )

with open(model_dir+'/vocab.all', 'r', encoding='utf8') as f:
	next(f)
	cnt = 1
	for row in f:
		vocabulary[cnt] = row[:-1]
		cnt += 1

with open(model_dir+'/file-list', 'r') as f:
	cnt = 0
	for row in f:
		file_list[cnt] = row[:-1]
		cnt += 1

file_num = 46972

file_len = np.zeros(46972)
idf = np.zeros(957436)
w2n = {}
idx = 0

with open(model_dir+'/inverted-file', 'r') as f:
	for row in f:
		a = row[:-1].split()
		if a[1] == '-1':
			for i in range( int(a[2]) ):
				f.readline()
			continue
		else:
			if vocabulary[ int(a[0]) ] in stop_words or vocabulary[ int(a[1]) ] in stop_words or not isChinese(vocabulary[ int(a[0]) ][0]) or not isChinese(vocabulary[ int(a[1]) ][0]) :
				for i in range( int(a[2]) ):
					next(f)
				continue
			word = vocabulary[int(a[0])]+vocabulary[int(a[1])]

		w2n[word] = idx

		idf[idx] = np.log( ( file_num - int(a[2] ) + 0.5) / (int(a[2]) + 0.5) )
		posting = []
		for i in range( int(a[2]) ):
			tmp = f.readline()[:-1]
			tmp = tmp.split()
			posting.append( tmp )
			file_len[ int(tmp[0]) ] += int(tmp[1])

		inverted[word] = posting
		idx += 1


avglen = file_len.mean()

file = open(ranked_list, 'w')
file.write('query_id,retrieved_docs\n')

tree = ET.ElementTree(file=query_file)
root = tree.getroot()

for T in range(len(root)):
	query = root[T][4].text[1:-2].split('、')
	query.append(root[T][1].text)
	Q = []
	for i in range( len(query) ):
		Q.extend( bigram(query[i]) )

	points = np.zeros( file_num )

	for w in Q:
		if w not in inverted: continue
		i = w2n[w]
		P = inverted[w]
		for p in P:
			doc_id = int(p[0])
			freq = int(p[1])
			tf = (freq*2.2 / ( freq + 1.2*(0.+1*file_len[doc_id]/avglen) ) )
			points[doc_id] += tf*idf[i]

	index = [ i for i in range(file_num) ]
	index = sorted(index, key=functools.cmp_to_key(mycmp))

	if '-r' in sys.argv:
		Q2 = []
		top = set( index[i] for i in range(10) )
		for w, P in inverted.items():
			for p in P:
				doc_id = int(p[0])
				if doc_id in top:
					Q2.append(w)

		for w in Q2:
			if w not in inverted: continue
			i = w2n[w]
			P = inverted[w]
			for p in P:
				doc_id = int(p[0])
				freq = int(p[1])
				tf = (freq*2.2 / ( freq + 1.2*(0.25+0.75*file_len[doc_id]/avglen) ) )
				points[doc_id] += tf*idf[i]*0.8/10

		index = [ i for i in range(file_num) ]
		index = sorted(index, key=functools.cmp_to_key(mycmp))

	file.write( root[T][0].text[-3:] )
	file.write(',')
	for i in range(100):
		file.write(file_list[index[i]][-15:].lower())
		file.write(' ')
	file.write('\n')
