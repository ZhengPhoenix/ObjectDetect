import re
import sys

REGEX = re.compile(r"\n(?P<name>[\s\S]+?.jpg)\D+?(?P<a>[0-9\.]+)\D+?(?P<b>[0-9\.]+)\D+?(?P<c>[0-9\.]+)\D+?(?P<d>[0-9\.]+)\D+?(?P<e>[0-9\.]+)\D+?(?P<f>[0-9\.]+)\D+?(?P<g>[0-9\.]+)\D+?(?P<h>[0-9\.]+)\D+?")

with open(sys.argv[1]) as f:
    data = f.read()
    
result = REGEX.findall('\n' + data)

with open(sys.argv[2],'w') as output_file:
    for i in result:
        output_file.write(' '.join(list(i))+'\n')
