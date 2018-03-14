#!/usr/bin/python3
import sys
import argparse
from utils.downloader import get_file


parser = argparse.ArgumentParser(description='Generate stuff.')
parser.add_argument('command')
parser.add_argument('-d','--data')
parser.add_argument('-m','--model')

def _main(args):
	if 'train' == args.command:
		if args.data is None:
			parser.error('training data required')
		if args.model is None:
			parser.error('model required')
		print('asdasd')
	return 0

sys.exit(_main(parser.parse_args()))