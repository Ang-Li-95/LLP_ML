import pandas as pd
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputDir',dest='inputDir',default="")
parser.add_argument('--input',dest='input',default="")
args = parser.parse_args()

if args.inputDir[-1] != '/':
  args.inputDir += '/'

data = pd.read_csv(args.inputDir + args.input)
data[:len(data)//2].to_csv(args.inputDir + "test" + args.input)

data[len(data)//2:].to_csv(args.inputDir + "train" + args.input)
