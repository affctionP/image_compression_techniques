
from PIL import Image
import numpy as np
path = input('Enter image path:')

image = np.asarray(Image.open(path))

pixels = []

for row in image:

    for ch in row:

        for pix in ch:

            pixels.append(pix)

class Node:

        def __init__(self, prob, symbol, left=None, right=None):
            # probability of symbol
            self.prob = prob

            # symbol 
            self.symbol = symbol
  
            # left node
            self.left = left

            # right node
            self.right = right

            # tree direction (0/1)
            self.code = ''

""" A function to print the codes of symbols by traveling Huffman Tree"""

codes = dict()

def Calculate_Codes(node, val=''):
        # huffman code for current node
        newVal = val + str(node.code)

        if(node.left):
           Calculate_Codes(node.left, newVal)
        if(node.right):
           Calculate_Codes(node.right, newVal)

        if(not node.left and not node.right):
           codes[node.symbol] = newVal
         
        return codes        

""" A  function to get the probabilities of symbols in given data"""

def Calculate_Probability(data):
        symbols = dict()
        for element in data:
            if symbols.get(element) == None:
               symbols[element] = 1
            else: 
               symbols[element] += 1     
        return symbols

""" A function to obtain the encoded output"""

def Output_Encoded(data, coding):
        encoding_output = []
        for c in data:
        #  print(coding[c], end = '')
            encoding_output.append(coding[c])
        
        string = ''.join([str(item) for item in encoding_output])    
        return string
        
""" A function to calculate the space difference between compressed and non compressed data"""    

def Total_Gain(data, coding):
         before_compression = len(data) * 8 # total bit space to stor              the data before compression
         after_compression = 0
         symbols = coding.keys()
         for symbol in symbols:
            count = data.count(symbol)
         after_compression += count * len(coding[symbol]) #calculate how  many bit is required for that symbol in total
         print("Space usage before compression (in bits):", before_compression)    
         print("Space usage after compression (in bits):",    after_compression)           

def Huffman_Encoding(data):
        symbol_with_probs = Calculate_Probability(data)
        symbols = symbol_with_probs.keys()
        probabilities = symbol_with_probs.values()
        print("symbols: ", symbols)
        print("probabilities: ", probabilities)
    
        nodes = []
    
        # converting symbols and probabilities into huffman tree nodes
        for symbol in symbols:
            nodes.append(Node(symbol_with_probs.get(symbol), symbol))
    
        while len(nodes) > 1:
              # sort all the nodes in ascending order based on their probability
              nodes = sorted(nodes, key=lambda x: x.prob)
              # for node in nodes:  
        #     print(node.symbol, node.prob)
    
        # pick 2 smallest nodes
              right = nodes[0]
              left = nodes[1]
    
              left.code = 0
              right.code = 1
    
        # combine the 2 smallest nodes to create new node
              newNode = Node(left.prob+right.prob, left.symbol+right.symbol, left, right)
    
              nodes.remove(left)
              nodes.remove(right)
              nodes.append(newNode)
            
        huffman_encoding = Calculate_Codes(nodes[0])
        print("symbols with codes", huffman_encoding)
        Total_Gain(data, huffman_encoding)
        encoded_output = Output_Encoded(data,huffman_encoding)
        return encoded_output, nodes[0] 



encoding, tree = Huffman_Encoding(pixels)
file = open('compressedati.txt','w')
file.write(encoding)
file.close()