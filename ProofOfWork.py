import hashlib
import time
import copy


class Block:
   def __init__(self, index, data, timestamp, previous_hash, nonce = 0):
      self.index = index
      self.data = data
      self.timestamp = timestamp
      self.previous_hash = previous_hash
      self.nonce = nonce
      
      
   def compute_hash(self):
      string_to_hash = str(self.index) + str(self.data) + str(self.timestamp) + str(self.previous_hash) + str(self.nonce)
      return hashlib.sha256(string_to_hash.encode()).hexdigest()


class Blockchain:
   difficulty = 5
   
   def __init__(self):
      self.unconfirmed_data = []
      self.chain = []
   
      
   def set_difficulty(self, diff):
      Blockchain.difficulty = int(diff)

      
   def add_genesis_block(self):
      genesis_block = Block(0, [], 0, "Chancellor on the brink...")
      genesis_block.hash = genesis_block.compute_hash()
      self.chain.append(genesis_block)
      #print(genesis_block.hash) #print statement
   
   
   @property
   def last_block(self):
      return self.chain[-1]


   def last_block_data(self):
      return str(self.chain[-1].data)


   def add_block(self, block, proof):
      previous_hash = self.last_block.hash
      #print(proof) #print statement
      
      if previous_hash != block.previous_hash:
         return False
         
      if not Blockchain.is_valid_proof(block, proof):
         return False
         
      block.hash = proof
      self.chain.append(block)
      return True
   
   
   @staticmethod
   def proof_of_work(block):
      block.nonce = 0
      #print(block.nonce)
      
      computed_hash = block.compute_hash()
      
      while not computed_hash.startswith('0' * Blockchain.difficulty):
         block.nonce += 1
         #print(block.nonce)
         computed_hash = block.compute_hash()
         
      return computed_hash
   
   
   def add_new_data(self, data):
      self.unconfirmed_data.append(data)   
   
   @classmethod 
   def is_valid_proof(cls, block, block_hash):
      return (block_hash.startswith('0' * Blockchain.difficulty) and block_hash == block.compute_hash())
   
   
   @classmethod   
   def check_chain_validity(cls, chain):
      result = True
      previous_hash = "Chancellor on the brink..."
      
      for block in chain:
         block_hash = block.hash
         #print("Validating: " + block.hash)
         #delattr(block, "hash")
         
         if (not cls.is_valid_proof(block, block_hash) or previous_hash != block.previous_hash) and previous_hash != "Chancellor on the brink...":
            result = False
            #print("Rejecting") #print statement
            break
            
         block.hash, previous_hash = block_hash, block_hash
      
      #print(result)
      
      return result
   
   
   def hiTestBye(self):
      return "Hello"

   
   def mine(self):
      if not self.unconfirmed_data:
         #print("False") #print statement
         return False
         
      last_block = self.last_block
      
      new_block = Block(index = last_block.index + 1, data = self.unconfirmed_data, timestamp = time.time(), previous_hash = last_block.hash)
      
      proof = self.proof_of_work(new_block)
      self.add_block(new_block, proof)
      
      self.unconfirmed_data = []
      
      return True
   
    
   def printBlockchain(self, name):
      
      chain_data = []
      
      for block in self.chain:
         chain_data.append(block.__dict__)
      
      #print(name + str(chain_data)) #print statement
   
   
   def consensus(self, new_blockchain):
   
      current_len = len(self.chain)
      length = len(new_blockchain.chain)
      
      #print(length) #print statement
      #print(current_len) #print statement
      
      if length > current_len and self.check_chain_validity(new_blockchain.chain):
         #print("Accept") #print statement
         return True
      
      #print("Reject") #print statement
      return False
   
   
   def clone(self, new_blockchain):
      self = copy.deepcopy(new_blockchain)
      return self
   
   
   def fakeBlock(self, fake_data):
      last_block = self.last_block
      self.unconfirmed_data = [];
      self.unconfirmed_data.append(fake_data)
      new_block = Block(index = last_block.index + 1, data = self.unconfirmed_data, timestamp = time.time(), previous_hash = last_block.hash)
      new_block_hash = new_block.compute_hash()
      new_block.hash = new_block_hash
      self.chain.append(new_block)
      self.unconfirmed_data = [];
      return self

#random real number between 100 and 1000 for measurement; noisy, row, and column are real
   

   def fixBlockchain(cls, chain):
      deleteBlock = False
      previous_hash = "Chancellor on the brink..."
      
      for block in chain:
         block_hash = block.hash
         
         if(not cls.is_valid_proof(block, block_hash) or previous_hash != block.previous_hash) and previous_hash != "Chancellor on the brink...":
            deleteBlock = True
         
         if deleteBlock:
            cls.chain.remove(block)

         block.hash, previous_hash = block_hash, block_hash
         
      return cls
      
      
   def saveOrphanedBlocks(self, new_blockchain, savedData):
      x = 0
      
      #print("Is it working?") #print statement
            
      while x < len(self.chain):
         #print(str(self.chain[x].data) + " vs " + str(new_blockchain.chain[x].data)) #print statement
         #print("length = " + str(len(self.chain[x].data))) #print statement
         finalData = str(self.chain[x].data)
     
         if finalData not in str(new_blockchain.chain[x].data):
            break
         x = x + 1
      
      finalX = x
      
      #print("x = " + str(finalX)) #print statement
      
      #if x < len(self.chain):
         #print("Chance Time!") #print statement
      
      while x < len(self.chain):
         y = finalX
         z = 0
         
         while(z < len(self.chain[x].data)):
            newTempData1 = str(self.chain[x].data[z]).replace("[", "")
            newTempData2 = newTempData1.replace("]", "")
            newFinalData = newTempData2.replace("'", "")
            
            while y < len(new_blockchain.chain):               
               if newFinalData in new_blockchain.chain[y].data:
                  #print(newFinalData) #print statement
                  #print("False Alarm!") #print statement
                  break
               y = y + 1
               
            if y >= len(new_blockchain.chain):
               
               #print(newFinalData) #print statement
               self.unconfirmed_data.append(newFinalData)
               savedData = savedData + 1
               #print("Orphaned Block Saved!") #print statement
         
            z = z + 1
         
         x = x + 1
         
         #print(self.unconfirmed_data) #print statement
   
      return savedData
      
   """
   def verify_and_add_block():
      block_data = request.get_json()
      block = Block(block_data["index"], block_data["transactions"], block_data["timestamp"], block_data["previous_hash"], block_data["nonce"])
      
      proof = block_data['hash']
      added = blockchain.add_block(block, proof) 
   """
   
   
   #if __name__ == '__main__':
      #mine(*sys.argv[1:])


"""
#Blockchains for Robots 1 and 2
blockchain1 = Blockchain()
blockchain2 = Blockchain()

blockchain1.add_genesis_block()
blockchain2.add_genesis_block()

#Robot 1 finds new data
blockchain1.add_new_data("1, 2, 3, 4")
blockchain1.mine()

#Robots 1 and 2 came into contact
if blockchain1.consensus(blockchain2):
   blockchain1 = blockchain1.clone(blockchain2)

if blockchain2.consensus(blockchain1):
   blockchain2 = blockchain2.clone(blockchain1)

#What is in Robot 1's blockchain?
chain_data = []
for block in blockchain1.chain:
   chain_data.append(block.__dict__)
print("Blockchain 1: " + str(chain_data))

#What is in Robot 2's blockchain?
chain_data = []
for block in blockchain2.chain:
   chain_data.append(block.__dict__)
print("Blockchain 2: " + str(chain_data))


#Robot 1 finds new data
blockchain2.add_new_data("11, 21, 31, 41")
blockchain2.mine()

#Robots 1 and 2 came into contact
if blockchain1.consensus(blockchain2):
   blockchain1 = blockchain1.clone(blockchain2)

if blockchain2.consensus(blockchain1):
   blockchain2 = blockchain2.clone(blockchain1)

#What is in Robot 1's blockchain?
chain_data = []
for block in blockchain1.chain:
   chain_data.append(block.__dict__)
print("Blockchain 1: " + str(chain_data))

#What is in Robot 2's blockchain?
chain_data = []
for block in blockchain2.chain:
   chain_data.append(block.__dict__)
print("Blockchain 2: " + str(chain_data))


#A malicious robot creates fake data.
fake_blockchain = Blockchain()
fake_blockchain.add_genesis_block()

fake_blockchain = fake_blockchain.fakeBlock(blockchain1)


#Robots 1 and 2 came into contact with the malicious robot.
if blockchain1.consensus(fake_blockchain):
   blockchain1 = copy.deepcopy(fake_blockchain)

if blockchain2.consensus(fake_blockchain):
   blockchain2 = copy.deepcopy(fake_blockchain)

#What is in Robot 1's blockchain?
chain_data = []
for block in blockchain1.chain:
   chain_data.append(block.__dict__)
print("Blockchain 1: " + str(chain_data))

#What is in Robot 2's blockchain?
chain_data = []
for block in blockchain2.chain:
   chain_data.append(block.__dict__)
print("Blockchain 2: " + str(chain_data))

#What is in the malicious robot's blockchain?
chain_data = []
for block in fake_blockchain.chain:
   chain_data.append(block.__dict__)
print("Fake Blockchain: " + str(chain_data))
"""


"""
#A malicious robot creates fake data.
fake_blockchain = copy.deepcopy(blockchain1)
last_block = blockchain1.last_block
new_block = Block(index = last_block.index + 1, data = "fake data", timestamp = time.time(), previous_hash = last_block.hash)
new_block_hash = new_block.compute_hash()
new_block.hash = new_block_hash
fake_blockchain.chain.append(new_block)
print(new_block_hash)


#Robots 1 and 2 came into contact with the malicious robot.
if blockchain1.consensus(fake_blockchain):
   blockchain1 = copy.deepcopy(fake_blockchain)

if blockchain2.consensus(fake_blockchain):
   blockchain2 = copy.deepcopy(fake_blockchain)

#What is in Robot 1's blockchain?
chain_data = []
for block in blockchain1.chain:
   chain_data.append(block.__dict__)
print("Blockchain 1: " + str(chain_data))

#What is in Robot 2's blockchain?
chain_data = []
for block in blockchain2.chain:
   chain_data.append(block.__dict__)
print("Blockchain 2: " + str(chain_data))

#What is in the malicious robot's blockchain?
chain_data = []
for block in fake_blockchain.chain:
   chain_data.append(block.__dict__)
print("Fake Blockchain: " + str(chain_data))
"""


"""
blockchain1.add_new_data("5, 6")
blockchain1.mine()

chain_data = []
for block in blockchain2.chain:
   chain_data.append(block.__dict__)
   
print(len(chain_data))
"""