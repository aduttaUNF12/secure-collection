# All classes are within one file, I'll clean this up later.
# Currently I have this file using multithreading, I can remove this for the sake of simplicity.
import hashlib
import time
import random
import sys
import time
import threading

#These are used to simulate communication between bots.
committeeSelections = []
blockProposerSelections = []
blockProposerReadyToReceiveData = False
dataForNewBlock = []

class Block:
    def __init__(self, index, data, timestamp, previous_hash, nonce, contributor):
        self.index = index
        self.data = data
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.contributor = contributor
        self.hash = self.compute_hash()

    def compute_hash(self):
        string_to_hash = str(self.index) + str(self.data) + str(self.timestamp) + str(self.previous_hash) + str(
            self.nonce)
        return hashlib.sha256(string_to_hash.encode()).hexdigest()


class Bot:

    def __init__(self, blockchain, publicKeys, privateSeed, committeeSize, personalPK):
        self.blockchain = blockchain
        self.algorithms = Algorithms()
        self.l = 1  # lookback parameter
        self.publicKeys = publicKeys
        self.personalPK = personalPK
        self.privateSeed = privateSeed
        self.committeeSize = committeeSize
        self.epoch = 1

    """This method is being used to test all of the methods in conjunction with each other."""
    def proof_of_stake(self):
        # print("Bot " + str(self.personalPK) + ": ")

        inCommittee = False
        isBlockProposer = False
        global committeeSelections
        global blockProposerSelections
        global blockProposerReadyToReceiveData

        committeeSelections.clear()
        blockProposerSelections.clear()
        blockProposerReadyToReceiveData = False
        dataForNewBlock.clear()

        # Each bot gets the committee selection from its extract_committee algorithm
        committee = self.algorithms.extract_committee(self.blockchain.chain, self.l, self.publicKeys, committeeSize,
                                                      self.privateSeed, self.epoch)
        print("Bot", self.personalPK, "committee selection:", committee)

        # Used to simulate each bot taking a slightly varying amount of time to send their committee selection
        time.sleep(random.random())

        # Share committee selection with other bots
        committeeSelections.append(committee)

        while len(committeeSelections) < len(self.publicKeys):
            time.sleep(0.1)
        print("Bot", self.personalPK, "committees received from other bots:", committeeSelections)

        for selection in committeeSelections:
            if selection != committee:
                print("Malicious intent suspected")

        # If respective bot is in the committee, seed random oracle to find block-proposer
        if self.personalPK in committee:
            inCommittee = True

        if inCommittee:
            blockProposer = self.algorithms.oracle(self.blockchain.chain, committee, self.privateSeed, self.epoch)
            print("Bot", self.personalPK, "block-proposer selection:", blockProposer)
            time.sleep(random.random())

            # Send block-proposer selection to other bots
            blockProposerSelections.append(blockProposer)

            # Wait until all block-proposer selections are received
            while len(blockProposerSelections) < len(committee):
                time.sleep(0.1)

            # Verify incoming block-proposer selections
            for selection in blockProposerSelections:
                if selection != blockProposer:
                    print("Malicious intent suspected")

            # create_block(self, chain, contributor, data, privateSeed):

            # Block proposer creates new block
            if blockProposer == self.personalPK:
                isBlockProposer = True
                # newBlock = self.algorithms.create_block(self.blockchain, self.personalPK, "data...", self.privateSeed)

        if isBlockProposer:
            blockProposerReadyToReceiveData = True
            while len(dataForNewBlock) < len(self.publicKeys) - 1:
                time.sleep(0.1)
            newData = {data[0]: data[1] for data in dataForNewBlock}
            print(newData)

        else:
            time.sleep(random.random())
            while not blockProposerReadyToReceiveData:
                time.sleep(1)
            # Block will put its newly collected data here instead of "*****"
            dataForNewBlock.append((self.personalPK, "*****"))

        # Block-proposer sends new block to other bots

        # Other bots verify block and add to chain

        # At the end of the epoch, the epoch number is increased by 1
        self.epoch += 1


class Algorithms:
    """Will select a committee randomly by seeding a random number generator using a private seed plus the epoch
    number. Committee can be any size less than the number of bots."""

    def extract_committee(self, chain, l, publicKeys, committeeSize, privateSeed, epoch):

        """Block contribution weight and data contribution weight are variables that can be experimented with.
        Putting too much weight on the number of blocks contributed may be problematic because it may allow a
        particular node to gain too much control over contributing blocks. However, this problem is most likely
        solved by the fact that the random oracle (block-proposer selection algorithm) has high entropy, which
        means that all bots in the committee have an equal chance of being selected as block-proposer."""
        blockContributionWeight = 0

        """An upper bound for how much data can be collected per timestep may need to be derived so that malicious
        bots cannot contribute an unrealistically large amount of data to the chain. This would give them a much
        higher chance of being selected as committtee members, but they would still have the same chance of being 
        selected as a block-proposer by the random oracle as the rest of the committee members."""
        dataContributionWeight = 1 - blockContributionWeight

        """Dictionary in the format:
            contributions = {
                publicKey1: [quantity of blocks contributed, quantity of data contributed]
                publicKey2: [quantity of blocks contributed, quantity of data contributed]
                etc...
            }
        """
        contributions = {i: [0, 0] for i in publicKeys}

        # prefix chain starting at 2l blocks back where l is a lookback parameter (number of blocks)
        prefixChain = chain[:(-2 * l) + 1]

        # Records the difference between the newest and oldest blocks in this prefix chain: chain[:-2l]
        oldestBlockTimestamp = prefixChain[0].timestamp
        newestBlockTimestamp = prefixChain[-1].timestamp
        timestampDifference = newestBlockTimestamp - oldestBlockTimestamp

        for block in prefixChain:

            # Older data will have a higher age weight applied to it
            if timestampDifference > 0:
                ageWeight = 1 - (block.timestamp - oldestBlockTimestamp) / timestampDifference
            else:
                ageWeight = 1

            # Adds 1 * ageWeight for every block that a bot has contributed
            if block.contributor is not None:
                contributions[block.contributor][0] += 1 * ageWeight

            # Adds the number of bytes of data times the ageWeight of the data that a bot has contributed
            for pk in publicKeys:
                contributions[pk][1] += sys.getsizeof(block.data[pk]) * ageWeight

        # For tracking the maximum quantity of data and number of blocks contributed by each bot
        maxBlocks = 0
        maxData = 0
        for pk in publicKeys:
            if contributions[pk][0] > maxBlocks:
                maxBlocks = contributions[pk][0]

            if contributions[pk][1] > maxData:
                maxData = contributions[pk][1]

        # Normalize and apply weights to data for each bot
        if maxBlocks > 0:
            for pk in publicKeys:
                contributions[pk][0] /= maxBlocks
                contributions[pk][0] *= blockContributionWeight

        for pk in publicKeys:
            contributions[pk][1] /= maxData
            contributions[pk][1] *= dataContributionWeight

        """List in the format:
            totalContributionsWeighted = {
                publicKey1: weighted contribution
                publicKey2: weighted contribution
                etc...
            }
        """
        totalContributionsWeighted = {i: 0 for i in publicKeys}

        # Combine normalized blocks contributed and data contributed into one value
        for pk in publicKeys:
            totalContributionsWeighted[pk] = sum(contributions[pk])

        # Gets the total weight of all contributions combined, used to obtain committee selection probabilities
        totalContributionsSum = sum(totalContributionsWeighted.values())

        # Calculate probability of each bot being selected into the committee
        for pk in publicKeys:
            totalContributionsWeighted[pk] /= totalContributionsSum

        """Seed randomness by using the shared privateSeed plus the number of the current epoch, this will
        ensure that the seed is different on every epoch"""
        random.seed(privateSeed + epoch)

        # List where the committee will be stored
        committee = []

        """Randomly selects the committee. Bots who have contributed more data have a higher chance of being
        selected for the committee, but are not guaranteed to be selected."""
        while len(committee) < committeeSize:
            selection = random.choices(publicKeys, weights=totalContributionsWeighted.values(), k=1)[0]

            if selection not in committee:
                committee.append(selection)

        # print("Committee:", committee)

        return committee

    """Random oracle will select a block-proposer out of a list of committee bots. Block-proposer selection is
       seeded using a combination of all of the nonces in the chain, the private seed, and the epoch number.
       Randomness is high-entropy. All bots in the committee have an equal chance of becoming the block-proposer."""

    def oracle(self, chain, committee, privateSeed, epoch):

        concatenatedNonces = ""

        """The concatenated nonce can be stored so that each time the oracle is queried only one 
        append is needed, which will make the amortized runtime O(n) where n is the final length
        of the blockchain"""
        for block in chain:
            concatenatedNonces += str(block.nonce)

        """Seed the random oracle with the concatenated nonces (known by all bots on network),
           plus the privatSeed"""
        random.seed(concatenatedNonces + str(privateSeed) + str(epoch))

        """The block proposer is selected from committee bots with high entropy random selection. 
        Committee bots who contributed more data do NOT have a higher chance of being selected 
        to be block-proposer."""
        blockProposerSelection = random.choices(committee, k=1)
        # print("Block-Proposer Selection:", blockProposerSelection)

        return blockProposerSelection[0]

    """The block-proposer will generate this when it creates the new block. This will be used to verify
       that the node who generated the block is honest. Since only honest nodes have the private seed,
       other honest nodes will be able to verify that the nonce that was generated is valid by seeding
       their own identical local nonce_generator algorithm with the private seed and the timestamp on 
       the block."""

    def nonce_generator(self, privateSeed, timestamp):
        random.seed(privateSeed + timestamp)
        return random.randint(0, 21474836470)

    # The block-proposer will use this to create the block.
    def create_block(self, chain, contributor, data, privateSeed):
        timestamp = time.time()
        previousHash = chain[-1].hash
        index = len(chain)
        nonce = self.nonce_generator(privateSeed, timestamp)
        return Block(index, data, timestamp, previousHash, nonce, contributor)


class Blockchain:
    difficulty = 2

    def __init__(self, initialDataList):
        self.unconfirmed_data = []
        self.chain = []

        for data in initialDataList:
            self.add_genesis_block(data)

    def add_genesis_block(self, data):
        # Needs to handle different cases
        genesis_block = Block(0, data, time.time(), "Chancellor on the brink...", random.randint(1, 21474836470), None)
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)
        # print(genesis_block.hash)

    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, block, proof):
        previous_hash = self.last_block.hash
        print(proof)

        if previous_hash != block.previous_hash:
            return False

        if not Blockchain.is_valid_proof(block, proof):
            return False

        block.hash = proof
        self.chain.append(block)
        return True

    """@staticmethod
    def proof_of_work(block):
        block.nonce = 0
        # print(block.nonce)

        computed_hash = block.compute_hash()

        while not computed_hash.startswith('0' * Blockchain.difficulty):
            block.nonce += 1
            # print(block.nonce)
            computed_hash = block.compute_hash()

        return computed_hash"""

    def extract_committee(self):
        pass

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
            # print("Validating: " + block.hash)
            # delattr(block, "hash")

            if (not cls.is_valid_proof(block,
                                       block_hash) or previous_hash != block.previous_hash) and previous_hash != "Chancellor on the brink...":
                result = False
                print("Rejecting")
                break

            block.hash, previous_hash = block_hash, block_hash

        # print(result)

        return result

    def hiTestBye(self):
        return "Hello"

    def mine(self):
        if not self.unconfirmed_data:
            return False

        last_block = self.last_block

        new_block = Block(index=last_block.index + 1, data=self.unconfirmed_data, timestamp=time.time(),
                          previous_hash=last_block.hash)

        proof = self.proof_of_work(new_block)
        self.add_block(new_block, proof)

        self.unconfirmed_data = []

        return True

    def consensus(self, new_blockchain):

        current_len = len(self.chain)
        length = len(new_blockchain.chain)

        # print(length)
        # print(current_len)

        if length > current_len and self.check_chain_validity(new_blockchain.chain):
            print("Accept")
            return True

        print("Reject")
        return False

    """
    def verify_and_add_block():
       block_data = request.get_json()
       block = Block(block_data["index"], block_data["transactions"], block_data["timestamp"], block_data["previous_hash"], block_data["nonce"])

       proof = block_data['hash']
       added = blockchain.add_block(block, proof) 
    """


# Robots 1 and 2

pk1 = 1
pk2 = 2
pk3 = 3

publicKeys = [pk1, pk2, pk3]

initialDataGenesisBlock1 = {
    pk1: "*",
    pk2: "**",
    pk3: "iiiiiiiiiwwwwwwwwwwwwwwww"
}

initialDataGenesisBlock2 = {
    pk1: "x",
    pk2: "xx",
    pk3: "wwwwwwwwwwwwwwwwwwwwwwwwww"
}

initialDataList = [initialDataGenesisBlock1, initialDataGenesisBlock2]

blockchain = Blockchain(initialDataList)

privateSeed = random.randint(1, 100)

committeeSize = 2

bot1 = Bot(blockchain, publicKeys, privateSeed, committeeSize, pk1)
bot2 = Bot(blockchain, publicKeys, privateSeed, committeeSize, pk2)
bot3 = Bot(blockchain, publicKeys, privateSeed, committeeSize, pk3)

t1 = threading.Thread(target=bot1.proof_of_stake, args=())
t2 = threading.Thread(target=bot2.proof_of_stake, args=())
t3 = threading.Thread(target=bot3.proof_of_stake, args=())

t1.start()
t2.start()
t3.start()


# Robot 1 finds new data
"""blockchain1.add_new_data("1, 2, 3, 4")
blockchain1.mine()

# Robots 1 and 2 came into contact
if blockchain1.consensus(blockchain2):
    blockchain1 = copy.deepcopy(blockchain2)

if blockchain2.consensus(blockchain1):
    blockchain2 = copy.deepcopy(blockchain1)

# What is in Robot 1's blockchain?
chain_data = []
for block in blockchain1.chain:
    chain_data.append(block.__dict__)
print("Blockchain 1: " + str(chain_data))

# What is in Robot 2's blockchain?
chain_data = []
for block in blockchain2.chain:
    chain_data.append(block.__dict__)
print("Blockchain 2: " + str(chain_data))

# A malicious robot creates fake data.
fake_blockchain = copy.deepcopy(blockchain1)
last_block = blockchain1.last_block
new_block = Block(index=last_block.index + 1, data="fake data", timestamp=time.time(), previous_hash=last_block.hash)
new_block_hash = new_block.compute_hash()
new_block.hash = new_block_hash
fake_blockchain.chain.append(new_block)
print(new_block_hash)

# Robots 1 and 2 came into contact with the malicious robot.
if blockchain1.consensus(fake_blockchain):
    blockchain1 = copy.deepcopy(fake_blockchain)

if blockchain2.consensus(fake_blockchain):
    blockchain2 = copy.deepcopy(fake_blockchain)

# What is in Robot 1's blockchain?
chain_data = []
for block in blockchain1.chain:
    chain_data.append(block.__dict__)
print("Blockchain 1: " + str(chain_data))

# What is in Robot 2's blockchain?
chain_data = []
for block in blockchain2.chain:
    chain_data.append(block.__dict__)
print("Blockchain 2: " + str(chain_data))

# What is in the malicious robot's blockchain?
chain_data = []
for block in fake_blockchain.chain:
    chain_data.append(block.__dict__)
print("Fake Blockchain: " + str(chain_data))"""

"""
blockchain1.add_new_data("5, 6")
blockchain1.mine()

chain_data = []
for block in blockchain2.chain:
   chain_data.append(block.__dict__)

print(len(chain_data))
"""