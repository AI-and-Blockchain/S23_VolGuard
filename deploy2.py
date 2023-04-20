import json
import os
from web3 import Web3
import solcx
from solcx import compile_source, compile_files
solcx.install_solc('0.5.7')
# Set up web3.py instance
INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
assert w3.isConnected()
# Replace the following with your private key and address
PRIVATE_KEY = 'YOUR-PRIVATE-KEY'
ACCOUNT_ADDRESS = 'YOUR-ACCOUNT-ADDRESS'
# Compile the Solidity contract

compiled_sol = compile_files("./Oracle/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
print(compiled_sol)
contract_interface = compiled_sol.get("bin")
# Deploy the contract
nonce = w3.eth.getTransactionCount(ACCOUNT_ADDRESS)
gas_estimate = w3.eth.estimateGas({"from": ACCOUNT_ADDRESS, "data": contract_interface})
transaction = {
  'from': ACCOUNT_ADDRESS,
  'data': contract_interface,
  'gas': gas_estimate,
  'gasPrice': w3.eth.gasPrice,
  'nonce': nonce,
  'chainId': 5 # Goerli testnet chain ID
}
signed_txn = w3.eth.account.signTransaction(transaction, PRIVATE_KEY)
txn_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)
# Wait for the transaction receipt
transaction_receipt = w3.eth.waitForTransactionReceipt(txn_hash)
contract_address = transaction_receipt['contractAddress']
print(f"Contract deployed at address: {contract_address}")
# '''