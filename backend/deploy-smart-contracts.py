import json
import os
from flask import Flask, request
from web3 import Web3
import solcx
from solcx import compile_source, compile_files
solcx.install_solc('0.5.7')
# Set up web3.py instance
INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
assert w3.isConnected()

# Compile the Solidity contract
compiled_sol = compile_files("./Oracle/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
contract_interface = compiled_sol.get("bin")

app = Flask(__name__)

@app.route('/deploy', methods=['POST'])
def deploy_contract():
    data = request.get_json()

    PRIVATE_KEY = data.get('privatekey')
    ACCOUNT_ADDRESS = data.get('accountaddr')
    selected_model_id = data.get('model')

    model_api_endpoints = {
        1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
        2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
        3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
        4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
    }

    model_api_endpoint = model_api_endpoints[selected_model_id]

    # Deploy the contract
    nonce = w3.eth.getTransactionCount(ACCOUNT_ADDRESS)
    gas_estimate = w3.eth.estimateGas({"from": ACCOUNT_ADDRESS, "data": contract_interface})
    transaction = {
        'from': ACCOUNT_ADDRESS,
        'data': contract_interface,
        'gas': gas_estimate,
        'gasPrice': w3.eth.gasPrice,
        'nonce': nonce,
        'chainId': 5  # Goerli testnet chain ID
    }
    signed_txn = w3.eth.account.signTransaction(transaction, PRIVATE_KEY)
    txn_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)
    # Wait for the transaction receipt
    transaction_receipt = w3.eth.waitForTransactionReceipt(txn_hash)
    contract_address = transaction_receipt['contractAddress']
    return {"contract_address": contract_address}

if __name__ == '__main__':
    app.run()
