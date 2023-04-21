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
contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle.sol:CentralizedOracle")
contract_bin = contract_interface.get('bin')
app = Flask(__name__)

@app.route('/deploy', methods=['POST'])
def deploy_contract():
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        print("-------------------------------------------------------------------------------------------------------------")
        data = request.get_json()
        print(data)
        INFURA_URL = "https://goerli.infura.io/v3/8e16cb87ad3f4ee19a3c24a3582daf8c"
        w3 = Web3(Web3.HTTPProvider(INFURA_URL))
        assert w3.is_connected()
        # PRIVATE_KEY = data.get('privatekey')
        ACCOUNT_ADDRESS = data.get('accountaddr')
        print(PRIVATE_KEY)
        print(ACCOUNT_ADDRESS)
        # PRIVATE_KEY = '0481b04ac17055a6ce23e5789e3c102838c69c65a4c6cd8fc6ecde86e2f5b600'
        # ACCOUNT_ADDRESS = '0x0ec58ddCd588b8666a267d08C6F1584b4633fdaE'
        selected_model_id = data.get('model')
        print(selected_model_id)
        if (selected_model_id== 1):
            compiled_sol = compile_files("./mysite/Oracle/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 2):
            compiled_sol = compile_files("./mysite/Oracle2/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle2/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 3):
            compiled_sol = compile_files("./mysite/Oracle3/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle3/contract/oracle.sol:CentralizedOracle")
        elif (selected_model_id== 4):
            compiled_sol = compile_files("./mysite/Oracle4/contract/oracle.sol", output_values=["abi", "bin-runtime", "bin"],)
            print(list(compiled_sol.keys()))
            contract_interface = compiled_sol.get("./mysite/Oracle4/contract/oracle.sol:CentralizedOracle")
        contract_bin = contract_interface.get('bin')
        model_api_endpoints = {
            1: "https://johnbartleydev.pythonanywhere.com/model1/predict",
            2: "https://johnbartleydev.pythonanywhere.com/model2/predict",
            3: "https://johnbartleydev.pythonanywhere.com/model3/predict",
            4: "https://johnbartleydev.pythonanywhere.com/model4/predict"
        }

        model_api_endpoint = model_api_endpoints[selected_model_id]

        # Deploy the contract
        nonce = w3.eth.get_transaction_count(ACCOUNT_ADDRESS)
        gas_estimate = w3.eth.estimate_gas({"from": ACCOUNT_ADDRESS, "data": contract_bin})
        transaction = {
          'from': ACCOUNT_ADDRESS,
          'data': contract_bin,
          'gas': gas_estimate,
          'gasPrice': w3.eth.gas_price,
          'nonce': nonce,
          'chainId': 5 # Goerli testnet chain ID
        }
        signed_txn = w3.eth.account.sign_transaction(transaction, PRIVATE_KEY)
        txn_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        # Wait for the transaction receipt
        transaction_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
        contract_address = transaction_receipt['contractAddress']
        print(f"Contract deployed at address: {contract_address}")
        return (contract_address)

if __name__ == '__main__':
    app.run()
