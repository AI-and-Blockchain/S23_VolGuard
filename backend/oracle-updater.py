from flask import Flask
from web3 import Web3, HTTPProvider
import requests
from threading import Timer

app = Flask(__name__)

# Replace with your contract ABI and address
contract_abi = [...]
contract_address = '...'

# Replace with your Ethereum private key and Infura API key
private_key = 'YOUR_PRIVATE_KEY'
infura_api_key = 'YOUR_INFURA_API_KEY'

# Connect to Ethereum
w3 = Web3(HTTPProvider(f"https://mainnet.infura.io/v3/{infura_api_key}"))
contract = w3.eth.contract(address=Web3.toChecksumAddress(contract_address), abi=contract_abi)

# Replace with your API URL
api_url = 'https://johnbartleydev.pythonanywhere.com/model1/predict'

# Your Ethereum address
my_address = '...'

# Function to fetch data from API and update stored number in the contract
def update_stored_number():
    try:
        response = requests.get(api_url)
        new_number = int(float(response.text))

        gas_price = w3.eth.gasPrice
        gas_estimate = contract.functions.updateStoredNumber(new_number).estimateGas({'from': my_address})

        transaction = contract.functions.updateStoredNumber(new_number).buildTransaction({
            'from': my_address,
            'gas': gas_estimate,
            'gasPrice': gas_price,
            'nonce': w3.eth.getTransactionCount(my_address),
        })

        signed_tx = w3.eth.account.signTransaction(transaction, private_key)
        transaction_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)
        print(f"Transaction hash: {transaction_hash.hex()}")

        receipt = w3.eth.waitForTransactionReceipt(transaction_hash)
        if receipt['status'] == 0:
            print("Transaction failed.")

    except ValueError as ve:
        print(f"Error updating stored number: {ve}")
    except requests.exceptions.RequestException as re:
        print(f"Error fetching data from API: {re}")

    Timer(6 * 60 * 60, update_stored_number).start()

@app.route('/')
def index():
    update_stored_number()
    return "Updated stored number", 200

if __name__ == '__main__':
    app.run()
