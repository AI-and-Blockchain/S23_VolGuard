require('dotenv').config();
const Web3 = require('web3');
const axios = require('axios');
const { abi } = require('./CentralizedOracleABI.json'); // ABI file of the CentralizedOracle contract

const INFURA_PROJECT_ID = process.env.INFURA_PROJECT_ID;
const PRIVATE_KEY = process.env.PRIVATE_KEY;
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;

const providerUrl = `https://mainnet.infura.io/v3/${INFURA_PROJECT_ID}`;
const web3 = new Web3(new Web3.providers.HttpProvider(providerUrl));
const contract = new web3.eth.Contract(abi, CONTRACT_ADDRESS);
const account = web3.eth.accounts.privateKeyToAccount(PRIVATE_KEY);
web3.eth.accounts.wallet.add(account);
web3.eth.defaultAccount = account.address;

async function fetchData(url, attribute) {
  try {
    const response = await axios.get(url);
    return response.data[attribute];
  } catch (error) {
    console.error(`Error fetching data: ${error}`);
    return null;
  }
}

async function updateRequest(requestId, result) {
  const gasPrice = await web3.eth.getGasPrice();
  const gasEstimate = await contract.methods.updateRequest(requestId, result).estimateGas();

  const tx = {
    from: account.address,
    to: CONTRACT_ADDRESS,
    gas: gasEstimate,
    gasPrice: gasPrice,
    data: contract.methods.updateRequest(requestId, result).encodeABI(),
  };

  try {
    const signedTx = await web3.eth.accounts.signTransaction(tx, PRIVATE_KEY);
    const receipt = await web3.eth.sendSignedTransaction(signedTx.rawTransaction);
    console.log(`Transaction hash: ${receipt.transactionHash}`);
  } catch (error) {
    console.error(`Error sending transaction: ${error}`);
  }
}

contract.events.NewRequest(async (error, event) => {
  if (error) {
    console.error(`Error: ${error}`);
    return;
  }

  const { id, urlToQuery, attributeToFetch } = event.returnValues;
  console.log(`New request received. ID: ${id}, URL: ${urlToQuery}, Attribute: ${attributeToFetch}`);

  const result = await fetchData(urlToQuery, attributeToFetch);
  if (result) {
    await updateRequest(id, result);
  } else {
    console.error('Failed to fetch data or update request');
  }
});
