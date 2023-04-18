import React, { useState } from "react";
import Web3Modal from "web3modal";
import Web3 from "web3";
import { Wallet } from '@heroicons/react/20/solid';


const WalletConnect = () => {
  const [connected, setConnected] = useState(false);
  const [account, setAccount] = useState("");

  const connectWallet = async () => {
    try {
      const web3Modal = new Web3Modal({
        network: "goerli",
        cacheProvider: true,
      });
  
      const provider = await web3Modal.connect();
      const web3 = new Web3(provider);
  
      const accounts = await web3.eth.getAccounts();
      setAccount(accounts[0]);
      setConnected(true);
    } catch (error) {
      console.error("Error connecting wallet:", error.message);
      // You can update the UI with the error message or a user-friendly message here.
    }
  };
  

  return (
    <button
      onClick={connectWallet}
      className="max-w-3xl py-2 px-4 bg-gray-600 hover:bg-gray-500 text-white font-semibold rounded-md whitespace-nowrap"
    >
      {connected ? `Connected: ${account}` : "Connect Wallet"} 
    </button>
  );
};

export default WalletConnect;
