import React, { useContext } from "react";
import Web3Modal from "web3modal";
import Web3 from "web3";
import WalletContext from "./WalletContext";

const WalletConnect = () => {
  const { connected, setConnected, account, setAccount } = useContext(WalletContext);

  const connectWallet = async () => {
    try {
      const web3Modal = new Web3Modal({
        network: "goerli",
        cacheProvider: true,
      });
  
      const provider = await web3Modal.connect();
      const web3 = new Web3(provider);
  
      // Check if the connected network is Goerli testnet
      const networkId = await web3.eth.net.getId();
      if (networkId !== 5) {
        // Display an error modal or message to inform the user to switch to the Goerli testnet
        // Disconnect the wallet provider
        await web3Modal.clearCachedProvider();
        await provider.disconnect?.();
  
        return;
      }
  
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
