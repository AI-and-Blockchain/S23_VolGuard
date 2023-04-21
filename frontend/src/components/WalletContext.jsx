import { createContext, useState } from "react";

const WalletContext = createContext();

export const WalletProvider = ({ children }) => {
  const [connected, setConnected] = useState(false);
  const [account, setAccount] = useState("");

  return (
    <WalletContext.Provider value={{ connected, setConnected, account, setAccount }}>
      {children}
    </WalletContext.Provider>
  );
};

export default WalletContext;
