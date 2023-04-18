import React from "react";
import WalletConnect from "./WalletConnect";

function Header() {
  return (
    <header className="bg-gray-800 py-4 px-10 flex">
      <div className="container mx-auto">
        <h1 className="text-3xl font-semibold text-white">Volguard ⌱</h1>
      </div>
      <WalletConnect />
    </header>
  );
}

export default Header;
