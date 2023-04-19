from brownie import Contract, web3, ENS
from brownie.network.account import Accounts
def main():

	ens = ENS.from_env()
	name = "oracle.sol"
	address = ens.address(name)
	if address is None:
		print(f"The ENS name '{name}' is not set")
	else:
		tx = ens.setAddr(name,"0x0000000000000000000000000000000000000000", {'from': Accounts[0]})
		print(f"Transaction hash: {tx.hash}")
	my_contract = Contract.from_abi("contracts", "oracle.sol", Accounts[0])
	my_contract.deploy({'from': Accounts[0]})
