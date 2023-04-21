// SPDX-License-Identifier: Apache 2.0
pragma solidity ^0.8.0;

contract SimpleOracle {
    address public owner;
    uint256 public storedNumber;

    event NumberUpdated(uint256 indexed newNumber);

    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function getStoredNumber() public view returns (uint256) {
        return storedNumber;
    }

    function updateStoredNumber(uint256 _newNumber) public onlyOwner {
        storedNumber = _newNumber;
        emit NumberUpdated(_newNumber);
    }
}
