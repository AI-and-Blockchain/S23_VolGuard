pragma solidity >=0.4.21 <0.6.0;

contract SimpleOracle {
    string public urlToQuery;
    string public result;
    address public oracleAddress;

    event NewRequest(string urlToQuery);
    event UpdatedRequest(string urlToQuery, string result);

    constructor(address _oracleAddress, string memory _urlToQuery) public {
        oracleAddress = _oracleAddress;
        urlToQuery = _urlToQuery;
    }

    function updateResult(string memory _result) public {
        require(msg.sender == oracleAddress, "Only the designated oracle can update the request.");
        result = _result;
        emit UpdatedRequest(urlToQuery, result);
    }
}
