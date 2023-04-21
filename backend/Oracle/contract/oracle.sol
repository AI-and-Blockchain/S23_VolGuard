pragma solidity >=0.4.21 <0.6.0;

contract CentralizedOracle {
    Request[] public requests;
    uint public currentRequestId = 0;
    address public oracleAddress;

    struct Request {
        uint id;
        string urlToQuery;
        string attributeToFetch;
        string result;
    }

    event NewRequest(uint id, string urlToQuery, string attributeToFetch);
    event UpdatedRequest(uint id, string urlToQuery, string attributeToFetch, string result);

    constructor(address _oracleAddress) public {
        oracleAddress = _oracleAddress;
    }

    function createRequest(string memory _urlToQuery, string memory _attributeToFetch) public {
        uint index = requests.push(Request(currentRequestId, _urlToQuery, _attributeToFetch, "")) - 1;
        emit NewRequest(currentRequestId, _urlToQuery, _attributeToFetch);
        currentRequestId++;
    }

    function updateRequest(uint _requestId, string memory _result) public {
        require(msg.sender == oracleAddress, "Only the designated oracle can update the request.");
        Request storage requestToUpdate = requests[_requestId];
        requestToUpdate.result = _result;
        emit UpdatedRequest(requestToUpdate.id, requestToUpdate.urlToQuery, requestToUpdate.attributeToFetch, requestToUpdate.result);
    }
}
