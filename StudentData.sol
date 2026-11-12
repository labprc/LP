
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract StudentData {

    // Structure to store student details
    struct Student {
        uint rollNo;
        string name;
        uint age;
        string department;
    }

    // Array to store students
    Student[] public students;

    // Counter to track fallback calls
    uint public fallbackCount;

    // Add a new student
    function addStudent(
        uint _rollNo,
        string memory _name,
        uint _age,
        string memory _department
    ) public {

        students.push(
            Student(_rollNo, _name, _age, _department)
        );
    }

    // Get student details using array index
    function getStudent(uint index)
        public
        view
        returns (uint, string memory, uint, string memory)
    {
        require(index < students.length, "Invalid student index");

        Student memory s = students[index];

        return (
            s.rollNo,
            s.name,
            s.age,
            s.department
        );
    }

    // Get total number of students
    function getStudentCount() public view returns (uint) {
        return students.length;
    }

    // Receive Ether without data
    receive() external payable {
    }

    // Handle unmatched function calls
    fallback() external payable {
        fallbackCount++;
    }
}
