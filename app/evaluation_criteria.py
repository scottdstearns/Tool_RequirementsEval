"""
Evaluation criteria for requirements based on INCOSE standards.
"""

# Standard INCOSE criteria for requirement evaluation
INCOSE_CRITERIA = """
1. COMPLETE: The requirement is fully stated in one place with no missing information. It doesn't require the reader to look for information in other requirements or documents.

2. CORRECT: The requirement accurately describes the system capability, characteristic, or constraint that will solve the customer's problem.

3. FEASIBLE: The requirement can be implemented within the constraints of the project, including technology, cost, and schedule constraints.

4. NECESSARY: The requirement defines an essential capability, characteristic, constraint, or quality factor. If it is removed, a deficiency would exist that cannot be fulfilled by other requirements.

5. SINGULAR: The requirement states a single capability, characteristic, or constraint. It does not combine multiple requirements.

6. UNAMBIGUOUS: The requirement is stated clearly and precisely, using terminology that is consistent and well-defined. It has only one interpretation.

7. VERIFIABLE: The requirement is stated in such a way that it can be verified that the requirement has been met in the finished system. It can be validated through inspection, analysis, demonstration, or test.

8. TRACEABLE: The requirement can be traced to higher-level requirements, customer needs, or other source documents.

9. CONSISTENT: The requirement does not contradict other requirements and is fully consistent with all authoritative external documentation.

10. INTERFACE DEFINED: If the requirement describes an interface to other systems, it specifies the interface completely.
"""

# Simplified criteria set for initial proof-of-concept
SIMPLIFIED_CRITERIA = """
1. COMPLETE: The requirement has no missing information and doesn't require looking elsewhere.

2. UNAMBIGUOUS: The requirement has only one possible interpretation and uses consistent, well-defined terminology.

3. SINGULAR: The requirement states only one thing and doesn't combine multiple requirements.

4. VERIFIABLE: The requirement can be verified through testing, inspection, or other means.

5. FEASIBLE: The requirement can be implemented with available technology and resources.
"""

# Default criteria to use in the application
DEFAULT_CRITERIA = SIMPLIFIED_CRITERIA 
