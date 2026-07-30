def get_error_details(prediction):

    rules = {

        "No Error": {
            "reverse": False,
            "action": "No Action Required",
            "reason": "No transaction error detected.",
            "severity": "None",
        },

        "Duplicate Transaction": {
            "reverse": True,
            "action": "Reverse Duplicate Transaction",
            "reason": "The same transaction appears to have been processed more than once.",
            "severity": "High",
        },

        "Incorrect Amount Entry": {
            "reverse": True,
            "action": "Reverse and Correct Amount",
            "reason": "The transaction amount differs from the intended amount.",
            "severity": "High",
        },

        "Pending Review": {
            "reverse": False,
            "action": "Hold for Investigation",
            "reason": "The transaction requires manual verification before further action.",
            "severity": "Medium",
        },

        "Technical Glitch / Timeout": {
            "reverse": True,
            "action": "Retry or Reverse Transaction",
            "reason": "A processing timeout or technical failure occurred during the transaction.",
            "severity": "High",
        },

        "Unauthorized Reversal Request": {
            "reverse": False,
            "action": "Reject Request",
            "reason": "The reversal request is not authorized and requires security verification.",
            "severity": "Critical",
        },

        "Wrong Beneficiary": {
            "reverse": True,
            "action": "Reverse Transaction",
            "reason": "Funds were sent to an unintended beneficiary.",
            "severity": "Critical",
        },

    }

    return rules.get(
        prediction,
        {
            "reverse": False,
            "action": "Manual Review",
            "reason": "Unknown transaction status.",
            "severity": "Medium",
        },
    )