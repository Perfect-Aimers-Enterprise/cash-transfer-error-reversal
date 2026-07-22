def get_error_details(prediction):

    rules = {

        "Successful Transfer": {
            "reverse": False,
            "action": "No Action Required",
            "reason": "Transaction completed successfully.",
            "severity": "None",
        },


        "Partial Processing": {
            "reverse": True,
            "action": "Reverse Transaction",
            "reason": "Debit completed but credit processing failed.",
            "severity": "High",
        },


        "Invalid Account": {
            "reverse": True,
            "action": "Reverse Transaction",
            "reason": "Destination account information is invalid.",
            "severity": "High",
        },


        "Insufficient Funds": {
            "reverse": False,
            "action": "Reject Transaction",
            "reason": "Sender account does not have sufficient balance.",
            "severity": "Medium",
        },


        "Beneficiary Mismatch": {
            "reverse": True,
            "action": "Reverse Transaction",
            "reason": "Beneficiary details do not match transaction information.",
            "severity": "High",
        },


        "Pending Settlement": {
            "reverse": False,
            "action": "Monitor Transaction",
            "reason": "Transaction is awaiting settlement confirmation.",
            "severity": "Low",
        },

    }


    return rules.get(
        prediction,
        {
            "reverse": True,
            "action": "Manual Review",
            "reason": "Unknown transaction error.",
            "severity": "Medium",
        }
    )